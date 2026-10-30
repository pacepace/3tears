"""What a SQL statement does, read from its text: the label the L3 broker runs it by, and whether it may be resent.

Labelled for every statement a pod sends, on its event loop, so a plain statement (a page's generated SQL) is read in a
few searches and any other from its tokens.
"""

from __future__ import annotations

import functools
import re
from typing import Final

__all__ = ["is_resendable", "statement_operation"]


#: a dollar-quote opener: ``$$`` or ``$tag$``. ``$1`` is a parameter, not a quote.
_DOLLAR_QUOTE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")

#: an unquoted word: a keyword or an identifier.
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9$]*")

#: the verbs a statement's label is drawn from, by the word that opens its main clause.
_MAIN_VERBS = {"SELECT": "select", "INSERT": "insert", "UPDATE": "update", "DELETE": "delete"}


def _end_of_quoted(query: str, start: int, quote: str, *, backslash_escapes: bool) -> int:
    """the index just past a quoted literal or identifier opening at ``start``.

    :param query: the statement
    :ptype query: str
    :param start: the index of the opening quote
    :ptype start: int
    :param quote: the quote character, doubled inside the text to escape it
    :ptype quote: str
    :param backslash_escapes: whether a backslash escapes the next character
        (an ``E'...'`` string)
    :ptype backslash_escapes: bool
    :return: the index after the closing quote, or the end of the text when unclosed
    :rtype: int
    """
    index = start + 1
    end = len(query)
    while index < end:
        char = query[index]
        if backslash_escapes and char == "\\":
            index += 2
        elif char == quote and query.startswith(quote * 2, index):
            index += 2
        elif char == quote:
            end = index
        else:
            index += 1
    return min(end + 1, len(query))


def _end_of_block_comment(query: str, start: int) -> int:
    """the index just past a block comment opening at ``start``; Postgres nests them.

    :param query: the statement
    :ptype query: str
    :param start: the index of the opening ``/*``
    :ptype start: int
    :return: the index after the matching ``*/``, or the end of the text when unclosed
    :rtype: int
    """
    depth = 0
    index = start
    result = len(query)
    while index < len(query) and result == len(query):
        if query.startswith("/*", index):
            depth += 1
            index += 2
        elif query.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                result = index
        else:
            index += 1
    return result


def _code_tokens(query: str) -> list[str]:
    """the statement's code as tokens: upper-cased words, ``(``, ``)``, ``,`` and other marks.

    literals, quoted identifiers, dollar-quoted text and comments are consumed
    whole, so a parenthesis or a keyword inside one is never read as code. A quoted
    identifier is one ``"`` token: it is a name, never a keyword.

    :param query: the statement
    :ptype query: str
    :return: the tokens, in order
    :rtype: list[str]
    """
    tokens: list[str] = []
    index = 0
    while index < len(query):
        char = query[index]
        dollar = _DOLLAR_QUOTE_RE.match(query, index)
        word = _WORD_RE.match(query, index)
        if char.isspace():
            index += 1
        elif query.startswith("--", index):
            newline = query.find("\n", index)
            index = len(query) if newline == -1 else newline + 1
        elif query.startswith("/*", index):
            index = _end_of_block_comment(query, index)
        elif char == "'":
            index = _end_of_quoted(query, index, "'", backslash_escapes=False)
            tokens.append("'")
        elif char in "Ee" and query.startswith("'", index + 1):
            index = _end_of_quoted(query, index + 1, "'", backslash_escapes=True)
            tokens.append("'")
        elif char == '"':
            index = _end_of_quoted(query, index, '"', backslash_escapes=False)
            tokens.append('"')
        elif dollar is not None:
            closing = query.find(dollar.group(0), dollar.end())
            index = len(query) if closing == -1 else closing + len(dollar.group(0))
            tokens.append("'")
        elif word is not None:
            tokens.append(word.group(0).upper())
            index = word.end()
        else:
            tokens.append(char)
            index += 1
    return tokens


def _after_parens(tokens: list[str], start: int) -> int:
    """the index just past the parenthesised group opening at ``start``.

    :param tokens: the statement's tokens
    :ptype tokens: list[str]
    :param start: the index of a ``(`` token
    :ptype start: int
    :return: the index after its matching ``)``, or past the end when unclosed
    :rtype: int
    """
    depth = 0
    index = start
    result = len(tokens) + 1
    while index < len(tokens) and result > len(tokens):
        if tokens[index] == "(":
            depth += 1
        elif tokens[index] == ")":
            depth -= 1
            if depth == 0:
                result = index + 1
        index += 1
    return result


def _main_clause_index(tokens: list[str]) -> int:
    """the index of the token opening the statement's main clause, past any ``WITH`` list.

    ``WITH [RECURSIVE] name [(columns)] AS [NOT] [MATERIALIZED] (body) [, ...]``
    is walked by its grammar, so a CTE whose name is a verb (``WITH update AS ...``)
    is read as the name it is.

    :param tokens: the statement's tokens
    :ptype tokens: list[str]
    :return: the index of the main clause's first token; past the end when the
        ``WITH`` list does not close
    :rtype: int
    """
    index = 0
    if tokens[:1] == ["WITH"]:
        index = 2 if tokens[1:2] == ["RECURSIVE"] else 1
        more = True
        while more and index < len(tokens):
            index += 1  # the CTE's name
            if tokens[index : index + 1] == ["("]:
                index = _after_parens(tokens, index)
            index += 1 if tokens[index : index + 1] == ["AS"] else 0
            index += 1 if tokens[index : index + 1] == ["NOT"] else 0
            index += 1 if tokens[index : index + 1] == ["MATERIALIZED"] else 0
            if tokens[index : index + 1] == ["("]:
                index = _after_parens(tokens, index)
            more = tokens[index : index + 1] == [","]
            index += 1 if more else 0
    return index


#: the statements whose operation is remembered: a bulk write sends one statement text a page after a page
_DETECTED_STATEMENTS = 64


#: what keeps a statement off :func:`_plain_operation`: a literal, a dollar-quoted text or a comment, inside which
#: a keyword or a mark is not code
_NOT_PLAIN_RE = re.compile(r"'|--|/\*|\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")

#: a quoted identifier, unclosed to the end: one name, never a keyword
_QUOTED_NAME_RE = re.compile(r'"(?:""|[^"])*"?')

#: a plain statement's first token, when it is a word
_FIRST_WORD_RE = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9$]*)?")

#: ``ON CONFLICT`` as two code words; what precedes it is read in :func:`_plain_operation`
_ON_CONFLICT_RE = re.compile(r"ON\s+CONFLICT(?![A-Za-z_0-9$])", re.IGNORECASE)


def _plain_operation(query: str) -> str | None:
    """:func:`statement_operation` for a plain statement, in a few C-level searches: None for any other.

    A page's multi-row ``INSERT`` is thousands of placeholders, and each page's text differs by its rows, so no
    cache holds it; read a character at a time in Python it held the event loop on every page (cobalt dev,
    2026-10-29: 58 of 70 stalls of a reports pod rebuilding its copy). A statement with no literal, dollar-quoted
    text or comment has only words, names and marks, so its main verb is its first word, and ``ON CONFLICT`` is
    code wherever it stands as two words. A ``WITH`` list, or an ``ON`` whose word begins before it, is left to the
    tokens.

    :param query: SQL query string
    :ptype query: str
    :return: the label; None when the statement is not plain
    :rtype: str | None
    """
    result: str | None = None
    if _NOT_PLAIN_RE.search(query) is None:
        code = _QUOTED_NAME_RE.sub('"', query)
        head = _FIRST_WORD_RE.match(code)
        verb = "" if head is None or head.group(1) is None else head.group(1).upper()
        if verb != "WITH":
            result = _MAIN_VERBS.get(verb, "select")
            if result == "insert":
                found = [match.start() for match in _ON_CONFLICT_RE.finditer(code)]
                # an ON whose word began before it ("xON") is no keyword: such a statement goes to the tokens
                joined = [start for start in found if start and (code[start - 1].isalnum() or code[start - 1] in "_$")]
                result = None if joined else "upsert" if found else "insert"
    return result


@functools.lru_cache(maxsize=_DETECTED_STATEMENTS)
def statement_operation(query: str) -> str:
    """label a statement with its SQL operation, from the verb of its main clause.

    A plain statement (a page's generated SQL) is labelled in a few searches (:func:`_plain_operation`); any other
    is read as tokens, and remembered per statement text: the label is a function of the text alone, and tokenizing a
    page's multi-row ``INSERT`` in Python, several times a statement, held the event loop for tens of milliseconds a
    page.

    The label tells the broker how to run the statement -- a ``select`` is fetched
    for rows, anything else executed for a count -- and names it in logs. It is the
    verb Postgres's own command tag would carry: a ``WITH ... INSERT`` is an
    ``insert``, and a data-modifying CTE under a ``SELECT`` is a ``select``. It is
    never what authorizes the statement; the broker reads that off the statement.

    :param query: SQL query string
    :ptype query: str
    :return: operation type (select, insert, update, delete, upsert); ``select``
        when the main clause opens with no verb this backend labels
    :rtype: str
    """
    plain = _plain_operation(query)
    return plain if plain is not None else _operation_of_tokens(query)


def _operation_of_tokens(query: str) -> str:
    """:func:`statement_operation` read from the statement's tokens (:func:`_code_tokens`): exact for any statement.

    :param query: SQL query string
    :ptype query: str
    :return: the label
    :rtype: str
    """
    tokens = _code_tokens(query)
    index = _main_clause_index(tokens)
    result = _MAIN_VERBS.get(tokens[index], "select") if index < len(tokens) else "select"
    if result == "insert" and any(
        tokens[position : position + 2] == ["ON", "CONFLICT"] for position in range(index, len(tokens))
    ):
        result = "upsert"
    return result


#: words whose presence anywhere in a statement's code may make it write, so it is never sent again: a
#: data-modifying CTE under a ``SELECT``, a sequence moved by ``nextval``/``setval``, a row lock taken by
#: ``FOR UPDATE``. Conservative on purpose: a read that names one only loses its retry.
_WRITING_WORDS: Final = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "TRUNCATE",
        "COPY",
        "CALL",
        "DO",
        "LOCK",
        "NEXTVAL",
        "SETVAL",
        "REFRESH",
        "CREATE",
        "DROP",
        "ALTER",
        "GRANT",
        "REVOKE",
    }
)


@functools.lru_cache(maxsize=_DETECTED_STATEMENTS)
def is_resendable(query: str) -> bool:
    """whether a statement only reads, so sending it again after it went unanswered changes nothing.

    Its main clause is a ``SELECT`` (:func:`statement_operation`) and no word of its code may write
    (:data:`_WRITING_WORDS`): ``WITH d AS (DELETE ... RETURNING id) SELECT id FROM d`` is labelled a
    ``select`` but deletes, so it is never resent. A ``SELECT`` calling a function that writes is read as
    a read; a statement that writes is sent through ``execute``, which is never resent on a timeout.
    Remembered per statement text, as :func:`statement_operation` is.

    :param query: the statement
    :ptype query: str
    :return: whether it may be sent again
    :rtype: bool
    """
    return statement_operation(query) == "select" and _WRITING_WORDS.isdisjoint(_code_tokens(query))
