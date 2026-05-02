"""tests for the no-silent-swallow ``find_silent_swallows`` walker."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_silent_swallow.walkers import (
    body_contains_log,
    body_reraises,
    body_silent_category,
    find_silent_swallows,
)


_LOGGER_NAMES = frozenset({"log", "logger", "_log", "_logger", "LOGGER"})
_LOGGER_METHODS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception"},
)
_MARKER = "# NOSILENT:"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _scan(src: Path, repo: Path) -> list:
    return find_silent_swallows(
        (src,), repo, _LOGGER_NAMES, _LOGGER_METHODS, _MARKER,
    )


# ------------------------------------------------------------------
# helpers — body_silent_category
# ------------------------------------------------------------------

class TestBodySilentCategory:
    def test_pass_classified(self) -> None:
        import ast
        body = ast.parse("def f():\n    pass\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "pass"

    def test_ellipsis_classified(self) -> None:
        import ast
        body = ast.parse("def f():\n    ...\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "ellipsis"

    def test_return_none_explicit_classified(self) -> None:
        import ast
        body = ast.parse("def f():\n    return None\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "return-none"

    def test_return_no_value_classified(self) -> None:
        import ast
        body = ast.parse("def f():\n    return\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "return-none"

    def test_continue_classified(self) -> None:
        import ast
        body = ast.parse(
            "for x in y:\n"
            "    continue\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "continue"

    def test_break_classified(self) -> None:
        import ast
        body = ast.parse(
            "for x in y:\n"
            "    break\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "break"

    def test_non_silent_returns_none(self) -> None:
        import ast
        body = ast.parse("def f():\n    x = 1\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None

    def test_return_value_not_none(self) -> None:
        import ast
        body = ast.parse(
            "def f():\n    return 42\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None

    def test_multi_statement_returns_none(self) -> None:
        import ast
        body = ast.parse(
            "def f():\n    x = 1\n    pass\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None


# ------------------------------------------------------------------
# helpers — body_contains_log
# ------------------------------------------------------------------

class TestBodyContainsLog:
    def test_bare_log_call(self) -> None:
        import ast
        body = ast.parse(
            "def f():\n    log.error('x')\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_self_log_call(self) -> None:
        import ast
        body = ast.parse(
            "def f():\n    self.log.error('x')\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_self_logger_call(self) -> None:
        import ast
        body = ast.parse(
            "def f():\n    ctx.logger.warning('x')\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_unrelated_attribute_not_logged(self) -> None:
        # ``self.cache.error(...)`` is not a logger call: the attribute
        # name ``cache`` is not in ``{log, logger}``.
        import ast
        body = ast.parse(
            "def f():\n    self.cache.error('x')\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is False

    def test_nested_log_call(self) -> None:
        # a logger call deeper in the body still counts.
        import ast
        body = ast.parse(
            "def f():\n"
            "    if cond:\n"
            "        log.error('x')\n"
        ).body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True


# ------------------------------------------------------------------
# helpers — body_reraises
# ------------------------------------------------------------------

class TestBodyReraises:
    def test_bare_raise(self) -> None:
        import ast
        body = ast.parse(
            "try:\n    pass\nexcept Exception:\n    raise\n"
        ).body[0].handlers[0].body  # type: ignore[attr-defined]
        assert body_reraises(body) is True

    def test_raise_new_error(self) -> None:
        import ast
        body = ast.parse(
            "try:\n    pass\nexcept Exception as e:\n    raise X() from e\n"
        ).body[0].handlers[0].body  # type: ignore[attr-defined]
        assert body_reraises(body) is True

    def test_no_raise(self) -> None:
        import ast
        body = ast.parse(
            "try:\n    pass\nexcept Exception:\n    pass\n"
        ).body[0].handlers[0].body  # type: ignore[attr-defined]
        assert body_reraises(body) is False


# ------------------------------------------------------------------
# walker — bare except
# ------------------------------------------------------------------

class TestBareExcept:
    def test_bare_except_pass_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except:\n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "no_silent_swallow.except"
        assert v.symbol == "BaseException"
        assert "bare `except:`" in v.reason

    def test_bare_except_with_log_still_flagged(self, tmp_path: Path) -> None:
        # bare ``except:`` is rejected unconditionally — the marker /
        # log / re-raise tolerated patterns only apply to typed
        # handlers.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except:\n"
            "        log.error('x')\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "BaseException"


# ------------------------------------------------------------------
# walker — typed except, silent
# ------------------------------------------------------------------

class TestTypedSilentExcept:
    def test_silent_pass_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "no_silent_swallow.except"
        assert v.symbol == "ValueError"
        assert "silent swallow" in v.reason
        assert "body=pass" in v.reason

    def test_silent_ellipsis_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        ...\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "body=ellipsis" in violations[0].reason

    def test_silent_return_none_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        return None\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "body=return-none" in violations[0].reason

    def test_silent_continue_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    for x in y:\n"
            "        try:\n"
            "            do()\n"
            "        except ValueError:\n"
            "            continue\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "body=continue" in violations[0].reason

    def test_silent_break_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    for x in y:\n"
            "        try:\n"
            "            do()\n"
            "        except ValueError:\n"
            "            break\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "body=break" in violations[0].reason


# ------------------------------------------------------------------
# walker — typed except, accepted patterns
# ------------------------------------------------------------------

class TestTypedExceptAccepted:
    def test_logged_handler_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError as e:\n"
            "        log.error(e)\n",
        )
        assert _scan(src, repo) == []

    def test_reraise_handler_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        raise\n",
        )
        # ``raise`` makes the body non-silent (not pass / ellipsis /
        # return / continue / break) — body_silent_category returns
        # None and the handler is skipped.
        assert _scan(src, repo) == []

    def test_logged_and_reraise_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        log.error('x')\n"
            "        raise\n",
        )
        assert _scan(src, repo) == []

    def test_marker_with_reason_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # marker placed on the except line itself.
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:  # NOSILENT: legitimate cleanup path\n"
            "        pass\n",
        )
        assert _scan(src, repo) == []

    def test_marker_above_except_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # marker placed on the line above the except.
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    # NOSILENT: shutdown is the expected path here\n"
            "    except asyncio.CancelledError:\n"
            "        pass\n",
        )
        assert _scan(src, repo) == []

    def test_non_silent_body_not_flagged(self, tmp_path: Path) -> None:
        # ``except: x = 1`` is recovery code, not a silent swallow.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        x = 1\n",
        )
        assert _scan(src, repo) == []


# ------------------------------------------------------------------
# walker — marker rationale-required
# ------------------------------------------------------------------

class TestMarkerReasonRequired:
    def test_empty_marker_reason_rejected(self, tmp_path: Path) -> None:
        # ``# NOSILENT:`` with nothing after the colon must NOT silence
        # the violation. preserves the canonical's intent that the
        # comment carries an operator's reason.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:  # NOSILENT:\n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "ValueError"

    def test_marker_with_only_whitespace_rejected(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:  # NOSILENT:    \n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1


# ------------------------------------------------------------------
# walker — contextlib.suppress
# ------------------------------------------------------------------

class TestSuppress:
    def test_bare_suppress_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    with suppress(KeyError):\n"
            "        do()\n",
        )
        violations = _scan(src, repo)
        # one violation for the suppress; the with-body is non-silent
        # so no extra flag.
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "no_silent_swallow.suppress"
        assert v.symbol == "KeyError"
        assert "contextlib.suppress" in v.reason

    def test_qualified_contextlib_suppress_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import contextlib\n"
            "\n"
            "def f():\n"
            "    with contextlib.suppress(KeyError):\n"
            "        do()\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].category == "no_silent_swallow.suppress"

    def test_suppress_with_marker_above_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    # NOSILENT: race-condition cleanup, key may already be gone\n"
            "    with suppress(KeyError):\n"
            "        del cache[key]\n",
        )
        assert _scan(src, repo) == []

    def test_suppress_with_marker_inline_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    with suppress(KeyError):  # NOSILENT: idempotent delete\n"
            "        del cache[key]\n",
        )
        assert _scan(src, repo) == []

    def test_suppress_multi_arg_symbol(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    with suppress(KeyError, ValueError):\n"
            "        do()\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        # the symbol is rendered as a parenthesised tuple of names.
        assert violations[0].symbol == "(KeyError, ValueError)"


# ------------------------------------------------------------------
# walker — multiple violations and ordering
# ------------------------------------------------------------------

class TestMultipleViolations:
    def test_mixed_violations_in_one_file(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except:\n"
            "        pass\n"
            "\n"
            "def g():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        pass\n"
            "\n"
            "def h():\n"
            "    with suppress(KeyError):\n"
            "        do()\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 3
        categories = [v.category for v in violations]
        assert categories.count("no_silent_swallow.except") == 2
        assert categories.count("no_silent_swallow.suppress") == 1


# ------------------------------------------------------------------
# walker — path-dep / multi-root behaviour
# ------------------------------------------------------------------

class TestPathDepWalking:
    def test_two_package_workspace_finds_violation(
        self, tmp_path: Path,
    ) -> None:
        # synthetic two-package workspace: package A holds clean code,
        # package B holds the silent swallow. with both src roots
        # passed to the walker (mimicking discover_src_roots's output
        # for a path-dep workspace), the violation in B is found.
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        log.error('x')\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        pass\n",
        )
        violations = find_silent_swallows(
            (a_src, b_src), tmp_path,
            _LOGGER_NAMES, _LOGGER_METHODS, _MARKER,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "ValueError"
        assert v.file == b_src / "pkg_b" / "mod.py"


# ------------------------------------------------------------------
# walker — symbol rendering for unusual handler shapes
# ------------------------------------------------------------------

class TestHandlerSymbol:
    def test_attribute_exception_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except foo.BarError:\n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "BarError"

    def test_tuple_exception_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except (KeyError, ValueError):\n"
            "        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "(KeyError, ValueError)"
