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
        (src,),
        repo,
        _LOGGER_NAMES,
        _LOGGER_METHODS,
        _MARKER,
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

        body = ast.parse("for x in y:\n    continue\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "continue"

    def test_break_classified(self) -> None:
        import ast

        body = ast.parse("for x in y:\n    break\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "break"

    def test_non_silent_returns_none(self) -> None:
        import ast

        body = ast.parse("def f():\n    x = recover()\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None

    def test_a_returned_constant_is_silent(self) -> None:
        # ``except ValueError: return False`` dropped unreadable judge cases from an arm while
        # the old one-shape walker looked straight past it.
        import ast

        body = ast.parse("def f():\n    return False\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "return-constant"

    def test_a_returned_empty_collection_is_silent(self) -> None:
        import ast

        for value in ("[]", "{}", "()", "dict()", "frozenset()"):
            body = ast.parse(f"def f():\n    return {value}\n").body[0].body  # type: ignore[attr-defined]
            assert body_silent_category(body) == "return-empty", value

    def test_an_assigned_default_is_silent(self) -> None:
        import ast

        body = ast.parse("def f():\n    ok = False\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "assign-default"

    def test_an_inert_multi_statement_body_is_silent_and_named_by_its_last_statement(self) -> None:
        import ast

        body = ast.parse("for x in y:\n    ok = None\n    continue\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) == "continue"

    def test_a_returned_value_that_is_not_a_default_is_not_silent(self) -> None:
        import ast

        body = ast.parse("def f():\n    return Failure(reason=str(exc))\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None

    def test_a_body_that_calls_anything_is_not_silent(self) -> None:
        import ast

        body = ast.parse("def f():\n    x = compute()\n    pass\n").body[0].body  # type: ignore[attr-defined]
        assert body_silent_category(body) is None


# ------------------------------------------------------------------
# helpers — body_contains_log
# ------------------------------------------------------------------


class TestBodyContainsLog:
    def test_bare_log_call(self) -> None:
        import ast

        body = ast.parse("def f():\n    log.error('x')\n").body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_self_log_call(self) -> None:
        import ast

        body = ast.parse("def f():\n    self.log.error('x')\n").body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_self_logger_call(self) -> None:
        import ast

        body = ast.parse("def f():\n    ctx.logger.warning('x')\n").body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True

    def test_unrelated_attribute_not_logged(self) -> None:
        # ``self.cache.error(...)`` is not a logger call: the attribute
        # name ``cache`` is not in ``{log, logger}``.
        import ast

        body = ast.parse("def f():\n    self.cache.error('x')\n").body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is False

    def test_nested_log_call(self) -> None:
        # a logger call deeper in the body still counts.
        import ast

        body = ast.parse("def f():\n    if cond:\n        log.error('x')\n").body[0].body  # type: ignore[attr-defined]
        assert body_contains_log(body, _LOGGER_NAMES, _LOGGER_METHODS) is True


# ------------------------------------------------------------------
# helpers — body_reraises
# ------------------------------------------------------------------


class TestBodyReraises:
    def test_bare_raise(self) -> None:
        import ast

        body = ast.parse("try:\n    pass\nexcept Exception:\n    raise\n").body[0].handlers[0].body  # type: ignore[attr-defined]
        assert body_reraises(body) is True

    def test_raise_new_error(self) -> None:
        import ast

        body = ast.parse("try:\n    pass\nexcept Exception as e:\n    raise X() from e\n").body[0].handlers[0].body  # type: ignore[attr-defined]
        assert body_reraises(body) is True

    def test_no_raise(self) -> None:
        import ast

        body = ast.parse("try:\n    pass\nexcept Exception:\n    pass\n").body[0].handlers[0].body  # type: ignore[attr-defined]
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
            "def f():\n    try:\n        do()\n    except:\n        pass\n",
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
            "def f():\n    try:\n        do()\n    except:\n        log.error('x')\n",
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
            "def f():\n    try:\n        do()\n    except ValueError:\n        pass\n",
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
            "def f():\n    try:\n        do()\n    except ValueError:\n        ...\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "body=ellipsis" in violations[0].reason

    def test_silent_return_none_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:\n        return None\n",
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
            "def f():\n    try:\n        do()\n    except ValueError as e:\n        log.error(e)\n",
        )
        assert _scan(src, repo) == []

    def test_reraise_handler_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:\n        raise\n",
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
            "def f():\n    try:\n        do()\n    except ValueError:\n        log.error('x')\n        raise\n",
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
            "    except ValueError:  # NOSILENT(expected): the refusal is this probe's answer\n"
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
            "    # NOSILENT(expected): shutdown is the expected path here\n"
            "    except asyncio.CancelledError:\n"
            "        pass\n",
        )
        assert _scan(src, repo) == []

    def test_non_silent_body_not_flagged(self, tmp_path: Path) -> None:
        # ``except: x = recover()`` is recovery code, not a silent swallow.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:\n        x = recover()\n",
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
            "def f():\n    try:\n        do()\n    except ValueError:  # NOSILENT:\n        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "ValueError"

    def test_marker_with_only_whitespace_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:  # NOSILENT:    \n        pass\n",
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
            "from contextlib import suppress\n\ndef f():\n    with suppress(KeyError):\n        do()\n",
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
            "import contextlib\n\ndef f():\n    with contextlib.suppress(KeyError):\n        do()\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].category == "no_silent_swallow.suppress"

    def test_suppress_with_marker_above_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    # NOSILENT(expected): race-condition cleanup, key may already be gone\n"
            "    with suppress(KeyError):\n"
            "        del cache[key]\n",
        )
        assert _scan(src, repo) == []

    def test_suppress_with_marker_inline_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "\n"
            "def f():\n"
            "    with suppress(KeyError):  # NOSILENT(expected): idempotent delete\n"
            "        del cache[key]\n",
        )
        assert _scan(src, repo) == []

    def test_suppress_multi_arg_symbol(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n\ndef f():\n    with suppress(KeyError, ValueError):\n        do()\n",
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
        self,
        tmp_path: Path,
    ) -> None:
        # synthetic two-package workspace: package A holds clean code,
        # package B holds the silent swallow. with both src roots
        # passed to the walker (mimicking discover_src_roots's output
        # for a path-dep workspace), the violation in B is found.
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:\n        log.error('x')\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py",
            "def f():\n    try:\n        do()\n    except ValueError:\n        pass\n",
        )
        violations = find_silent_swallows(
            (a_src, b_src),
            tmp_path,
            _LOGGER_NAMES,
            _LOGGER_METHODS,
            _MARKER,
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
            "def f():\n    try:\n        do()\n    except foo.BarError:\n        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "BarError"

    def test_tuple_exception_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n    try:\n        do()\n    except (KeyError, ValueError):\n        pass\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "(KeyError, ValueError)"


# ------------------------------------------------------------------
# walker — the shapes the one-statement walker missed
# ------------------------------------------------------------------


class TestShapesTheOldWalkerMissed:
    def test_a_handler_returning_false_is_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py", "def f():\n    try:\n        do()\n    except ValueError:\n        return False\n"
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "return-constant" in violations[0].reason

    def test_a_handler_assigning_a_default_is_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "def f():\n    try:\n        x = do()\n    except KeyError:\n        x = None\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "assign-default" in violations[0].reason

    def test_an_except_star_handler_is_scanned(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "def f():\n    try:\n        do()\n    except* ValueError:\n        pass\n")
        assert len(_scan(src, repo)) == 1


# ------------------------------------------------------------------
# walker — a reason no longer excuses silence; a checked kind does
# ------------------------------------------------------------------


def _handler(marker: str, caught: str = "ValueError", body: str = "return None", prefix: str = "") -> str:
    return f"def f():\n{prefix}    try:\n        do()\n    except {caught}:\n        {marker}\n        {body}\n"


class TestMarkerKinds:
    def test_a_bare_reason_no_longer_excuses_a_silent_handler(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT: the caller treats None as absent"))
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "no longer excuses" in violations[0].reason

    def test_a_bare_reason_still_sits_harmlessly_on_a_handler_that_logs(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT: logged below", body="log.warning('x')"))
        assert _scan(src, repo) == []

    def test_an_unknown_kind_is_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(harmless): trust me"))
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "unknown NOSILENT kind 'harmless'" in violations[0].reason

    def test_a_kind_without_a_reason_is_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(expected):"))
        assert len(_scan(src, repo)) == 1

    def test_expected_is_accepted_for_a_narrow_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(expected): a miss is the documented None", "KeyError"))
        assert _scan(src, repo) == []

    def test_expected_is_refused_for_a_broad_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(expected): anything at all", "(KeyError, Exception)"))
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "narrower than Exception" in violations[0].reason

    def test_optional_import_is_accepted_around_an_import(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "try:\n    import numpy\nexcept ImportError:  # NOSILENT(optional-import): numpy is an extra\n"
            "    numpy = None\n",
        )
        assert _scan(src, repo) == []

    def test_optional_import_is_refused_around_no_import(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(optional-import): not really", "ImportError"))
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "guards no import" in violations[0].reason

    def test_optional_import_is_refused_for_another_type(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "try:\n    import x\nexcept OSError:  # NOSILENT(optional-import): no\n    x = None\n",
        )
        assert len(_scan(src, repo)) == 1

    def test_attempt_is_accepted_inside_a_loop(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f(formats):\n"
            "    for fmt in formats:\n"
            "        try:\n"
            "            return parse(fmt)\n"
            "        except ValueError:  # NOSILENT(attempt): one format of several; none raises below\n"
            "            continue\n"
            "    raise ValueError('no format')\n",
        )
        assert _scan(src, repo) == []

    def test_attempt_is_accepted_when_a_raise_follows_the_try(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            _handler("# NOSILENT(attempt): one parse of two", body="pass")
            + "    try:\n        return other()\n    except ValueError as exc:\n        raise Bad() from exc\n",
        )
        assert _scan(src, repo) == []

    def test_attempt_is_refused_with_no_loop_and_no_raise_after(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", _handler("# NOSILENT(attempt): one parse of several"))
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "inside a loop" in violations[0].reason

    def test_attempt_does_not_count_a_raise_in_a_nested_function(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            _handler("# NOSILENT(attempt): one parse of several") + "    def g():\n        raise Bad()\n",
        )
        assert len(_scan(src, repo)) == 1

    def test_cleanup_is_accepted_inside_an_except_that_reraises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "def f():\n"
            "    try:\n"
            "        write()\n"
            "    except BaseException:\n"
            "        # NOSILENT(cleanup): the temp file goes; the write's own error is re-raised below\n"
            "        with suppress(OSError):\n"
            "            unlink()\n"
            "        raise\n",
        )
        assert _scan(src, repo) == []

    def test_cleanup_is_refused_outside_a_reraising_except(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\n"
            "def f():\n"
            "    # NOSILENT(cleanup): best effort\n"
            "    with suppress(Exception):\n"
            "        close()\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "re-raises the primary error" in violations[0].reason

    def test_a_bare_reason_no_longer_excuses_a_suppress(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\ndef f():\n    # NOSILENT: idempotent delete\n"
            "    with suppress(KeyError):\n        del cache[key]\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert "no longer excuses" in violations[0].reason

    def test_expected_is_refused_on_a_suppress_of_exception(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from contextlib import suppress\ndef f():\n    # NOSILENT(expected): whatever happens\n"
            "    with suppress(Exception):\n        close()\n",
        )
        assert len(_scan(src, repo)) == 1
