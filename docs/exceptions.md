# Exceptions

**A failure either reaches the caller, or it is recorded where an operator will see it. Never neither.**

A handler that catches a failure and hands back `None`, `[]`, `False`, a partial result or a "success" has told its
caller something untrue. Logging first does not change that: the caller still acts on the wrong answer. Those
answers read as ordinary results at every layer above, which is how a replay that stopped short of the stream head
came to report a clean tail, and how an embedder outage came to read as "no relevant memories".

## What a handler may do

| The failure is... | Do this |
|---|---|
| something the caller must know about | **Raise.** Let it propagate, or wrap it with `raise NewError(...) from exc`. |
| something the caller is built to report: a tool result, an HTTP response, a result object with an error field | **Return the failure as that value**, and `contain()` it if the catch is broad. |
| a specific, expected condition: a missing key, an unparseable field, an optional import | **Catch that type, not `Exception`.** A narrow catch is ordinary control flow; the type name is its documentation. |
| absorbed at a boundary that must keep going: a supervisor loop, a best-effort side effect, teardown | **`contain()` it.** |

Never write a "soft-fail" wrapper (`_safe_embed(...) -> X | None`). It turns one swallow into one at every call site,
and the callers can't tell an outage from an empty answer.

A security check fails closed: if it can't classify an input, it refuses the input.

## The helpers (`threetears.observe`)

```python
try:
    await refresh_grants()
except Exception as exc:
    contain(exc, log=log, what="grant cache refresh failed", extra={"epoch": epoch})
```

- `contain(exc, *, log, what, extra=None, level="warning")` logs `what` verbatim, at `"warning"`, `"error"` or `"critical"`, with the
  traceback, adds `contained_site` and `exc_type` to the record's `extra_data`, and counts
  `contained_errors_total{site, exc_type}`. `site` is the catching function, `module:Qualified.name`. It does not
  change control flow.
- `with contained(what, *, log, extra=None):` does the same for a block whose failure falls through. It absorbs
  `Exception` and nothing wider.
- `await cancel_and_wait(task)` stops another task. Don't hand-write `task.cancel()` followed by `await task` under
  `except CancelledError: pass`: that swallows the caller's own cancellation too.

`contain()` makes a failure visible, not honest. A handler that contains a failure and then returns `[]` still hands
its caller a default that reads as an answer. The metric only makes that countable.

## What enforces it

`tests/enforcement/test_no_silent_swallow.py`, over every workspace `src` tree. No comment excuses a broad or
cancellation handler:

- **`except Exception`**, alone or in a tuple, re-raises on some path or calls `contain()`. `suppress(Exception)`
  becomes `with contained(...)`.
- **`except BaseException`** re-raises. It catches cancellation, `KeyboardInterrupt` and `SystemExit`.
- **`except CancelledError`** re-raises. `suppress(CancelledError)` is refused.
- **A narrow handler whose body is only `pass`, `...`, `return None`, `continue` or `break`** logs, re-raises or
  carries `# NOSILENT: <reason>`. Use the marker only where the failure is the expected outcome and there is nothing
  to tell anyone.

Packages not yet converted are listed in `_UNCONTAINED_PENDING` in that file. The list only shrinks: an entry with
nothing left to convert fails the gate, and a package not on it, including every new one, is checked.

**Under pytest, a contained programming error fails the test.** If `contain()` absorbs a `TypeError`,
`AttributeError`, `NameError` or `AssertionError`, the test fails at teardown (`tests/contain_tripwire.py`). In
production that bug would be logged and the code would carry on; in CI it fails. A test that drives one there on
purpose says so with `@pytest.mark.expect_contained(TypeError, why="...")`.

## Converting a package

The conversion changes no behaviour. Each handler keeps its control flow and its return value; only the recording
changes.

1. For each `except Exception as exc:` that doesn't re-raise, replace its log call with `contain(exc, log=<the same
   logger>, what=<the same message>, extra=<the same extra_data fields>)`. Pass `level="error"` where it logged
   at `error`, `exception` or `critical`. Where it logged nothing, write a message that says what failed. Bind
   `as exc` if the handler doesn't. Leave the body's returns, assignments and `continue`s exactly as they are.
2. `with suppress(Exception):` becomes `with contained("<what failed>", log=log):`.
3. `task.cancel()` followed by `await task` under `except`/`suppress(CancelledError)` becomes
   `await cancel_and_wait(task)`. A task catching its own cancellation to tidy up re-raises after the tidy-up.
4. A test that matches a converted record's `extra_data` exactly now also sees `contained_site` and `exc_type`. Add
   them to the expected value; don't loosen the comparison.
5. If the tripwire fails a test, the conversion has found a bug. Report it; don't mark it expected to make CI pass.
6. Delete the package's line from `_UNCONTAINED_PENDING`.

`packages/observe` is the worked example.
