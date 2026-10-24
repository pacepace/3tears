"""The package imports without its optional extras, and building a model that needs one names it.

Every provider module imports its backend when it loads, an optional extra's under a guard. A host
without the ``claude-cli`` or ``voyageai`` extra must still import :mod:`threetears.models`, and the
first call that needs the missing extra must raise an ``ImportError`` saying which extra to install.

Each check runs in a fresh interpreter with the extra's modules refused at import, as an install
without the extra refuses them.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _refusing(refused: tuple[str, ...], body: str) -> subprocess.CompletedProcess[str]:
    """run ``body`` in a fresh interpreter that refuses to import ``refused`` and their submodules.

    :param refused: top-level module names an install without the extra lacks
    :ptype refused: tuple[str, ...]
    :param body: the program to run once the refusal is in place
    :ptype body: str
    :return: the finished process
    :rtype: subprocess.CompletedProcess[str]
    """
    block = textwrap.dedent(
        f"""
        import importlib.abc, sys
        _REFUSED = {refused!r}
        class _Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == r or name.startswith(r + ".") for r in _REFUSED):
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
                return None
        sys.meta_path.insert(0, _Refuse())
        """
    )
    return subprocess.run(
        [sys.executable, "-c", block + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_a_subscription_model_without_the_claude_cli_extra_names_the_extra() -> None:
    result = _refusing(
        ("claude_agent_sdk", "langchain_claude_code"),
        """
        from threetears.models.providers.claude_cli import create_subscription_chat

        try:
            create_subscription_chat("claude-sonnet-4-5", "sk-ant-oat01-notarealtoken")
        except ImportError as exc:
            assert "3tears-models[claude-cli]" in str(exc), str(exc)
        else:
            raise AssertionError("a subscription model was built without the extra")
        print("refused")
        """,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_voyageai_embeddings_without_the_voyageai_extra_name_the_extra() -> None:
    result = _refusing(
        ("langchain_voyageai", "voyageai"),
        """
        from threetears.models.providers.voyageai import create_voyageai_embedding

        try:
            create_voyageai_embedding("pa-notarealkey")
        except ImportError as exc:
            assert "3tears-models[voyageai]" in str(exc), str(exc)
        else:
            raise AssertionError("a VoyageAI embedding was built without the extra")
        print("refused")
        """,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_the_package_imports_without_either_extra() -> None:
    result = _refusing(
        ("claude_agent_sdk", "langchain_claude_code", "mcp", "langchain_voyageai", "voyageai"),
        """
        import threetears.models
        import threetears.models.claude_cli_pool

        print("imported")
        """,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_a_process_without_the_voyageai_extra_gets_no_stub_module() -> None:
    result = _refusing(
        ("langchain_voyageai",),
        """
        import sys
        import threetears.models  # noqa: F401
        stubbed = "voyageai.object.multimodal_embeddings" in sys.modules
        print("stubbed" if stubbed else "clean")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("clean"), "importing threetears.models put the voyageai stub in sys.modules"
