"""the public-surface reader: what it counts, and every shape it refuses to read as empty."""

from __future__ import annotations

import pytest

from threetears.enforcement.release import SurfaceReadError, http_routes, module_surface


class TestWhatCounts:
    """the names a consumer can reach."""

    def test_all_entries_and_exported_class_methods(self) -> None:
        """
        ``__all__`` names, and public methods of an exported class.

        :return: nothing
        :rtype: None
        """
        source = (
            '__all__ = ["Thing", "helper"]\n'
            "class Thing:\n    def run(self): ...\n    def _hidden(self): ...\n"
            "class Other:\n    def run(self): ...\n"
            "def helper(): ...\n"
        )
        assert module_surface(source, "m.py") == {"Thing", "Thing.run", "helper"}

    def test_a_module_without_all_exports_what_it_defines(self) -> None:
        """
        Python's ``import *`` rule, less imports and private names.

        :return: nothing
        :rtype: None
        """
        source = "import os\nfrom x import y\nCONSTANT = 1\ntype Alias = int\ndef run(): ...\ndef _hidden(): ...\n"
        assert module_surface(source, "m.py") == {"CONSTANT", "Alias", "run"}

    def test_literal_additions_under_an_optional_extra_count(self) -> None:
        """
        ``__all__.append`` / ``.extend`` / ``+=`` / ``.insert`` with literals are read, not refused.

        :return: nothing
        :rtype: None
        """
        source = (
            '__all__ = ["base"]\n'
            'try:\n    from x import Slack\n    __all__.append("Slack")\nexcept ImportError:\n    pass\n'
            'try:\n    __all__.extend(["A", "B"])\n    __all__ += ("C",)\n    __all__.insert(0, "D")\n'
            "except ImportError:\n    pass\n"
        )
        assert module_surface(source, "m.py") == {"base", "Slack", "A", "B", "C", "D"}

    def test_routes_come_from_the_extractor(self) -> None:
        """
        a decorator route is surface only when the caller asks for routes.

        :return: nothing
        :rtype: None
        """
        source = (
            '__all__ = []\n@router.get("/things")\nasync def things(): ...\n@router.post("")\nasync def make(): ...\n'
        )
        assert module_surface(source, "m.py") == set()
        assert module_surface(source, "m.py", (http_routes,)) == {"route GET /things", "route POST "}


class TestNeverQuietlyEmpty:
    """each unreadable shape is an error naming the module, never an empty set."""

    @pytest.mark.parametrize(
        ("source", "reason"),
        [
            ('__all__ = [*base.__all__, "x"]\n', "not a literal list of strings"),
            ('__all__ = sorted(["b", "a"])\n', "not a literal list of strings"),
            ('__all__ = ["a", NAME]\n', "not a literal list of strings"),
            ('__all__ = ["a"]\n__all__ += other\n', "augmented with a non-literal"),
            ('__all__ = ["a"]\n__all__ -= ["a"]\n', "changed by"),
            ('__all__ = ["a"]\n__all__.extend(names)\n', "cannot read"),
            ('__all__ = ["a"]\n__all__.remove("a")\n', "cannot read"),
            ('__all__ = ["a"]\n__all__ = ["b"]\n', "assigned more than once"),
            ('if X:\n    __all__ = ["a"]\n', "assigned inside a block"),
            ('__all__.append("a")\n', "changed but never assigned"),
            ("def broken(:\n", "does not parse"),
        ],
    )
    def test_an_unreadable_module_is_an_error(self, source: str, reason: str) -> None:
        """
        the error names the module and the reason.

        :param source: module source
        :ptype source: str
        :param reason: text the reason must contain
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        with pytest.raises(SurfaceReadError) as raised:
            module_surface(source, "pkg/odd.py")
        assert raised.value.module_path == "pkg/odd.py"
        assert reason in raised.value.reason
        assert str(raised.value).startswith("pkg/odd.py: ")

    def test_an_extractor_that_cannot_read_is_an_error_naming_the_module(self) -> None:
        """
        an extractor's ``ValueError`` becomes the module's read error.

        :return: nothing
        :rtype: None
        """

        def refuses(_tree: object) -> set[str]:
            raise ValueError("DISPATCH is not a literal")

        with pytest.raises(SurfaceReadError, match=r"^pkg/tool\.py: DISPATCH is not a literal$"):
            module_surface("__all__ = []\n", "pkg/tool.py", (refuses,))
