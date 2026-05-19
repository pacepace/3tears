def test_core_import():
    from threetears.core import __version__

    assert __version__ == "0.8.4"


def test_cross_package_imports():
    from threetears.core import __version__ as core_version
    from threetears.agent.memory import __version__ as memory_version
    from threetears.agent.tools import __version__ as tools_version

    assert core_version == "0.8.4"
    assert memory_version == "0.8.4"
    assert tools_version == "0.8.4"
