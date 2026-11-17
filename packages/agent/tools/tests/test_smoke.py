def test_agent_tools_import() -> None:
    from threetears.agent.tools import __version__

    assert __version__ == "0.95.0"
