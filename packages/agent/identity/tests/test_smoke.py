def test_agent_identity_import() -> None:
    from threetears.agent.identity import __version__

    assert __version__ == "0.95.0"
