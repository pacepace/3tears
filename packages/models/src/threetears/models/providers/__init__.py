"""provider factory functions returning configured LangChain models.

Each provider module exposes one or more ``create_*`` factory functions
that return configured LangChain ``BaseChatModel`` or ``Embeddings``
instances. Whisper is the exception — transcription has no
``BaseChatModel`` analog and exposes :class:`WhisperTranscriptionProvider`
directly.
"""
