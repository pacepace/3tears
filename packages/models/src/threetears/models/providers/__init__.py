"""provider factory functions returning configured LangChain models.

Each provider module exposes one or more ``create_*`` factory functions
that return configured LangChain ``BaseChatModel`` or ``Embeddings``
instances. Whisper is the exception — transcription has no
``BaseChatModel`` analog and exposes :class:`WhisperTranscriptionProvider`
directly.

Provider wire quirks are owned here too, so nothing above the provider
layer has to know a provider's request shape:
:func:`structured_output_kwargs` translates a json-schema into the
provider-native structured-output directive, alongside the tool-name
translation handled by the per-provider chat wrappers.
"""

from __future__ import annotations

from threetears.models.providers.structured_output import (
    StructuredOutputError,
    StructuredOutputSchemaError,
    StructuredOutputUnsupportedError,
    structured_output_kwargs,
)

__all__ = [
    "StructuredOutputError",
    "StructuredOutputSchemaError",
    "StructuredOutputUnsupportedError",
    "structured_output_kwargs",
]
