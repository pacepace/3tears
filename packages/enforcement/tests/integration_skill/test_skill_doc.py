"""tests for SKILL.md parsing: grammar, sentinel, fenced imports."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.integration_skill import (
    SENTINEL,
    SkillParseError,
    parse_skill_document,
)

_VALID = f"""---
name: integrating-example
description: example skill
attested:
  threetears.core: "sha256:0123abcd @ 1.2.3"
  threetears.agent.wake: "sha256:deadbeef @ 0.1.0"
---

> {SENTINEL}

## Quickstart

```python
import uuid

from threetears.core import DataStore
from threetears.core.cache.sqlite import SQLiteBackend

store = DataStore(uuid.uuid4(), None, None)
await store.create_table(None)
```

prose between blocks

```python
import threetears.observe
```
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "SKILL.md"
    path.write_text(text)
    return path


class TestParseSkillDocument:
    def test_parses_meta_attested_and_blocks(self, tmp_path: Path) -> None:
        doc = parse_skill_document(_write(tmp_path, _VALID))
        assert doc.meta["name"] == "integrating-example"
        assert doc.attested["threetears.core"].fingerprint == "sha256:0123abcd"
        assert doc.attested["threetears.core"].version == "1.2.3"
        assert doc.attested["threetears.agent.wake"].version == "0.1.0"
        assert len(doc.code_blocks) == 2

    def test_sentinel_matches_through_blockquote(self, tmp_path: Path) -> None:
        doc = parse_skill_document(_write(tmp_path, _VALID))
        assert doc.opens_with_sentinel

    def test_paraphrased_sentinel_does_not_match(self, tmp_path: Path) -> None:
        text = _VALID.replace("It must not be trusted", "It should not be trusted")
        doc = parse_skill_document(_write(tmp_path, text))
        assert not doc.opens_with_sentinel

    def test_body_not_opening_with_sentinel_does_not_match(self, tmp_path: Path) -> None:
        text = _VALID.replace(f"> {SENTINEL}", f"## Intro\n\n> {SENTINEL}")
        doc = parse_skill_document(_write(tmp_path, text))
        assert not doc.opens_with_sentinel

    def test_extracts_threetears_imports_including_top_level_await_blocks(self, tmp_path: Path) -> None:
        doc = parse_skill_document(_write(tmp_path, _VALID))
        assert ("threetears.core", "DataStore") in doc.threetears_imports()
        assert ("threetears.core.cache.sqlite", "SQLiteBackend") in doc.threetears_imports()
        assert ("threetears.observe", None) in doc.threetears_imports()
        assert all(module.startswith("threetears") for module, _ in doc.threetears_imports())

    def test_missing_opening_fence_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SkillParseError, match="opening frontmatter fence"):
            parse_skill_document(_write(tmp_path, "# no frontmatter\n"))

    def test_missing_closing_fence_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SkillParseError, match="closing frontmatter fence"):
            parse_skill_document(_write(tmp_path, "---\nname: x\n"))

    def test_malformed_frontmatter_line_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SkillParseError, match="does not match the frontmatter grammar"):
            parse_skill_document(_write(tmp_path, "---\nname: x\n!!bogus!!\n---\nbody\n"))

    def test_attested_entry_outside_block_raises(self, tmp_path: Path) -> None:
        text = '---\nname: x\n  threetears.core: "sha256:ab @ 1"\n---\nbody\n'
        with pytest.raises(SkillParseError, match="outside an 'attested:' block"):
            parse_skill_document(_write(tmp_path, text))

    def test_unterminated_python_fence_raises(self, tmp_path: Path) -> None:
        text = "---\nname: x\n---\n```python\nimport threetears.core\n"
        with pytest.raises(SkillParseError, match="unterminated"):
            parse_skill_document(_write(tmp_path, text))
