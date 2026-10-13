# Recordings

What Claude answered when each example that calls a model was last run live, one JSONL file per example
(`<example>.jsonl`, plus `llm_analysis.bundle.json`, the bundle that recording was made from). An example run with
no `ANTHROPIC_API_KEY` replays its file here and calls nothing. `RECORD=1 python <example>.py`, with the key set,
records it again; [`../README.md`](../README.md#recording) has every command.
