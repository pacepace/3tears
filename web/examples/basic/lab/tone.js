// An AudioWorklet module with an import of its own: a constant 0.5 on every sample.
import { LEVEL } from "/app/lab/tone-dep.js";

registerProcessor(
  "tone",
  class extends AudioWorkletProcessor {
    process(_inputs, outputs) {
      for (const channel of outputs[0]) channel.fill(LEVEL);
      return true;
    }
  },
);
