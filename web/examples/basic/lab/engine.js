// A module worker: a static import, a dynamic import and a fetch, all from the worker.
import { first } from "/app/lab/engine-dep.js";

const { second } = await import("/app/lab/engine-late.js");
const { third } = await (await fetch("/app/lab/engine-data.json")).json();
postMessage(`${first}${second}${third}`);
