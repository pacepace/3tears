// A classic worker: importScripts from the worker.
importScripts("/app/lab/classic-dep.js");
postMessage(self.LAB_VALUE);
