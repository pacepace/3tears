# 3tears-keda-autoscaling

Reusable Helm **library chart** providing a trigger-agnostic KEDA `ScaledObject`
renderer for any 3tears consumer. Lifted from `14-eng-ai-bot`'s Hub chart
(`k8s/helm/aibots/`, `docs/resilience/scale-task-00-keda-chart-gate.md`), which had
already built this pattern generically -- this chart generalizes it out of the
`aibots.*` Helm namespace so a consumer depends on `3tears`, not on the Hub.

A **library chart** ships no standalone Kubernetes resources of its own; it only
exposes named templates (`_helpers.tpl`) for a consuming chart's own `templates/*.yaml`
to `include`. Take it as a Helm dependency in your own `Chart.yaml`:

```yaml
dependencies:
  - name: 3tears-keda-autoscaling
    version: "0.1.0"
    repository: "file://../../path/to/3tears/charts/3tears-keda-autoscaling"
    # or a real chart repository once one is published
```

## Autoscaling (KEDA)

Event-driven autoscaling via [KEDA](https://keda.sh) -- `ScaledObject`s driven off
NATS-JetStream consumer lag (the 3tears-recommended default trigger for a pod backed
by a JetStream durable consumer), or Prometheus / cpu / memory where that fits a pod
better.

### The ops contract -- two knobs

1. Flip the global master switch once, in your consuming chart's own `values.yaml`:

   ```yaml
   autoscaling:
     enabled: true
   ```

2. Per pod, flip that pod's switch and (optionally) raise its ceiling:

   ```yaml
   worker:
     autoscaling:
       enabled: true
       maxReplicas: 20   # everything else already ships sane defaults
   ```

A pod's `ScaledObject` renders **only when BOTH** `autoscaling.enabled` AND that pod's
`<pod>.autoscaling.enabled` are true. Both default `false` in a well-behaved consuming
chart, so nothing autoscales until an operator opts in, and a KEDA-less cluster renders
no `keda.sh` object at all.

### Consuming the helper

Add one line per autoscalable pod, in a `templates/<pod>-scaledobject.yaml` in your OWN
chart (not this library chart):

```yaml
{{- include "threetears.scaledObject" (dict "context" $ "component" "worker" "config" .Values.worker.autoscaling) }}
```

`component` must match the component label your own `templates/<pod>-deployment.yaml`
uses, since the `ScaledObject`'s `scaleTargetRef` is derived as
`<fullname>-<component>`. See `templates/_helpers.tpl`'s `threetears.scaledObject`
docstring for the full parameter/values-block contract, and `ci/example-consumer/` for
a complete worked example.

### KEDA is an opt-in prerequisite, not bundled runtime

KEDA installs cluster-wide CRDs + an operator, so a consuming chart must not fight a
cluster that already runs a shared KEDA. Two install modes, gated by `keda.install` (set
in the CONSUMING chart's values, passed through to this library's vendored subchart):

- **Recommended (default, `keda.install: false`)** -- install KEDA once, out of band, as
  a shared cluster add-on:

  ```bash
  helm repo add kedacore https://kedacore.github.io/charts
  helm install keda kedacore/keda -n keda --create-namespace
  ```

- **Opt-in bundled (`keda.install: true`)** -- let this library's vendored KEDA subchart
  dependency (see `Chart.yaml`) get installed as part of your consuming chart's release.
  Only on a cluster where no other KEDA exists (two operators reconciling the same CRDs
  is split-brain). Values under a `keda:` block in your consuming chart's values pass
  through to the subchart. The subchart is vendored under `charts/` here and pinned at
  `2.20.1`, matching the Hub's own pin.

### Proving the helper

`ci/example-consumer/` is a minimal, throwaway consumer chart (not a real deployable)
that depends on this library via a relative `file://` path and renders one worked
example `ScaledObject`. It mirrors the Hub's own worked-example-plus-CI-overlay pattern:

```bash
cd charts/3tears-keda-autoscaling
helm dependency build ci/example-consumer
helm template ci/example-consumer  # renders NO ScaledObject (both gates default off)
helm template ci/example-consumer -f ci/example-consumer/ci-values.yaml \
  --show-only templates/example-scaledobject.yaml
# renders a valid nats-jetstream ScaledObject
```

## Chart placement note

This is the first non-Python deliverable in the 3tears repo (a pure Python uv-workspace
monorepo otherwise). It lives under a new top-level `charts/` directory, confirmed with
the 3tears maintainer at the time this chart was lifted (3tears-migration shard 19b,
`14-eng-ai-survey/docs/3tears-migration/19b-3tears-keda-chart-lift-proposal.md`) rather
than assumed unilaterally.
