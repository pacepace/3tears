{{/*
Expand the name of a consuming chart. Mirrors the standard Helm starter-chart helper
(lifted from 14-eng-ai-bot/k8s/helm/aibots/templates/_helpers.tpl's threetears.name
equivalent, aibots.name) so a consumer chart can call this via the library dependency
instead of re-declaring its own copy.
*/}}
{{- define "threetears.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name for a consuming chart.
*/}}
{{- define "threetears.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels for a consuming chart's resources.
*/}}
{{- define "threetears.labels" -}}
helm.sh/chart: {{ include "threetears.name" . }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "threetears.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
threetears.scaledObject -- reusable KEDA ScaledObject renderer.

Lifted from 14-eng-ai-bot/k8s/helm/aibots/templates/_helpers.tpl's aibots.scaledObject
(docs/resilience/scale-task-00-keda-chart-gate.md), generalized out of the aibots.*
namespace so any 3tears consumer chart can depend on this library instead of copying the
Hub's inline copy. Behavior is unchanged from the source pattern.

Renders a keda.sh/v1alpha1 ScaledObject targeting a Deployment named
<fullname>-<component>, dual-gated on the consuming chart's GLOBAL `autoscaling.enabled`
AND the per-component config's `enabled` flag -- both must be true or this renders
NOTHING. Trigger-agnostic: the `triggers` list passes through verbatim (nats-jetstream,
cpu, prometheus, ... whatever KEDA supports).

Usage from a CONSUMING chart (one that has taken this chart as a Helm dependency, e.g.
in its own Chart.yaml: `dependencies: [{name: 3tears-keda-autoscaling, ...}]`):

    {{- include "threetears.scaledObject" (dict "context" $ "component" "worker" "config" .Values.worker.autoscaling) }}

Parameters (single dict argument):
  context   - the root context ($) of the CONSUMING chart, for fullname/labels + the
              global autoscaling.enabled gate. Must expose .Values.autoscaling.enabled
              (bool) at minimum.
  component - the pod component label; the scaleTargetRef Deployment name is derived as
              <fullname>-<component>, so the consuming chart's own Deployment template
              must produce that same name
  config    - the pod's `<pod>.autoscaling` values block, supplying:
                enabled          - per-pod opt-in (bool)
                minReplicas      - scale floor (int)
                maxReplicas      - scale ceiling (int)
                pollingInterval  - seconds between trigger evaluations (int, optional)
                cooldownPeriod   - seconds of inactivity before scaling back to
                                   minReplicas (int, optional)
                triggers         - KEDA trigger list, emitted verbatim. A NATS-JetStream
                                   depth trigger (the 3tears-recommended default for a
                                   pod backed by a JetStream durable consumer, e.g.
                                   14-eng-ai-survey's shard 19 ExportJobsData redesign)
                                   looks like:
                                     - type: nats-jetstream
                                       metadata:
                                         natsServerMonitoringEndpoint: "nats:8222"
                                         account: "$G"
                                         stream: <jetstream-stream>
                                         consumer: <durable-consumer>
                                         lagThreshold: "<per-replica target lag>"
                                         activationLagThreshold: "<lag to scale 0->1>"
                                   A cpu/memory/prometheus trigger is equally valid; the
                                   helper passes the list through unmodified.
*/}}
{{- define "threetears.scaledObject" -}}
{{- $ctx := .context -}}
{{- $config := .config -}}
{{- if and $ctx.Values.autoscaling.enabled $config.enabled -}}
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {{ include "threetears.fullname" $ctx }}-{{ .component }}
  labels:
    {{- include "threetears.labels" $ctx | nindent 4 }}
    app.kubernetes.io/component: {{ .component }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "threetears.fullname" $ctx }}-{{ .component }}
  minReplicaCount: {{ $config.minReplicas }}
  maxReplicaCount: {{ $config.maxReplicas }}
  {{- with $config.pollingInterval }}
  pollingInterval: {{ . }}
  {{- end }}
  {{- with $config.cooldownPeriod }}
  cooldownPeriod: {{ . }}
  {{- end }}
  triggers:
    {{- toYaml $config.triggers | nindent 4 }}
{{- end -}}
{{- end }}
