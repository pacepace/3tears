"""Normalize a campaign writer-input package into the report input the generator reads.

Usage: python3 -I bundle_to_report.py <writer-input.json> <out.json> [--examples 12]

Rules this adapter follows (see GENERATOR.md):
- Never recompute a statistic the pipeline already produced. Readings, floors, contrasts,
  intervals and judge tiers are copied through as given.
- Arms are keyed by their campaign name (alias). Each arm also carries its served model and the
  settings that differ from the control, so the generator can build short names.
- Everything is grouped by what the page needs, not by how the pipeline stores it.
- The shape A-C references (r1-r3) use simpler synthetic data built for their pages, not this schema;
  read them for page structure, and this adapter's output (schema/report-input.schema.json) for data.
"""

import json, re, sys, collections


def load(path):
    d = json.load(open(path))
    msg = d["writer_user_message"]
    i = msg.index("{")
    summary, _ = json.JSONDecoder().raw_decode(msg[i:])
    return d, d["bundle"], summary


def judge_reason(judge_str, dim):
    # judge verdicts arrive as a list of dicts, or as a truncated repr string; pull the reason for one dim
    if isinstance(judge_str, list):
        for j in judge_str:
            if isinstance(j, dict) and j.get("dim") == dim:
                return (j.get("reason") or "").strip() or None
        return None
    m = re.search(r"'dim': '%s'.*?'reason': ['\"](.*?)(?:['\"]\}|…|$)" % re.escape(dim), judge_str or "", re.S)
    return m.group(1).strip() if m else None


def main(src, out, n_examples=12):
    d, b, s = load(src)
    asp_full = b["aspects"][0]
    aspect = s["aspects"][0]  # per-aspect summary: cells, measures, contrasts, disclosures live here
    # ---- arms: name -> role, served model, variant key, settings that differ from control
    served = {x["arm"]: x["served_models"][0] for x in aspect.get("disclosures", {}).get("arm_served_models", [])}
    vk_by_model = {a["levers"]["model"]["display"]: a["variant_key"] for a in b["arms"]}
    levers_by_vk = {a["variant_key"]: {k: v.get("display") for k, v in a["levers"].items()} for a in b["arms"]}
    arms = []
    for a in s["arms"]["arms"]:
        model = served.get(a["arm"], a["levels"].get("model"))
        vk = vk_by_model.get(model)
        arms.append(
            dict(
                id=a["arm"],
                name=a["arm"],
                role="control" if a["role"] == "control" else "challenger",
                served_model=model,
                variant_key=vk,
                settings={"model": model},
            )
        )
    ctl = next(a for a in arms if a["role"] == "control")
    for a in arms:
        if a is ctl or not a["variant_key"] or not ctl["variant_key"]:
            continue
        arm_levers, ctl_levers = levers_by_vk[a["variant_key"]], levers_by_vk[ctl["variant_key"]]
        diff = {k: arm_levers[k] for k in arm_levers if arm_levers.get(k) != ctl_levers.get(k)}
        a["settings"].update(diff)
    arm_by_vk = {a["variant_key"]: a["id"] for a in arms}
    mech = {m["arm"]: m for m in aspect.get("arm_mechanisms", [])}
    for a in arms:
        if a["id"] in mech:
            a["reasoning_ratio"] = mech[a["id"]]["mean"]

    # ---- per-arm measures and judged dims (from the summary cells)
    cell_arm = {c["cell"]: c["arm"] for c in aspect["cells"]}
    measures = {}
    for name, meta in aspect["measures"].items():
        per = {}
        for c in aspect["cells"]:
            m = c["measures"].get(name)
            if m and "mean" in m:
                per[c["arm"]] = {k: m.get(k) for k in ("mean", "ci", "n", "n_cases", "p95")}
        measures[name] = dict(
            label=None,
            unit=meta.get("unit"),
            higher_is_better=meta.get("higher_is_better"),
            merit_axis=meta.get("merit_axis"),
            means=meta.get("means"),
            per_arm=per,
        )
    tiers = {t["rubric_dim"]: t for t in aspect.get("judge_evidence_tiers", [])}
    judged = {}
    for c in aspect["cells"]:
        for dim, m in c["judged"].items():
            j = judged.setdefault(
                dim,
                dict(
                    scale=tiers.get(dim, {}).get("scale"),
                    tier=m.get("tier"),
                    judge_model=tiers.get(dim, {}).get("judge_model"),
                    calibration=tiers.get(dim, {}).get("calibration", {}).get("state"),
                    per_arm={},
                ),
            )
            j["per_arm"][c["arm"]] = dict(mean=m["mean"], n=m["n"], n_cases=m["n_cases"])

    # ---- contrasts (multiple-comparison corrected), keyed to arm names
    contrasts = []
    for c in aspect.get("contrasts", []):
        contrasts.append(
            dict(
                arm=cell_arm[c["cell"]],
                control=cell_arm[c["control"]],
                kind=c["reading"],
                measure=c["name"],
                verdict=c["verdict"],
                delta=c["delta"],
                interval=c["interval"],
                p_adjusted=c.get("p_adjusted"),
                materiality=c.get("materiality"),
                confounds=[x["dimension"] for x in c.get("mechanism_confounds", [])],
            )
        )
    families = aspect.get("contrast_families", [])

    # ---- aspects with their question, readings and floors
    aspects = []
    for a in s["aspects"]:
        q = a["question"]
        aspects.append(
            dict(
                id=a["aspect"],
                name=a["aspect"],
                margin=q.get("margin"),
                asked=q.get("asked"),
                exploratory_measures=q.get("reading_scope", {}).get("exploratory_measures", []),
                readings=[
                    {
                        k: r.get(k)
                        for k in (
                            "arm",
                            "reading",
                            "change",
                            "within_interval",
                            "gain_confidence",
                            "margin",
                            "control_passed",
                            "arm_passed",
                            "cases",
                            "confirmatory",
                            "why",
                            "precision",
                        )
                    }
                    for r in a["readings"]
                ],
                depth=a.get("depth", {}).get("arms"),
                pass_k=a.get("depth", {}).get("pass_k"),
            )
        )
    floors = s.get("floors", [])
    scoreboard = {
        x["arm"]: {
            k: x[k] for k in ("ran", "lost", "usable", "passed", "failed", "cases", "reply_cost_usd", "test_cost_usd")
        }
        for x in aspect.get("scoreboard", [])
    }

    # ---- strata from the full bundle: judged means per arm per stratum
    strata = collections.OrderedDict()
    for cm in asp_full["cell_measures"]:
        arm = arm_by_vk.get(cm["variant_key"])
        for st in cm["strata"]:
            row = strata.setdefault(st["stratum"], dict(id=st["stratum"], n_cases=st["n_cases"], per_arm={}))
            row["per_arm"][arm] = {j["dimension"]: dict(mean=j["mean"], sem=j["sem"], n=j["n"]) for j in st["judged"]}
    strata = sorted(strata.values(), key=lambda r: -r["n_cases"])

    # ---- examples: failed trials on the primary judged dim, one per (arm, stratum), candidates first,
    #      strata ordered by how much the candidates lose there (cases x drop), so the biggest loss is shown first
    prim = "persona.holds_character" if "persona.holds_character" in judged else next(iter(judged), None)
    loss = {}
    for st in strata:
        c = st["per_arm"].get(ctl["id"], {}).get(prim)
        if not c:
            continue
        loss[st["id"]] = sum(
            st["n_cases"] * max(0.0, c["mean"] - st["per_arm"][a["id"]][prim]["mean"])
            for a in arms
            if a is not ctl and prim in st["per_arm"].get(a["id"], {})
        )
    ex, seen = [], set()
    trials = asp_full["trials"]

    def unescape(text):
        try:
            return text.encode("latin-1", "backslashreplace").decode("unicode_escape")
        except UnicodeDecodeError, UnicodeEncodeError:
            return text

    for t in sorted(
        trials, key=lambda t: (arm_by_vk.get(t["arm"]) == ctl["id"], -loss.get(t["stratum"], 0), t["stratum"])
    ):
        arm = arm_by_vk.get(t["arm"])
        if not arm or (t.get("judged") or {}).get(prim) != 0 or (arm, t["stratum"]) in seen:
            continue
        seen.add((arm, t["stratum"]))
        ex.append(
            dict(
                trial=t["trial_id"],
                case=t["case_label"],
                stratum=t["stratum"],
                arm=arm,
                repeat=t["repeat"],
                reason=judge_reason(t.get("judge"), prim),
                excerpt=unescape(t.get("excerpt") or "")[:420],
            )
        )
        if len(ex) >= n_examples:
            break

    disc = aspect.get("disclosures", {})
    run_arm = {}
    for c in [aspect.get("design", {}).get("control_arm") or {}] + list(aspect.get("design", {}).get("contrasts", [])):
        for rid in c.get("run_ids", []):
            run_arm[rid] = c.get("arm")
    short_runs = [dict(run=r, arm=run_arm.get(r), note=v) for r, v in disc.get("short_runs", {}).items()]
    for f in families:
        if f.get("alpha") and f.get("family_size"):
            f["level"] = f"{100 - 100 * f['alpha'] / f['family_size']:.2f}%"
    report = dict(
        schema="report-input/1",
        meta=dict(
            campaign=s["campaign"]["name"],
            question=s["campaign"]["question"],
            behavior=s["campaign"].get("behavior"),
            kind=s.get("kind"),
            kind_means=s.get("kind_means"),
            design_shape=aspect.get("design", {}).get("shape"),
            run_date=(re.search(r"\d{4}-\d{2}-\d{2}", disc.get("time_axis_withheld") or "") or [None])[0],
            provenance=d.get("provenance"),
            synthetic=False,
        ),
        arms=arms,
        aspects=aspects,
        floors=floors,
        bars=s.get("bars", []),
        guardrails=aspect.get("guardrails", []),
        scoreboard=scoreboard,
        measures=measures,
        judged=judged,
        primary_judged=prim,
        contrasts=contrasts,
        contrast_families=families,
        strata=strata,
        examples=ex,
        caveats=dict(
            late_criteria=aspect.get("late_criteria", []),
            judge_tiers={k: v["tier"] for k, v in tiers.items()},
            confounds=aspect.get("confound_catalog", {}),
            apparatus_confounds=disc.get("apparatus_confounds", []),
            latency=disc.get("latency_contended"),
            short_runs=short_runs,
            unreported=disc.get("unreported_observations", []),
        ),
        prior_insights=s.get("prior_insights", []),
    )
    json.dump(report, open(out, "w"), indent=1)
    print(
        f"arms={[(a['id'], a['role'], a['served_model']) for a in arms]}  strata={len(strata)}  contrasts={len(contrasts)}  examples={len(ex)}"
    )


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[3]) if len(a) > 3 else 12)
