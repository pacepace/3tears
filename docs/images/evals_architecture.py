"""Write the evals architecture diagram the root README shows, as two static SVGs.

Run it from anywhere; it writes ``evals-architecture-light.svg`` and ``evals-architecture-dark.svg``
beside itself:

    python docs/images/evals_architecture.py

Each SVG is self-contained: no script, no external font, so GitHub renders it as an image and the text
falls back to the reader's system sans-serif. The README's numbered legend follows each arrow's label
below, so a change to an arrow's number changes the legend too.
"""

import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = html.escape

# ---------------------------------------------------------------- boxes
# (id, x, y, w, h, title, [(text, cls)], w4)
BOXES = [
    ("op", 224, 14, 376, 40, "Operator", [("an agent or Claude client on MCP, or a person in the web UI", "s")], False),
    # 3tears modules a host may use (left)
    (
        "core",
        28,
        268,
        156,
        72,
        "or a 3tears core store",
        [("DataStore · L1/L2/L3,", "s"), ("one the host already", "s"), ("runs for its own data", "s")],
        False,
    ),
    (
        "observe",
        28,
        360,
        156,
        72,
        "or 3tears observe",
        [("init_telemetry builds", "s"), ("the OTel pipeline the", "s"), ("sink harvests", "s")],
        False,
    ),
    (
        "tmodels",
        28,
        452,
        156,
        248,
        "or 3tears-models",
        [
            ("the direct-HTTP client", "s"),
            ("(Model, Catalogue, wire,", "s"),
            ("retry, published rates)", "s"),
            ("", "s"),
            ("or a LangChain chat", "s"),
            ("model, create_chat_model", "s"),
            ("", "s"),
            ("the object under test", "s"),
            ("runs on it too", "s"),
        ],
        False,
    ),
    # host: internals (left col), adapters (right col, facing evals)
    ("mcp", 404, 112, 196, 44, "MCP server", [("evals · evals_admin · evals_analyst", "s")], False),
    ("web", 404, 168, 196, 44, "Web UI", [("campaigns · launch", "s")], False),
    ("db", 224, 236, 156, 56, "Host's own database", [("its own SQL, document", "s"), ("or other store", "s")], False),
    ("adapter", 404, 236, 196, 56, "Eval DB adapter", [("host code: implements", "s"), ("DocumentStore", "s")], False),
    ("otel", 224, 330, 156, 48, "Host's OTel pipeline", [("its own TracerProvider", "s")], False),
    ("trace", 404, 330, 196, 48, "Trace sink", [("host code: TraceSink", "s")], False),
    (
        "out",
        224,
        430,
        376,
        140,
        "Object Under Test",
        [
            ("anything behind the engine's CandidateKind seam:", "s"),
            ("the host's own kinds: a persona or chat agent,", "s"),
            ("a classifier (the reference host's two)", "s"),
            ("kinds the engine ships: an eval judge, a report", "s"),
            ("writer, any callable: a pipeline, a tool-using", "s"),
            ("agent, an agent acting on a world", "s"),
        ],
        False,
    ),
    ("sdk", 224, 600, 156, 56, "or its own SDK", [("an OpenAI, OpenRouter", "s"), ("or Anthropic client", "s")], False),
    (
        "clients",
        404,
        600,
        196,
        56,
        "Model adapter",
        [("builds a client per role;", "s"), ("supplies the host's rates", "s")],
        False,
    ),
    (
        "rview",
        404,
        844,
        196,
        56,
        "Report page",
        [("web UI: sandboxed iframe;", "s"), ("the owner reads it here", "s")],
        False,
    ),
    # evals: ports (left col, facing the host)
    (
        "cat",
        640,
        112,
        160,
        100,
        "Catalogue · ops",
        [("MCP mount (FastMCP)", "s"), ("and REST over ops;", "s"), ("analysis_generate, job_poll", "s")],
        False,
    ),
    ("port", 640, 236, 160, 56, "Storage port", [("EvalStorage over", "s"), ("DocumentStore", "s")], False),
    ("tport", 640, 330, 160, 48, "TraceSink port", [("the cell's spans", "s")], False),
    ("kport", 640, 466, 160, 68, "Kind seam", [("CandidateKind:", "s"), ("prepare, invoke", "s")], False),
    (
        "mports",
        640,
        584,
        160,
        96,
        "Model ports",
        [
            ("CompletionClient", "m"),
            ("ToolTurnClient", "m"),
            ("bounds from host rates,", "s"),
            ("replay, ProviderError", "s"),
        ],
        False,
    ),
    (
        "rport",
        640,
        844,
        160,
        56,
        "report_read",
        [("serves the stored doc;", "s"), ("old plans: regenerate", "s")],
        False,
    ),
    # evals: run column
    (
        "runner",
        832,
        112,
        252,
        448,
        "Launch and runner",
        [
            ("run_launch → one EvalRun per arm", "s"),
            ("execute_run: case × model × k", "s"),
            ("cells, each through the kind", "s"),
            ("seam, then judged", "s"),
        ],
        False,
    ),
    ("sim", 832, 584, 120, 60, "Simulator", [("role simulator", "s")], False),
    ("judge", 964, 584, 120, 60, "Judge", [("role judge", "s")], False),
    # evals: report column
    ("bundle", 1112, 112, 176, 48, "Evidence bundle", [("→ report-input/1", "s")], False),
    ("snap", 1112, 268, 376, 44, "Report snapshot", [("EvalReportSnapshot · frozen, by id", "s")], False),
    (
        "analyst",
        1112,
        340,
        164,
        68,
        "Analyst (LLM)",
        [("writes the report", "s"), ("100 calls, 30/revision", "s")],
        False,
    ),
    ("checks", 1112, 440, 164, 64, "Checks (code)", [("validity + invariants", "s"), ("doc_verdicts", "m")], False),
    (
        "editor",
        1112,
        536,
        164,
        80,
        "Editor (LLM)",
        [("reviews the page", "s"), ("pass: no 0, total ≥ 11,", "s"), ("no must_fix", "s")],
        False,
    ),
    (
        "tools",
        1324,
        340,
        164,
        292,
        "evals_analyst tool",
        [
            ("read-only queries over", "s"),
            ("the frozen snapshot", "s"),
            ("", "s"),
            ("analyst cut:", "t2"),
            ("every workflow, incl.", "s"),
            ("blocks and report_check", "s"),
            ("", "s"),
            ("editor cut:", "t2"),
            ("facts, analyses, gaps;", "s"),
            ("no blocks", "s"),
        ],
        False,
    ),
    (
        "gapcall",
        1334,
        564,
        144,
        56,
        "capability_gap",
        [("missing or suspect;", "s"), ("need · why · workaround", "s")],
        False,
    ),
    (
        "endings",
        1112,
        656,
        376,
        100,
        "Endings",
        [
            ("stored: a review passes", "s"),
            ("stored_with_findings: 2 revisions used", "s"),
            ("over_budget: the last reviewed draft, if any", "s"),
            ("refused (still faulty after 3 bounces), review_failed,", "s"),
            ("failed: nothing stored; tally and sessions kept", "s"),
        ],
        False,
    ),
    (
        "gate",
        1112,
        780,
        376,
        48,
        "Store gate",
        [("EvalReportDoc + linked EvalAnalysis; sessions → transcripts", "s")],
        False,
    ),
    ("page", 1112, 852, 376, 48, "Doc page", [("doc_page.build → HTML · Markdown, with its made line", "s")], False),
    ("gaplist", 1112, 940, 164, 56, "analyst_gaps_list", [("grouped, ranked,", "s"), ("issue-ready text", "s")], False),
    (
        "gapstore",
        1324,
        940,
        164,
        56,
        "eval_analyst_gap",
        [("durable, with provenance;", "s"), ("code notices too", "s")],
        False,
    ),
    # third party
    ("openrouter", 228, 1056, 160, 52, "OpenRouter", [("openrouter.ai/api/v1", "s")], False),
    ("anthropic", 404, 1056, 160, 52, "Anthropic API", [("Messages", "s")], False),
    ("github", 580, 1056, 160, 52, "GitHub issues", [("one per gap group", "s")], False),
    # outside any run
    (
        "digest",
        784,
        1066,
        160,
        52,
        "Gap digest job",
        [("posts the digest, files", "s"), ("issues, own credentials", "s")],
        False,
    ),
    (
        "person",
        784,
        1150,
        128,
        72,
        "Coding agent",
        [("Claude Code or any", "s"), ("other, with the", "s"), ("approved issue", "s")],
        False,
    ),
    (
        "session",
        932,
        1150,
        128,
        72,
        "3tears change",
        [("packages/evals: an", "s"), ("action or skill text;", "s"), ("tests, Critic", "s")],
        False,
    ),
    (
        "release",
        1080,
        1150,
        128,
        72,
        "Merge or release",
        [("tag vX.Y.Z → PyPI,", "s"), ("or a pinnable", "s"), ("commit", "s")],
        False,
    ),
    (
        "pin",
        1228,
        1150,
        128,
        72,
        "Host pin",
        [("pyproject pins", "s"), ("3tears-evals", "s"), ("to the version", "s")],
        False,
    ),
]

ZONES = [
    ("z-3t", 16, 76, 180, 640, "3TEARS · MODULES", 28, 93),
    ("z-host", 212, 76, 400, 940, "HOST APP", 224, 93),
    ("z-3t", 628, 76, 892, 940, "3TEARS · EVALS (threetears.evals)", 640, 93),
    ("z-third", 212, 1040, 540, 100, "THIRD PARTY", 224, 1132),
    ("z-dev", 768, 1040, 752, 220, "OUTSIDE ANY RUN · THE GAP LOOP AND DEVELOPMENT", 780, 1252),
]

# (d, label, accent, w4, both_ends, badges, style)  style: "" | "dot" | "dev" | "join" | "dotjoin"
ARROWS = [
    ("M500,54 V112", "1", False, False, False, [(500, 83)], ""),
    ("M320,54 V190 H404", "1", False, False, False, [(320, 83)], ""),
    ("M600,134 H620 V162 H640", "2", False, False, False, [(620, 148)], ""),
    ("M600,190 H620 V162", "", False, False, False, [], "join"),
    # storage
    ("M640,264 H600", "3", False, False, False, [(620, 264)], ""),
    ("M800,264 H832", "3", False, False, True, [(816, 264)], ""),
    ("M404,264 H380", "", False, False, False, [], ""),
    ("M440,292 V306 H184", "", False, False, False, [], ""),
    # launch, cells
    ("M800,150 H832", "4", False, False, False, [(816, 150)], ""),
    ("M832,500 H800", "5", False, False, False, [(816, 500)], ""),
    ("M640,500 H600", "5", False, False, False, [(620, 500)], ""),
    ("M224,500 H184", "6", False, False, False, [(204, 500)], ""),
    ("M892,560 V584", "8", False, False, False, [(892, 572)], ""),
    ("M1024,560 V584", "8", False, False, False, [(1024, 572)], ""),
    # models
    ("M892,644 V668 H800", "10", False, False, False, [(840, 668)], ""),
    ("M1024,644 V668 H892", "", False, False, False, [], "join"),
    ("M640,628 H600", "9", False, False, False, [(620, 628)], ""),
    ("M404,628 H380", "9", False, False, False, [], ""),
    ("M420,600 V588 H184", "9", False, False, False, [(300, 588)], ""),
    ("M106,700 V1026", "7", False, False, False, [(106, 900)], "join"),
    ("M330,656 V1026", "11", False, False, False, [(330, 900)], "join"),
    ("M106,1026 H484", "", False, False, False, [], "join"),
    ("M308,1026 V1056", "", False, False, False, [], ""),
    ("M484,1026 V1056", "", False, False, False, [], ""),
    # tracing
    ("M380,354 H404", "", False, False, False, [], ""),
    ("M440,378 V392 H184", "", False, False, False, [], ""),
    ("M600,354 H640", "12", False, False, False, [(620, 354)], ""),
    ("M800,354 H832", "12", False, False, False, [(816, 354)], ""),
    # analysis
    ("M1084,136 H1112", "13", False, False, False, [(1098, 136)], ""),
    ("M1200,160 V268", "14", True, False, False, [(1200, 214)], ""),
    ("M760,112 V100 H1440 V268", "15", True, False, False, [(1100, 100)], ""),
    ("M1176,312 V340", "16", True, False, False, [(1176, 326)], ""),
    ("M1400,312 V340", "16", False, False, False, [(1400, 326)], ""),
    ("M1276,356 H1324", "17", False, False, True, [(1300, 356)], ""),
    ("M1276,608 H1324", "17", False, False, True, [(1300, 608)], ""),
    ("M1112,374 H1104 V700 H960 V668", "18", False, False, False, [(1030, 700)], "join"),
    ("M1112,576 H1104", "", False, False, False, [], "join"),
    ("M1150,408 V440", "19", True, False, False, [(1150, 424)], ""),
    ("M1210,440 V408", "20", False, False, False, [(1210, 424)], ""),
    ("M1150,504 V536", "21", True, False, False, [(1150, 520)], ""),
    ("M1276,586 H1300 V388 H1276", "22", False, False, False, [(1300, 470)], ""),
    ("M1190,616 V656", "23", True, False, False, [(1190, 636)], ""),
    ("M1300,756 V780", "24", True, False, False, [(1300, 768)], ""),
    ("M1300,828 V852", "25", True, False, False, [(1300, 840)], ""),
    ("M1112,876 H800", "26", True, False, False, [(960, 876)], ""),
    ("M640,872 H600", "27", True, False, False, [(620, 872)], ""),
    # gap loop and new capability
    ("M1478,592 H1504 V968 H1488", "28", False, False, False, [(1504, 780)], ""),
    ("M1324,968 H1276", "29", False, False, False, [(1300, 968)], ""),
    ("M1194,996 V1092 H944", "30", False, False, False, [(1194, 1060)], ""),
    ("M784,1092 H740", "31", False, False, False, [(762, 1092)], ""),
    ("M660,1108 V1186 H784", "32", False, False, False, [(700, 1186)], "dev"),
    ("M912,1186 H932", "33", False, False, False, [(922, 1186)], "dev"),
    ("M1060,1186 H1080", "34", False, False, False, [(1070, 1186)], "dev"),
    ("M1208,1186 H1228", "35", False, False, False, [(1218, 1186)], "dev"),
    ("M1356,1186 H1532 V480 H1488", "36", False, False, False, [(1532, 800)], "dev"),
]
W4_BADGES = []
JOINS = [(620, 162), (1104, 576), (892, 668), (960, 668), (106, 1026), (330, 1026)]
TEXTS = [  # small labels, on the marks
    ("≤3", 1220, 428, 0),
    ("revise ≤2", 1296, 540, -90),
    ("that a host may use", 28, 107, 0),
    ("owner approves", 668, 1160, 0),
]
FILL = {
    **{k: "bh" for k in "mcp web rview adapter db trace otel out clients sdk pin digest".split()},
    **{k: "bd" for k in "release session person".split()},
    **{k: "bx" for k in "openrouter anthropic github".split()},
    "op": "bn",
}
OPTION = {"db", "core", "otel", "observe", "sdk", "tmodels"}
LABEL = (
    "The designed evals flow. A question enters the Host App over MCP or its web UI and reaches the engine's catalogue. "
    "The engine runs cells against the host's object under test and reads, writes, traces and calls models only "
    "through ports the host fills, each with an adapter backed by the host's own system or a 3tears module. "
    "analysis_generate freezes the evidence, the analyst and editor write and review the report, and the host "
    "serves the stored page. A gap the analyst reports becomes an issue, then a change to 3tears built outside any run."
)


# ---------------------------------------------------------------- render
# Arial-metric faces first: the boxes are sized for them, and every desktop has one (Helvetica on macOS, Arial on
# Windows, Liberation Sans on most Linux). A wider fallback can run a long line to a box's edge.
SANS = '"Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'
WIDTH, HEIGHT = 1548, 1272

LIGHT = {
    "bg": "#f6f6f2",
    "fg": "#1c2329",
    "muted": "#4f5963",
    "line": "#46515c",
    "accent": "#b4441b",
    "box": "#ffffff",
    "boxline": "#a3adb7",
    "z-host": "#eedabf",
    "z-3t": "#acc8d9",
    "z-third": "#e5e5e5",
    "z-dev": "#f8f7fa",
    "hatch": "#d8cfe6",
    "host-ln": "#b0925a",
    "dev": "#6a4c9c",
    "sub": "#6f8aa3",
    "box-h": "#fff7ec",
    "box-t": "#c3dbea",
    "box-x": "#f9f9f9",
    "box-d": "#fdfcff",
    "box-n": "#ffffff",
}
DARK = {
    "bg": "#121519",
    "fg": "#e5e8eb",
    "muted": "#b3bdc7",
    "line": "#a9b4bf",
    "accent": "#f08a5d",
    "box": "#1b2026",
    "boxline": "#4a5561",
    "z-host": "#272117",
    "z-3t": "#203541",
    "z-third": "#2a2a2a",
    "z-dev": "#17181c",
    "hatch": "#3a3446",
    "host-ln": "#7a6638",
    "dev": "#b9a2e6",
    "sub": "#6b8296",
    "box-h": "#332b1f",
    "box-t": "#304a59",
    "box-x": "#393939",
    "box-d": "#2f2e32",
    "box-n": "#1b2026",
}

CSS = """
.bg { fill: var(--bg); }
.z-host { fill: var(--z-host); stroke: var(--host-ln); stroke-width: 1.2; }
.hostrepo { fill: none; stroke: var(--host-ln); stroke-width: 1.2; }
.z-3t { fill: var(--z-3t); }
.z-third { fill: var(--z-third); stroke: var(--boxline); stroke-width: 1.2; stroke-dasharray: 1.5 3; stroke-linecap: round; }
.z-dev { fill: url(#hatch); stroke: var(--dev); stroke-width: 1.2; stroke-dasharray: 7 3 2 3; }
.hbg { fill: var(--z-dev); } .hln { stroke: var(--hatch); stroke-width: 1.4; }
.subzone { fill: none; stroke: var(--sub); stroke-dasharray: 3 3; }
text { font-family: SANS; }
.zl { font-size: 10.5px; font-weight: 600; letter-spacing: .08em; fill: var(--fg); }
.box { fill: var(--box); stroke: var(--boxline); stroke-width: 1; }
.box.bh { fill: var(--box-h); } .box.bt { fill: var(--box-t); } .box.bx { fill: var(--box-x); }
.box.bd { fill: var(--box-d); } .box.bn { fill: var(--box-n); }
.box.opt { stroke: var(--line); stroke-dasharray: 4 3; }
.t { font-size: 13px; font-weight: 600; fill: var(--fg); }
.t2 { font-size: 11px; font-weight: 600; fill: var(--fg); }
.s { font-size: 11px; fill: var(--muted); }
.m { font-family: MONO; font-size: 10.5px; fill: var(--muted); }
.lbl { font-size: 10px; font-weight: 500; fill: var(--muted); }
.ln { fill: none; stroke: var(--line); stroke-width: 1.3; }
.ln.dot { stroke-dasharray: 1.5 3.5; stroke-width: 2; stroke-linecap: round; }
.ln.ac { stroke: var(--accent); stroke-width: 2; }
.ln.dev { stroke: var(--dev); stroke-width: 1.8; stroke-dasharray: 8 3 2 3; }
.jn { fill: var(--line); }
.ahp { fill: var(--line); } .ahpA { fill: var(--accent); } .ahpD { fill: var(--dev); }
.badge circle { fill: var(--bg); stroke: var(--line); stroke-width: 1; }
.badge text { font-size: 9.5px; font-weight: 600; fill: var(--fg); text-anchor: middle; }
.badge.ac circle { stroke: var(--accent); } .badge.ac text { fill: var(--accent); }
.badge.dv circle { stroke: var(--dev); } .badge.dv text { fill: var(--dev); }
""".replace("SANS", SANS).replace("MONO", MONO)

DEFS = (
    "<defs>"
    '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
    'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="ahp" d="M0,1 L10,5 L0,9 z"/></marker>'
    '<marker id="ahA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
    'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="ahpA" d="M0,1 L10,5 L0,9 z"/></marker>'
    '<marker id="ahD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
    'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="ahpD" d="M0,1 L10,5 L0,9 z"/></marker>'
    '<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<rect class="hbg" width="7" height="7"/><line class="hln" x1="0" y1="0" x2="0" y2="7"/></pattern>'
    "</defs>"
)


def box_svg(b: tuple) -> str:
    bid, x, y, w, h, title, lines, _w4 = b
    fill = FILL.get(bid, "bt") + (" opt" if bid in OPTION else "")
    out = [f'<g id="{bid}"><rect class="box {fill}" x="{x}" y="{y}" width="{w}" height="{h}" rx="3"/>']
    ty = y + (25 if not lines else 19)
    out.append(f'<text class="t" x="{x + 10}" y="{ty}">{E(title)}</text>')
    for i, (t, c) in enumerate(lines):
        if t:
            out.append(f'<text class="{c}" x="{x + 10}" y="{ty + 16 + 14.5 * i:g}">{E(t)}</text>')
    out.append("</g>")
    return "".join(out)


def zone_svg(z: tuple) -> str:
    zc, x, y, w, h, label, lx, ly = z
    return (
        f'<rect class="{zc}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        f'<text class="zl" x="{lx}" y="{ly}">{E(label)}</text>'
    )


def badge(lbl: str, x: float, y: float, accent: bool, dev: bool) -> str:
    c = "badge" + (" ac" if accent else "") + (" dv" if dev else "")
    return f'<g class="{c}"><circle cx="{x}" cy="{y}" r="8.5"/><text x="{x}" y="{y + 3.6:g}">{lbl}</text></g>'


def arrow_svg(a: tuple) -> str:
    d, lbl, accent, _w4, both, badges, style = a
    dev = style == "dev"
    cls = "ln" + (" ac" if accent else "") + (" dot" if style in ("dot", "dotjoin") else "") + (" dev" if dev else "")
    m = "ahA" if accent else ("ahD" if dev else "ah")
    end = "" if style in ("join", "dotjoin") else f' marker-end="url(#{m})"'
    start = f' marker-start="url(#{m})"' if both else ""
    return (
        f'<g><path class="{cls}" d="{d}"{end}{start}/>'
        + "".join(badge(lbl, bx, by, accent, dev) for bx, by in badges)
        + "</g>"
    )


def svg(tokens: dict[str, str]) -> str:
    css = CSS
    for key in sorted(tokens, key=len, reverse=True):  # bake each colour in, so no renderer needs CSS variables
        css = css.replace(f"var(--{key})", tokens[key])
    p = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        f'role="img" aria-label="{E(LABEL)}">',
        f"<title>{E(LABEL)}</title>",
        f"<style>{css}</style>",
        DEFS,
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
    ]
    p += [zone_svg(z) for z in ZONES]
    p.append(
        '<rect class="z-host" x="1222" y="1130" width="140" height="100" rx="5"/>'
        '<text class="zl" x="1230" y="1144">HOST APP REPO</text>'
        '<rect class="z-host" x="776" y="1048" width="176" height="78" rx="5"/>'
        '<text class="zl" x="784" y="1062">HOST APP · SCHEDULED</text>'
        '<rect class="subzone" x="1100" y="240" width="404" height="676" rx="5"/>'
        '<text class="zl" x="1212" y="256">REPORT WRITER · write_report</text>'
    )
    p += [arrow_svg(a) for a in ARROWS]
    p += [box_svg(b) for b in BOXES]
    p += [f'<circle class="jn" cx="{x}" cy="{y}" r="2.6"/>' for x, y in JOINS]
    p += [
        f'<text class="lbl" x="{x}" y="{y}"' + (f' transform="rotate({r} {x} {y})"' if r else "") + f">{E(t)}</text>"
        for t, x, y, r in TEXTS
    ]
    p.append("</svg>\n")
    return "".join(p)


if __name__ == "__main__":
    for name, tokens in (("light", LIGHT), ("dark", DARK)):
        out = HERE / f"evals-architecture-{name}.svg"
        out.write_text(svg(tokens), encoding="utf-8")
        print(out)
