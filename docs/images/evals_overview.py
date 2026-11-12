"""Write the simplified evals diagram the root README shows, as two static SVGs.

Run it from anywhere; it writes ``evals-overview-light.svg`` and ``evals-overview-dark.svg`` beside itself:

    python docs/images/evals_overview.py

The canvas is narrower than GitHub's README column, so the SVG is shown at its own size and no text is
scaled below its set size. Each label leaves room for a wider fallback font than the Arial-metric faces
the boxes are sized for. The full diagram, with every arrow, is ``evals_architecture.py``.
"""

import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = html.escape
WIDTH, HEIGHT = 820, 692

# Arial-metric faces first: every desktop has one (Helvetica on macOS, Arial on Windows, Liberation Sans on
# most Linux).
SANS = '"Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif'

# The full diagram's palette. Host and 3tears differ in lightness and in outline, not only in hue: the
# host's zone has a solid outline and 3tears' none, an optional module is dashed, and the offline zone is
# hatched with a dash-dot outline.
LIGHT = {
    "bg": "#f6f6f2", "fg": "#1c2329", "muted": "#4f5963", "line": "#46515c", "accent": "#b4441b",
    "boxline": "#a3adb7", "z-host": "#eedabf", "z-3t": "#acc8d9", "z-dev": "#f8f7fa", "hatch": "#d8cfe6",
    "host-ln": "#b0925a", "dev": "#6a4c9c", "sub": "#6f8aa3", "box-h": "#fff7ec", "box-t": "#c3dbea",
    "box-d": "#fdfcff",
}  # fmt: skip
DARK = {
    "bg": "#121519", "fg": "#e5e8eb", "muted": "#b3bdc7", "line": "#a9b4bf", "accent": "#f08a5d",
    "boxline": "#4a5561", "z-host": "#272117", "z-3t": "#203541", "z-dev": "#17181c", "hatch": "#3a3446",
    "host-ln": "#7a6638", "dev": "#b9a2e6", "sub": "#6b8296", "box-h": "#332b1f", "box-t": "#304a59",
    "box-d": "#2f2e32",
}  # fmt: skip

# Rows shared by the optional modules, the host's adapters and the engine's ports.
ROW = {"ui": 96, "store": 184, "trace": 272, "model": 360, "kind": 448}
ROW_H = 64

# (x, y, w, h, title, lines, fill)  fill: "h" host, "t" 3tears, "o" optional 3tears module, "d" offline
BOXES = [
    # optional 3tears modules
    (24, ROW["store"], 144, ROW_H, "3tears core", ["a store over", "DataStore"], "o"),
    (24, ROW["trace"], 144, ROW_H, "3tears observe", ["the OTel pipeline", "it builds"], "o"),
    (24, ROW["model"], 144, ROW_H, "3tears-models", ["its model", "clients"], "o"),
    # the host's adapters
    (204, ROW["ui"], 156, ROW_H, "Web UI or MCP", ["asks, and reads", "the report"], "h"),
    (204, ROW["store"], 156, ROW_H, "Storage adapter", ["on its own", "database"], "h"),
    (204, ROW["trace"], 156, ROW_H, "Trace sink", ["optional; on its", "OTel pipeline"], "h"),
    (204, ROW["model"], 156, ROW_H, "Model adapter", ["on its own SDK", "client"], "h"),
    (204, ROW["kind"], 156, ROW_H, "Code under test", ["its production", "code, via a kind"], "h"),
    # evals
    (398, 88, 136, 424, "Engine", [], "t"),
    (576, 116, 212, 52, "Analyst (model)", ["writes the report"], "t"),
    (576, 196, 212, 52, "Checks (code)", ["validate each draft"], "t"),
    (576, 276, 212, 52, "Editor (model)", ["reviews the page"], "t"),
    (576, 356, 212, 52, "Stored report", ["read through the UI"], "t"),
    # offline
    (568, 576, 220, 76, "Coding agent", ["in the 3tears repo turns", "an approved gap into", "a change"], "d"),
    (316, 576, 200, 76, "Release or pin", ["a 3tears release, or", "a pinnable commit"], "d"),
]

# Text inside the engine, level with the arrow each port takes.
ENGINE_TEXT = [
    ("Engine", "t", 112),
    ("catalogue", "p", ROW["ui"] + 37),
    ("runs every case", "s", 166),
    ("and judges it", "s", 182),
    ("storage port", "p", ROW["store"] + 37),
    ("trace port", "p", ROW["trace"] + 37),
    ("model ports", "p", ROW["model"] + 37),
    ("kind seam", "p", ROW["kind"] + 37),
]

# (path, style, start_marker)  style: "" solid, "opt" dashed, "ac" the report's path, "dev" offline
ARROWS = [
    # optional modules, which an adapter may be built on
    (f"M204,{ROW['store'] + 32} H168", "opt", False),
    (f"M204,{ROW['trace'] + 32} H168", "opt", False),
    (f"M204,{ROW['model'] + 32} H168", "opt", False),
    # the host's adapters and the engine's ports
    (f"M360,{ROW['ui'] + 32} H398", "", True),
    (f"M398,{ROW['store'] + 32} H360", "", True),
    (f"M360,{ROW['trace'] + 32} H398", "", False),
    (f"M398,{ROW['model'] + 32} H360", "", False),
    (f"M398,{ROW['kind'] + 32} H360", "", False),
    # the report writer
    ("M534,132 H576", "ac", False),
    ("M640,168 V196", "ac", False),
    ("M640,248 V276", "ac", False),
    ("M640,328 V356", "ac", False),
    ("M788,222 H800 V150 H788", "", False),
    ("M576,302 H562 V142 H576", "", False),
    # the gap loop, outside any run
    ("M788,128 H812 V614 H788", "dev", False),
    ("M568,614 H516", "dev", False),
    ("M316,614 H282 V522", "dev", False),
]

# (text, x, y, cls, anchor, rotate)
LABELS = [
    ("bounce", 760, 188, "lbl", "end", 0),
    ("draft", 648, 188, "lbl", "start", 0),
    ("clean", 648, 268, "lbl", "start", 0),
    ("revise", 572, 268, "lbl", "start", 0),
    ("passes", 648, 348, "lbl", "start", 0),
    ("capability gap", 800, 564, "lbl", "end", 0),
    ("the host moves its pin", 274, 644, "lbl", "end", 0),
]

ZONES = [
    # (class, x, y, w, h, label, label_x, label_y)
    ("z-3t", 12, 156, 168, 276, "3TEARS, OPTIONAL", 24, 176),
    ("z-host", 192, 36, 180, 484, "HOST APP", 204, 60),
    ("z-3t", 384, 36, 432, 484, "3TEARS-EVALS", 398, 60),
    ("z-dev", 12, 540, 804, 140, "OFFLINE: THE TOOL-IMPROVEMENT LOOP", 24, 564),
]
NOTES = [  # (text, x, y, cls)
    ("depends on no other 3tears package", 398, 78, "note"),
    ("REPORT WRITER", 560, 106, "zl"),
]

CSS = """
.bg { fill: var(--bg); }
.z-host { fill: var(--z-host); stroke: var(--host-ln); stroke-width: 1.4; }
.z-3t { fill: var(--z-3t); }
.z-dev { fill: url(#hatch); stroke: var(--dev); stroke-width: 1.4; stroke-dasharray: 7 3 2 3; }
.hbg { fill: var(--z-dev); } .hln { stroke: var(--hatch); stroke-width: 1.4; }
.subzone { fill: none; stroke: var(--sub); stroke-dasharray: 3 3; }
text { font-family: SANS; }
.zl { font-size: 12px; font-weight: 700; letter-spacing: .06em; fill: var(--fg); }
.note { font-size: 13px; font-style: italic; fill: var(--fg); }
.box { stroke: var(--boxline); stroke-width: 1; }
.box.h { fill: var(--box-h); } .box.t { fill: var(--box-t); } .box.d { fill: var(--box-d); }
.box.o { fill: var(--box-t); stroke: var(--line); stroke-width: 1.4; stroke-dasharray: 5 3; }
.t { font-size: 14px; font-weight: 700; fill: var(--fg); }
.s { font-size: 13px; fill: var(--muted); }
.p { font-size: 13px; font-weight: 700; fill: var(--muted); }
.lbl { font-size: 12px; font-weight: 600; fill: var(--muted); }
.ln { fill: none; stroke: var(--line); stroke-width: 1.4; }
.ln.opt { stroke-dasharray: 4 3; }
.ln.ac { stroke: var(--accent); stroke-width: 2.2; }
.ln.dev { stroke: var(--dev); stroke-width: 2; stroke-dasharray: 8 3 2 3; }
.ah { fill: var(--line); } .ahA { fill: var(--accent); } .ahD { fill: var(--dev); }
""".replace("SANS", SANS)


def marker(mid: str) -> str:
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
        f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="{mid}" d="M0,1 L10,5 L0,9 z"/></marker>'
    )


DEFS = (
    "<defs>"
    + marker("ah")
    + marker("ahA")
    + marker("ahD")
    + '<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<rect class="hbg" width="7" height="7"/><line class="hln" x1="0" y1="0" x2="0" y2="7"/></pattern>'
    "</defs>"
)

LABEL = (
    "The designed evals architecture, simplified. The host app brings its own UI, storage, tracing and model "
    "clients through small adapters, any of which but the UI may be built on an optional 3tears module instead, "
    "and its code under test through a kind. The evals engine depends on no other 3tears package. Its report "
    "writer loops: the analyst writes, code checks bounce a faulty draft, the editor reviews, the analyst "
    "revises, and the report is stored. Offline, a capability gap the analyst reports becomes a change to "
    "3tears that ships back to the host."
)


def box_svg(b: tuple) -> str:
    x, y, w, h, title, lines, fill = b
    out = [f'<rect class="box {fill}" x="{x}" y="{y}" width="{w}" height="{h}" rx="3"/>']
    if title == "Engine":
        out += [f'<text class="{c}" x="{x + 10}" y="{ty}">{E(t)}</text>' for t, c, ty in ENGINE_TEXT]
        return "".join(out)
    ty = y + 22
    out.append(f'<text class="t" x="{x + 10}" y="{ty}">{E(title)}</text>')
    out += [f'<text class="s" x="{x + 10}" y="{ty + 18 + 16 * i}">{E(t)}</text>' for i, t in enumerate(lines)]
    return "".join(out)


def arrow_svg(a: tuple) -> str:
    d, style, both = a
    m = {"ac": "ahA", "dev": "ahD"}.get(style, "ah")
    start = f' marker-start="url(#{m})"' if both else ""
    return f'<path class="ln {style}" d="{d}" marker-end="url(#{m})"{start}/>'


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
    for zc, x, y, w, h, label, lx, ly in ZONES:
        p.append(f'<rect class="{zc}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>')
        p.append(f'<text class="zl" x="{lx}" y="{ly}">{E(label)}</text>')
    p.append('<rect class="subzone" x="552" y="88" width="254" height="336" rx="5"/>')
    p += [f'<text class="{c}" x="{x}" y="{y}">{E(t)}</text>' for t, x, y, c in NOTES]
    p += [arrow_svg(a) for a in ARROWS]
    p += [box_svg(b) for b in BOXES]
    for t, x, y, c, anchor, r in LABELS:
        rot = f' transform="rotate({r} {x} {y})"' if r else ""
        p.append(f'<text class="{c}" x="{x}" y="{y}" text-anchor="{anchor}"{rot}>{E(t)}</text>')
    p.append("</svg>\n")
    return "".join(p)


if __name__ == "__main__":
    for name, tokens in (("light", LIGHT), ("dark", DARK)):
        out = HERE / f"evals-overview-{name}.svg"
        out.write_text(svg(tokens), encoding="utf-8")
        print(out)
