"""Parse the CSV lines printed by the three ESP32-S3 benchmark binaries
and emit a LaTeX macro file (or patch the paper's .tex source in place)
that fills the placeholders in the paper.

Workflow:
    1. Capture the serial output of each ESP32-S3 run into a text file
       (e.g. via `idf.py monitor` followed by Ctrl+] to exit, or by
       redirecting to a file). One file per example.
    2. Run this script with the three captured files as arguments:
           python 04_collect_results.py \\
               bicycle_4_32_32_1/serial.log \\
               vdp_2_64_64_1/serial.log \\
               pendulum_2_32_32_1/serial.log
       It tolerates extra log lines and only picks the
       DUAL_<EX>,<min>,<med>,<max>,<bytes> /
       RVAD_<EX>,<min>,<med>,<max>,<bytes> CSV rows.
    3. The script writes paper_measurements.tex into this directory.
       Either copy the contents over the placeholder macros at the top of
       the paper's .tex source, or pass --patch-tex <path/to/paper.tex>
       to have the script update the macros in place (a backup of the
       original is written next to it as <paper>.tex.bak).

For convenience this script will look at default filenames if no
arguments are given:
    bicycle_4_32_32_1/serial.log
    vdp_2_64_64_1/serial.log
    pendulum_2_32_32_1/serial.log
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXAMPLES = [
    ("BICYCLE", "bicycle_4_32_32_1"),
    ("VDP",     "vdp_2_64_64_1"),
    ("PEND",    "pendulum_2_32_32_1"),
]

# Macro names in the paper's .tex (single source of truth)
MACRO_KEYS = {
    "BICYCLE": dict(
        dual_us="bicycleDualUs",
        ad_us="bicycleAdUs",
        ad_bytes="bicycleAdBytes",
    ),
    "VDP": dict(
        dual_us="vdpDualUs",
        ad_us="vdpAdUs",
        ad_bytes="vdpAdBytes",
    ),
    "PEND": dict(
        dual_us="pendDualUs",
        ad_us="pendAdUs",
        ad_bytes=None,  # not used in paper for pendulum, optional
    ),
}


@dataclass
class Row:
    method: str   # "DUAL" or "RVAD"
    example: str  # "BICYCLE", "VDP", "PEND"
    min_us: float
    med_us: float
    max_us: float
    dyn_bytes: int


_CSV_RE = re.compile(
    r"^(DUAL|RVAD)_(BICYCLE|VDP|PEND),"
    r"([-+]?\d*\.?\d+),"
    r"([-+]?\d*\.?\d+),"
    r"([-+]?\d*\.?\d+),"
    r"(\d+)\s*$"
)


def parse_log(path: Path) -> list[Row]:
    rows: list[Row] = []
    if not path.exists():
        print(f"  warning: {path} not found, skipping")
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = _CSV_RE.match(line)
        if m:
            rows.append(Row(
                method=m.group(1),
                example=m.group(2),
                min_us=float(m.group(3)),
                med_us=float(m.group(4)),
                max_us=float(m.group(5)),
                dyn_bytes=int(m.group(6)),
            ))
    return rows


def patch_tex_in_place(tex_path: Path, by_key: dict, ex_keys) -> None:
    """Update the \\newcommand{\\<macro>}{...} lines at the top of the
    paper's .tex source with the measured values. Writes a .bak alongside.

    The pattern we match is:
        \\newcommand{\\<macro>}{<anything>}
    constrained to the macros we know about; this is robust to manual edits
    elsewhere in the file.
    """
    text = tex_path.read_text()
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    backup.write_text(text)

    macro_values: dict[str, str] = {}
    for ex_key in ex_keys:
        keys = MACRO_KEYS[ex_key]
        dual = by_key.get(("DUAL", ex_key))
        rvad = by_key.get(("RVAD", ex_key))
        if dual is None or rvad is None:
            continue
        macro_values[keys["dual_us"]] = f"{dual.med_us:.2f}"
        macro_values[keys["ad_us"]] = f"{rvad.med_us:.2f}"
        if keys["ad_bytes"] is not None:
            macro_values[keys["ad_bytes"]] = str(rvad.dyn_bytes)

    n_replaced = 0
    for macro, value in macro_values.items():
        # Match \newcommand{\macro}{...} where the value MAY contain nested
        # braces (e.g. {\textit{XX.X}}). We do brace-counting with a
        # forward scan, anchored on the prefix \newcommand{\<macro>}{.
        # Note: in a regex pattern, "\n" is newline; we need "\\n" (two
        # backslashes in the regex, written as r"\\n" in Python source) to
        # mean a literal backslash followed by 'n'.
        prefix = r"\\newcommand\{\\" + re.escape(macro) + r"\}\{"
        m = re.search(prefix, text)
        if m is None:
            print(f"  warning: macro \\{macro} not found in {tex_path.name}")
            continue
        start = m.end()  # points just past the opening "{"
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth != 0:
            print(f"  warning: unbalanced braces around \\{macro}; skipping")
            continue
        end = i  # points just past the matching closing "}"
        text = text[: m.end()] + value + text[end - 1 :]
        n_replaced += 1

    tex_path.write_text(text)
    print(f"\nPatched {n_replaced} macros in {tex_path}.")
    print(f"Backup written to {backup}.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parse ESP32 benchmark logs and fill paper placeholders."
    )
    p.add_argument(
        "logs",
        nargs="*",
        help="Serial log files. If omitted, uses <example>/serial.log for each.",
    )
    p.add_argument(
        "--patch-tex",
        type=Path,
        default=None,
        help="Path to the paper's .tex source. If given, the macros are "
             "patched in place (a .bak file is written next to the source).",
    )
    args = p.parse_args()

    if args.logs:
        log_paths = [Path(s) for s in args.logs]
    else:
        log_paths = [
            ROOT / "bicycle_4_32_32_1" / "serial.log",
            ROOT / "vdp_2_64_64_1" / "serial.log",
            ROOT / "pendulum_2_32_32_1" / "serial.log",
        ]

    all_rows: list[Row] = []
    for log_path in log_paths:
        print(f"Parsing {log_path} ...")
        rs = parse_log(log_path)
        for r in rs:
            print(f"  {r}")
        all_rows.extend(rs)

    by_key = {(r.method, r.example): r for r in all_rows}

    out_lines = [
        "% Auto-generated by 04_collect_results.py.",
        "% Replace the placeholder macros at the top of the paper's .tex",
        "% source with the lines below (or rerun with --patch-tex).",
        "",
    ]
    ex_keys = [k for k, _ in EXAMPLES]
    for ex_key in ex_keys:
        keys = MACRO_KEYS[ex_key]
        dual = by_key.get(("DUAL", ex_key))
        rvad = by_key.get(("RVAD", ex_key))
        if dual is None:
            print(f"  WARNING: no DUAL_{ex_key} row found")
            continue
        if rvad is None:
            print(f"  WARNING: no RVAD_{ex_key} row found")
            continue
        out_lines.append(f"% --- {ex_key} ---")
        out_lines.append(
            f"\\renewcommand{{\\{keys['dual_us']}}}{{{dual.med_us:.2f}}}"
        )
        out_lines.append(
            f"\\renewcommand{{\\{keys['ad_us']}}}{{{rvad.med_us:.2f}}}"
        )
        if keys["ad_bytes"] is not None:
            out_lines.append(
                f"\\renewcommand{{\\{keys['ad_bytes']}}}{{{rvad.dyn_bytes}}}"
            )
        out_lines.append("")

    out_path = ROOT / "paper_measurements.tex"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {out_path}")

    if args.patch_tex is not None:
        if not args.patch_tex.exists():
            print(f"  ERROR: --patch-tex {args.patch_tex} does not exist.")
            sys.exit(2)
        patch_tex_in_place(args.patch_tex, by_key, ex_keys)
    else:
        print("\nCopy the \\renewcommand block over the corresponding")
        print("\\newcommand block at the top of the paper's .tex source,")
        print("or re-run with --patch-tex <path/to/paper.tex>.")


if __name__ == "__main__":
    main()
