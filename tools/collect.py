#!/usr/bin/env python3
"""Turn the serial logs into the measurement block of the paper.

    python tools/collect.py */serial.log --sim sim_results.txt --tex paper.tex

Each .tex file contains a block

    % BEGIN MEASUREMENTS
    ...
    % END MEASUREMENTS

which is replaced by lines \\Mdef{key}{value}. In the text, \\M{key} prints
the value (a magenta question mark while the key is undefined). Without
--tex, the block is printed.

Keys
  t.<EX>.<method>.<ASM|STEP>.<EN|MASK>.<min|med|p99|max|mean>   time (us)
  m.<EX>.<method>.<static|stack|heap|allocs|work|code>          memory (bytes)
  e.<EX>.<method>   max scaled error, states away from ReLU boundaries
  ek.<EX>.<method>  max scaled error, states near a ReLU boundary
  k.<EX>.near, k.<EX>.n, f.<EX>.<method>                        near-boundary counts, gate flips
  r.<EX>.<method>   median assembly time divided by that of the static reverse-mode code
  p.<EX>.<method>.<250|1000>   maximum step time as a percentage of a 250 Hz / 1 kHz period
  i.<key>, i.<EX>.<key>   provenance (INFO lines);  sim.<key>   closed-loop simulations
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

CODE_OWNERS = {
    "dual_seq": [r"^bench::m_dual_seq\(", r"^dual_cbf::forward_directional"],
    "dual_shared": [r"^bench::m_dual_shared\(", r"^dual_cbf::evaluate_cbf_shared"],
    "rev_static": [r"^bench::m_rev_static\(", r"^rev_static::"],
    "rev_heap": [r"^bench::m_rev_heap\(", r"^reverse_mode_ad::"],
    "hyperdual": [r"^bench::m_hyperdual\(", r"^dual_cbf::forward_hyperdual", r"^dual_cbf::evaluate_"],
    "jet3": [r"^bench::m_jet3\(", r"^jet3::"],
    "rev2_static": [r"^bench::m_rev2_static\(", r"^rev2_static::"],
    "rev_fd": [r"^bench::m_rev_fd\(", r"^reverse_mode_ad::"],
}
BASELINE = {1: "rev_static", 2: "rev2_static"}


def sci(x: float) -> str:
    if x == 0.0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


RAW: dict = {}      # unrounded times, keyed like K


def q(x, places: int) -> str:
    """Decimal rounding, half up, of a number or of its decimal string."""
    step = Decimal(1).scaleb(-places)
    return str(Decimal(str(x)).quantize(step, rounding=ROUND_HALF_UP))


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def read_log(path: Path) -> str:
    """Text of a log. Logs redirected in Windows PowerShell are UTF-16."""
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    if data.count(b"\x00") > len(data) // 4:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def parse_log(path: Path, K: dict) -> None:
    ex = None
    err = defaultdict(float)
    errk = defaultdict(float)
    sizes = []
    methods = []
    for raw in read_log(path).splitlines():
        line = ANSI.sub("", raw).strip()
        p = line.split(",")
        tag = p[0]
        if tag == "INFO" and len(p) >= 3:
            key, val = p[1], ",".join(p[2:])
            if key == "example":
                ex = val
            if key == "app_version" and not re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?|v?\d+\.\d+.*", val):
                continue      # built outside a git repository: no commit to report
            if ex:
                K[f"i.{ex}.{key}"] = val
            K.setdefault(f"i.{key}", val)
        elif tag == "TIME" and len(p) == 11:
            _, e, m, win, irq, n, mn, med, p99, mx, mean = p
            if m not in methods:
                methods.append(m)
            for name, v in zip(("min", "med", "p99", "max", "mean"), (mn, med, p99, mx, mean)):
                RAW[f"t.{e}.{m}.{win}.{irq}.{name}"] = Decimal(v)
                K[f"t.{e}.{m}.{win}.{irq}.{name}"] = q(v, 2)
        elif tag == "COLD" and len(p) == 4:
            K[f"t.{p[1]}.{p[2]}.cold"] = q(p[3], 2)
        elif tag == "MEM" and len(p) == 7:
            _, e, m, st, sk, hp, al = p
            K[f"m.{e}.{m}.static"], K[f"m.{e}.{m}.stack"] = st, sk
            K[f"m.{e}.{m}.heap"], K[f"m.{e}.{m}.allocs"] = hp, al
            if hp != "NA":
                K[f"m.{e}.{m}.work"] = str(int(st) + int(sk) + int(hp))
        elif tag in ("ERR", "ERRK") and len(p) == 6:
            d = err if tag == "ERR" else errk
            d[(p[1], p[2])] = max(d[(p[1], p[2])], float(p[5]))
        elif tag == "KINK" and len(p) == 5:
            K[f"k.{p[1]}.tol"], K[f"k.{p[1]}.near"], K[f"k.{p[1]}.n"] = sci(float(p[2])), p[3], p[4]
        elif tag == "FLIP" and len(p) == 4:
            K[f"f.{p[1]}.{p[2]}"] = p[3]
        elif tag == "SIZE" and len(p) >= 4:
            sizes.append((p[1], int(p[2]), ",".join(p[3:])))
    for (e, m), v in err.items():
        K[f"e.{e}.{m}"] = sci(v)
    for (e, m), v in errk.items():
        K[f"ek.{e}.{m}"] = sci(v)
    if ex is None:
        return
    for m in methods:
        pats = CODE_OWNERS.get(m, [])
        code = sum(sz for kind, sz, name in sizes
                   if kind == "code" and any(re.search(pt, name) for pt in pats))
        if code:
            K[f"m.{ex}.{m}.code"] = str(code)
    rd = int(K.get(f"i.{ex}.relative_degree", "1"))
    base = RAW.get(f"t.{ex}.{BASELINE[rd]}.ASM.EN.med")
    for m in methods:
        med = RAW.get(f"t.{ex}.{m}.ASM.EN.med")
        if base and med:
            K[f"r.{ex}.{m}"] = q(med / base, 2)
        smax = RAW.get(f"t.{ex}.{m}.STEP.EN.max")
        if smax:
            K[f"p.{ex}.{m}.250"] = q(100 * smax / 4000, 1)
            K[f"p.{ex}.{m}.1000"] = q(100 * smax / 1000, 1)


def aggregates(K: dict) -> None:
    """Ranges quoted in the abstract and the summary."""
    pairs = [("BICYCLE", "dual_seq"), ("VDP", "dual_seq"), ("PENDULUM", "hyperdual")]
    r = [float(K[f"r.{e}.{m}"]) for e, m in pairs if f"r.{e}.{m}" in K]
    if len(r) == len(pairs):
        K["g.ratio_min"], K["g.ratio_max"] = f"{min(r):.1f}", f"{max(r):.1f}"
    pairs_sh = [("BICYCLE", "dual_shared"), ("VDP", "dual_shared")]
    r = [float(K[f"r.{e}.{m}"]) for e, m in pairs_sh if f"r.{e}.{m}" in K]
    if len(r) == len(pairs_sh):
        K["g.ratio_shared_min"], K["g.ratio_shared_max"] = f"{min(r):.1f}", f"{max(r):.1f}"
    s = [RAW[f"t.{e}.{m}.STEP.EN.max"] for e, m in pairs if f"t.{e}.{m}.STEP.EN.max" in RAW]
    if len(s) == len(pairs):
        K["g.step_max"] = q(max(s), 0)
        K["g.step_pct250"] = q(100 * max(s) / 4000, 1)
        K["g.step_pct1000"] = q(100 * max(s) / 1000, 1)
    allocs = [v for k, v in K.items() if k.startswith("m.") and k.endswith(".allocs")
              and not any(h in k for h in ("rev_heap", "rev_fd"))]
    if allocs and all(a != "NA" for a in allocs):
        K["g.static_allocs"] = str(sum(int(a) for a in allocs))


def block(K: dict) -> str:
    lines = ["% BEGIN MEASUREMENTS"]
    lines += [f"\\Mdef{{{k}}}{{{K[k]}}}" for k in sorted(K)]
    lines.append("% END MEASUREMENTS")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--sim", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra key, e.g. --set i.compiler_commit=abc1234")
    ap.add_argument("--tex", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    K: dict = {}
    logs = []
    for item in args.logs:      # expand wildcards here: Windows shells do not
        hits = sorted(Path().glob(str(item).replace("\\", "/"))) if "*" in str(item) else [item]
        logs.extend(hits)
    if not logs:
        raise SystemExit("No log file found. Run from the experiments folder, e.g. "
                         "python tools/collect.py */serial.log")
    for log in logs:
        parse_log(log, K)
    aggregates(K)
    if args.sim and args.sim.exists():
        for line in args.sim.read_text().splitlines():
            p = line.strip().split(",")
            if p[0] == "SIM" and len(p) == 3:
                K[f"sim.{p[1]}"] = p[2]
    for kv in args.set:
        k, v = kv.split("=", 1)
        K[k] = v
    K = {k: v.replace("_", "\\_") if k.startswith("i.") else v for k, v in K.items()}
    text = block(K)
    if not args.tex:
        print(text)
        return
    for tex in args.tex:
        src = tex.read_text()
        new, n = re.subn(r"% BEGIN MEASUREMENTS.*?% END MEASUREMENTS", lambda _: text, src, flags=re.S)
        if n != 1:
            raise SystemExit(f"{tex}: measurement block not found")
        tex.with_suffix(tex.suffix + ".bak").write_text(src)
        tex.write_text(new)
        print(f"{tex}: wrote {len(K)} keys")


if __name__ == "__main__":
    main()
