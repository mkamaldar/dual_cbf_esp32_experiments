#!/usr/bin/env python3
"""Per-method code size and static storage, read from the linked ELF file.

    python tools/size_report.py <example>/build/<project>.elf

Uses xtensa-esp32s3-elf-objdump when it is on PATH (after export.sh),
otherwise objdump. Symbols are grouped by the namespace that owns them and
classified by the section that holds them:
  code    .flash.text / .iram0.text / .text   (flash or IRAM)
  ram     .dram0.bss / .dram0.data / .bss / .data   (mutable buffers: the workspace)
  flash   .flash.rodata / .rodata   (read-only data: the weights)
The output lines start with SIZE, so that they can be appended to serial.log.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from collections import defaultdict

GROUPS = ["bench::m_", "dual_cbf::", "rev_static::", "jet3::",
          "rev2_static::", "reverse_mode_ad::", "ref64::"]


def classify(section: str) -> str | None:
    if "text" in section:
        return "code"
    if "rodata" in section:
        return "flash"
    if "bss" in section or "data" in section:
        return "ram"
    return None


def demangle(name: str) -> str:
    tool = "xtensa-esp32s3-elf-c++filt" if shutil.which("xtensa-esp32s3-elf-c++filt") else "c++filt"
    try:
        return subprocess.run([tool, name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--objdump", default=None)
    args = ap.parse_args()
    tool = args.objdump or ("xtensa-esp32s3-elf-objdump"
                            if shutil.which("xtensa-esp32s3-elf-objdump") else "objdump")
    out = subprocess.run([tool, "-t", "-C", args.elf],
                         capture_output=True, text=True, check=True).stdout
    total = defaultdict(int)
    rows = []
    for line in out.splitlines():
        # address, flags (7 columns), section, size, name
        m = re.match(r"^[0-9a-fA-F]+\s.{7}\s(\S+)\s+([0-9a-fA-F]+)\s+(.*)$", line)
        if not m:
            continue
        section, size, name = m.group(1), int(m.group(2), 16), m.group(3).strip()
        if name.startswith("_Z") and "$" in name:
            # Compiler-made clone (e.g. "...$isra$0"): the suffix prevents
            # demangling, so demangle the name without it.
            name = demangle(name.split("$", 1)[0]) + " [clone]"
        kind = classify(section)
        if kind is None or size == 0:
            continue
        for g in GROUPS:
            if name.startswith(g):
                rows.append((g, kind, size, name))
                total[(g, kind)] += size
                break
    for g, kind, size, name in sorted(rows):
        print(f"SIZE,{kind},{size},{name}")
    for (g, kind), size in sorted(total.items()):
        print(f"SIZETOTAL,{g},{kind},{size}")


if __name__ == "__main__":
    main()
