#!/usr/bin/env python3
"""Build, flash, run, and record the six benchmark projects with one command.

Run it from the experiments folder, in a terminal where ESP-IDF is active
(on Windows: the "ESP-IDF PowerShell" or "ESP-IDF CMD" shortcut):

    python tools/run_experiments.py                  all six projects
    python tools/run_experiments.py vdp_2_64_64_1    one project
    python tools/run_experiments.py --port COM5      name the serial port

For each project the script
  1. deletes build/ and sdkconfig, sets the target, and builds;
  2. flashes the board;
  3. resets the board and records its output, from the BEGIN line to the END
     line, in <project>/serial.log (UTF-8);
  4. appends the code-size report of the ELF file to the same log.
At the end it writes logs.zip with the six logs.

A project whose serial.log is already complete is skipped; --force repeats it.
Other options: --skip-build (reuse build/), --capture-only (no build, no
flash: reset the board and record), --list-ports.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ["bicycle_4_32_32_1", "vdp_2_64_64_1", "pendulum_2_32_32_1",
            "depth_2x32", "depth_4x32", "depth_8x32"]
MARK = re.compile(r"^----- (BEGIN|END) (\w+) -----")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RUN_TIMEOUT_S = 60 * 60       # one project, from BEGIN to END
UART_VIDS = {0x10C4: "CP210x", 0x1A86: "CH34x", 0x0403: "FTDI"}
NATIVE_USB_VID = 0x303A


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str) -> "None":
    say()
    say("STOPPED: " + msg)
    sys.exit(1)


# --------------------------------------------------------------- environment
def check_environment(need_idf: bool) -> None:
    if need_idf and not os.environ.get("IDF_PATH"):
        fail("ESP-IDF is not active in this window.\n"
             "  Windows: close this window, open the Start menu, type ESP-IDF, and start\n"
             "           'ESP-IDF 5.x PowerShell' (or 'ESP-IDF 5.x CMD'). Run the command there.\n"
             "  Linux/macOS: run  . ~/esp/esp-idf/export.sh  first.")
    try:
        import serial  # noqa: F401
    except ImportError:
        fail("The Python module 'serial' (pyserial) is missing. It is part of the ESP-IDF\n"
             "  Python environment, so this window is probably not an ESP-IDF window.")


def idf(args: list[str], cwd: Path, log: Path) -> int:
    """Run idf.py with the Python of this window; show and save its output."""
    cmd = [sys.executable, str(Path(os.environ["IDF_PATH"]) / "tools" / "idf.py")] + args
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ idf.py " + " ".join(args) + "\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        shown = time.time()
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            f.write(text)
            # Show progress lines and errors, not the whole build output.
            s = text.strip()
            if re.match(r"^\[\d+/\d+\]", s):
                if time.time() - shown > 5.0:
                    say("    " + s[:100])
                    shown = time.time()
            elif re.search(r"error|Error|FAILED|Project build complete|Hard resetting|"
                           r"Connecting|Writing at|Serial port|fatal", s):
                say("    " + s[:160])
        return proc.wait()


# ---------------------------------------------------------------------- port
def list_ports() -> list:
    from serial.tools import list_ports as lp
    return sorted(lp.comports(), key=lambda p: p.device)


def describe(p) -> str:
    kind = UART_VIDS.get(p.vid, "native USB" if p.vid == NATIVE_USB_VID else "")
    return f"{p.device}  {p.description}" + (f"  [{kind}]" if kind else "")


def find_port(requested: str | None) -> str:
    if requested:
        return requested
    ports = list_ports()
    uart = [p for p in ports if p.vid in UART_VIDS]
    native = [p for p in ports if p.vid == NATIVE_USB_VID]
    if len(uart) == 1:
        say(f"Serial port: {describe(uart[0])}")
        return uart[0].device
    if not uart and native:
        fail("The board is connected through its native USB port (" + native[0].device + ").\n"
             "  The benchmark prints on the UART. Move the cable to the USB connector\n"
             "  labeled UART (or COM) on the board and run the command again.")
    if not ports:
        fail("No serial port found. Check that the board is plugged in with a data cable\n"
             "  (not a charge-only cable). On Windows, the board should appear in Device\n"
             "  Manager under 'Ports (COM & LPT)'; if it appears under 'Other devices',\n"
             "  install the CP210x driver from Silicon Labs.")
    say("More than one serial port (or none recognized as a UART bridge):")
    for p in ports:
        say("    " + describe(p))
    fail("Name the port of the board, for example:\n"
         "    python tools/run_experiments.py --port " + ports[0].device)
    return ""


# ------------------------------------------------------------------- capture
def reset_board(ser) -> bool:
    """Pulse EN through RTS, as esptool does. Returns False if the port has no modem lines."""
    try:
        ser.dtr = False          # IO0 high: normal boot
        ser.rts = True           # EN low: chip in reset
        time.sleep(0.2)
        ser.rts = False          # EN high: chip runs
        return True
    except (OSError, ValueError):
        return False


def capture(port: str, out: Path, baud: int = 115200) -> tuple[str, str]:
    """Record one run (BEGIN line to END line). Returns (example name, validation result)."""
    import serial
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.25
    try:
        ser.dtr = False
        ser.rts = False
    except (OSError, ValueError):
        pass
    try:
        ser.open()
    except serial.SerialException as e:
        fail(f"Cannot open {port}: {e}\n"
             "  Close other programs that use the port (another monitor window, Arduino IDE,\n"
             "  PuTTY) and run the command again.")
    reset_board(ser)

    lines: list[str] = []
    recording = False
    example = ""
    result = ""
    buf = b""
    t_open = time.time()
    t_begin = 0.0
    t_last_line = time.time()
    t_note = time.time()
    asked_button = False
    restarts = 0
    try:
        while True:
            chunk = ser.read(4096)
            now = time.time()
            if chunk:
                buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = ANSI.sub("", raw.decode("utf-8", errors="replace")).rstrip("\r")
                t_last_line = now
                m = MARK.match(line)
                if m and m.group(1) == "BEGIN":
                    if recording:
                        restarts += 1
                        if restarts > 2:
                            out.with_suffix(".partial.log").write_text(
                                "\n".join(lines) + "\n", encoding="utf-8")
                            fail("The board restarts during the run. The output before the last "
                                 f"restart is in {out.with_suffix('.partial.log')}; see that file.")
                        say("    (the board restarted; recording starts again)")
                    lines, recording, example, result = [], True, m.group(2), ""
                    t_begin = now
                if not recording:
                    continue
                lines.append(line)
                if line.startswith("INFO,validation,"):
                    result = line.split(",")[2].strip()
                if line.startswith(("INFO,status", "INFO,validation", "-----")) or \
                        "Guru Meditation" in line or "abort()" in line:
                    say("    " + line)
                if m and m.group(1) == "END":
                    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    return example, result
            if not recording and not asked_button and now - t_open > 10.0:
                say("    No output yet. Press the RST (or EN) button on the board once.")
                asked_button = True
            if not recording and now - t_open > 180.0:
                fail(f"No BEGIN line from the board on {port} within 3 minutes.\n"
                     "  Check that the cable is in the connector labeled UART, then run again.")
            if recording and now - t_note > 30.0:
                quiet = int(now - t_last_line)
                say(f"    running, {int(now - t_begin)} s"
                    + (f" (no new line for {quiet} s; the reference computation is slow)"
                       if quiet > 20 else f" ({len(lines)} lines)"))
                t_note = now
            if recording and now - t_begin > RUN_TIMEOUT_S:
                out.with_suffix(".partial.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
                fail(f"No END line after {RUN_TIMEOUT_S // 60} minutes. The partial output is in "
                     f"{out.with_suffix('.partial.log')}; see that file.")
    finally:
        ser.close()


# ------------------------------------------------------------------- project
def elf_of(proj: Path) -> Path:
    text = (proj / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*project\(\s*([\w\-]+)\s*\)", text, flags=re.M)
    if not m:
        fail(f"No project(...) line in {proj / 'CMakeLists.txt'}")
    return proj / "build" / (m.group(1) + ".elf")


def log_state(log: Path) -> tuple[bool, str]:
    """(complete, validation) of an existing serial.log."""
    if not log.exists():
        return False, ""
    text = log.read_text(encoding="utf-8", errors="replace")
    val = re.search(r"^INFO,validation,(\w+)", text, flags=re.M)
    complete = bool(re.search(r"^----- END \w+ -----", text, flags=re.M)) and "\nSIZE," in text
    return complete, (val.group(1) if val else "")


def append_size(proj: Path, log: Path) -> int:
    elf = elf_of(proj)
    if not elf.exists():
        fail(f"{elf} does not exist; the size report needs a build. Run without --capture-only.")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "size_report.py"), str(elf)],
                       capture_output=True, text=True)
    rows = [ln for ln in r.stdout.splitlines() if ln.startswith("SIZE")]
    if r.returncode != 0 or not rows:
        fail("The size report failed:\n" + (r.stderr or r.stdout)[-800:])
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return len(rows)


def run_project(name: str, args, port_box: list) -> tuple[str, str, float]:
    proj = ROOT / name
    log = proj / "serial.log"
    build_log = proj / "build_output.txt"
    t0 = time.time()
    complete, val = log_state(log)
    if complete and val == "PASS" and not args.force:
        say(f"[{name}] serial.log is complete (validation PASS); skipped. Use --force to repeat.")
        return "skipped", val, 0.0

    if not args.capture_only:
        if not args.skip_build:
            say(f"[{name}] 1/4 build (a few minutes; full output in {build_log.name})")
            build_log.unlink(missing_ok=True)
            shutil.rmtree(proj / "build", ignore_errors=True)
            (proj / "sdkconfig").unlink(missing_ok=True)
            if idf(["set-target", "esp32s3"], proj, build_log) != 0:
                fail(f"'idf.py set-target esp32s3' failed in {name}. See {build_log}.")
            if idf(["build"], proj, build_log) != 0:
                fail(f"The build of {name} failed. See {build_log}.")
        if not port_box:
            port_box.append(find_port(args.port))
        say(f"[{name}] 2/4 flash on {port_box[0]}")
        if idf(["-p", port_box[0], "flash"], proj, build_log) != 0:
            fail(f"Flashing {name} failed. If the output above shows 'Connecting....' without\n"
                 "  success: hold the BOOT button, press and release RST, release BOOT, and run\n"
                 f"  the command again. Otherwise see {build_log}.")
    if not port_box:
        port_box.append(find_port(args.port))

    say(f"[{name}] 3/4 run and record (several minutes; a long pause after 'validating' is normal)")
    example, val = capture(port_box[0], log)
    say(f"[{name}] 4/4 code-size report")
    n = append_size(proj, log)
    dt = time.time() - t0
    say(f"[{name}] done: example {example}, validation {val or 'not reported'}, "
        f"{n} size lines, {dt / 60:.1f} min")
    return "ran", val, dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="*", help="project folders (default: all six)")
    ap.add_argument("--port", help="serial port of the board, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--force", action="store_true", help="repeat projects whose log is complete")
    ap.add_argument("--skip-build", action="store_true", help="flash the existing build")
    ap.add_argument("--capture-only", action="store_true", help="no build, no flash")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    check_environment(need_idf=not (args.capture_only or args.list_ports))
    if args.list_ports:
        ports = list_ports()
        for p in ports:
            say(describe(p))
        if not ports:
            say("No serial port found.")
        return

    names = [n.strip("/\\") for n in args.projects] or PROJECTS
    for n in names:
        if not (ROOT / n / "CMakeLists.txt").exists():
            fail(f"'{n}' is not a project folder. The projects are: " + ", ".join(PROJECTS))
    if " " in str(ROOT):
        say(f"WARNING: the folder path contains a space ({ROOT}).\n"
            "         ESP-IDF builds can fail in such a path. If the build fails, move the\n"
            "         folder to a short path without spaces, e.g. C:\\esp\\dual_cbf_esp32_experiments.")

    say(f"Experiments folder: {ROOT}")
    say(f"Projects: {', '.join(names)}")
    say()
    port_box: list = []
    results = {}
    for n in names:
        results[n] = run_project(n, args, port_box)
        say()

    say("Summary")
    bad = []
    for n in PROJECTS:
        complete, val = log_state(ROOT / n / "serial.log")
        status = "complete" if complete else "missing"
        say(f"  {n:22s} log {status:9s} validation {val or '-'}")
        if not complete or val != "PASS":
            bad.append(n)
    logs = [ROOT / n / "serial.log" for n in PROJECTS if (ROOT / n / "serial.log").exists()]
    if logs:
        z = ROOT / "logs.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in logs:
                zf.write(p, arcname=f"{p.parent.name}/serial.log")
        say()
        say(f"Wrote {z} with {len(logs)} log(s).")
    if bad:
        say("Not finished or not PASS: " + ", ".join(bad))
        if any(log_state(ROOT / n / "serial.log")[1] == "FAIL" for n in bad):
            say("A validation FAIL is not expected. The logs are in logs.zip.")
        sys.exit(2)
    say("ALL SIX PROJECTS COMPLETE, VALIDATION PASS. The logs are in logs.zip.")


if __name__ == "__main__":
    main()
