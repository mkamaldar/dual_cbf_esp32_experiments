# ESP32-S3 experiments for dual-cbf-compiler

This repository contains the embedded experiments of the paper

> M. Kamaldar, "[Extracting exact Lie derivatives without backpropagation: A dual compiler for neural control barrier functions][paper]," 2026.

The paper presents a compiler that translates a trained neural control
barrier function (CBF) into a self-contained C++ header, which evaluates the
barrier and its Lie derivatives by forward-mode differentiation with dual
numbers, with buffers of fixed size and without dynamic memory allocation.
The compiler is a separate Python package, `dual-cbf-compiler`
([PyPI][pypi], [GitHub][compiler]).

On an ESP32-S3 microcontroller, the experiments compare the emitted code with
reverse-mode code that is likewise generated ahead of time with statically
allocated buffers. The measured quantities are the execution time, the
static, stack, and heap memory, the code size, and the numerical error
against a double-precision reference. The repository contains the training
scripts, the generated headers, the firmware, the measurement tools, the
serial logs behind the numbers of the paper, and the closed-loop simulations.

[paper]: https://drive.google.com/file/d/1ryW5-aeYk7cyXl2FccErl3kusam0zbzE/view?usp=sharing
[pypi]: https://pypi.org/project/dual-cbf-compiler/
[compiler]: https://github.com/mkamaldar/dual_cbf_compiler

## Contents

- [Experiments and methods](#experiments-and-methods)
- [Measurements of the paper](#measurements-of-the-paper)
- [What can be reproduced, and with what](#what-can-be-reproduced-and-with-what)
- [Step 0: repository and Python environment](#step-0-repository-and-python-environment)
- [Step 1: measurement block from the stored logs](#step-1-measurement-block-from-the-stored-logs)
- [Step 2: host pre-flight](#step-2-host-pre-flight)
- [Step 3: retrain and regenerate](#step-3-retrain-and-regenerate)
- [Step 4: closed-loop simulations and figure](#step-4-closed-loop-simulations-and-figure)
- [Step 5: build, flash, run, record](#step-5-build-flash-run-record)
- [Step 6: collect](#step-6-collect)
- [What is measured, and how](#what-is-measured-and-how)
- [Provenance of the reported numbers](#provenance-of-the-reported-numbers)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Status and intended use](#status-and-intended-use)
- [Citation](#citation)
- [License](#license)

## Experiments and methods

Each experiment is one ESP-IDF project:

| Project | System | Network | Activation | Inputs | Relative degree |
| --- | --- | --- | --- | --- | --- |
| `bicycle_4_32_32_1/` | kinematic bicycle, lane keeping | 4-32-32-1 | ReLU | 2 | 1 |
| `vdp_2_64_64_1/` | Van der Pol oscillator | 2-64-64-1 | ReLU | 1 | 1 |
| `pendulum_2_32_32_1/` | inverted pendulum | 2-32-32-1 | softplus | 1 | 2 |
| `depth_2x32/`, `depth_4x32/`, `depth_8x32/` | Van der Pol oscillator | 2, 4, and 8 hidden layers of width 32 | ReLU | 1 | 1 |

The networks are trained by least-squares regression onto analytic target
barriers. They are not verified to satisfy the condition of a CBF.

The firmware of each project runs the following methods on the same states:

| Name | Method |
| --- | --- |
| `dual_seq` | emitted `evaluate_cbf`: one dual forward pass per vector field |
| `dual_shared` | emitted `evaluate_cbf_shared`: the real part is propagated once, with all dual parts |
| `rev_static` | reverse mode, generated ahead of time, static buffers |
| `rev_heap` | reverse mode with a heap-allocated cache |
| `hyperdual` | emitted `evaluate_hocbf`: hyper-dual passes (relative degree two) |
| `jet3` | three-component truncated second-order algebra (relative degree two) |
| `rev2_static` | forward-over-reverse, generated ahead of time, static buffers (relative degree two) |
| `rev_fd` | finite difference of the reverse-mode gradient (relative degree two) |

All methods read the same `static const float` parameter arrays and are
compiled with the same flags, `-O2 -ffast-math -funroll-loops`.

## Measurements of the paper

The folder [`paper_logs/`](paper_logs) contains the six serial logs from
which the numbers of the paper are computed. The following excerpt gives the
median time of the constraint assembly with interrupts enabled, over 1,000
seeded states, and the memory of each method.

| Experiment | Method | Median assembly time (µs) | Static (B) | Stack (B) | Heap (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| Bicycle | `dual_seq` | 97.00 | 256 | 296 | 0 |
| | `dual_shared` | 61.61 | 512 | 600 | 0 |
| | `rev_static` | 47.89 | 512 | 24 | 0 |
| | `rev_heap` | 143.56 | 0 | 328 | 816 |
| Van der Pol | `dual_seq` | 237.29 | 512 | 552 | 0 |
| | `dual_shared` | 181.41 | 768 | 824 | 0 |
| | `rev_static` | 169.73 | 1024 | 152 | 0 |
| | `rev_heap` | 281.98 | 0 | 328 | 1576 |
| Pendulum | `hyperdual` | 510.78 | 512 | 696 | 0 |
| | `jet3` | 739.40 | 384 | 584 | 0 |
| | `rev2_static` | 272.13 | 1280 | 440 | 0 |
| | `rev_fd` | 986.51 | 0 | 408 | 800 |

[Step 1](#step-1-measurement-block-from-the-stored-logs) prints all
measurements, including the maxima, the step times, the code sizes, the
numerical errors, and the depth sweep.

## What can be reproduced, and with what

| Step | Result | Needs |
| --- | --- | --- |
| 1 | All measurement values of the paper, from the stored logs | Python |
| 2 | Agreement of all methods with the double-precision reference on the host | `g++`, `make` |
| 3 | The trained networks and the generated headers | Python, PyTorch |
| 4 | The closed-loop simulations, the grid check, and the figure | Python, `g++`; MATLAB for the figure of the paper |
| 5 | New serial logs on hardware | ESP32-S3 board, ESP-IDF v5.5.4 |
| 6 | Measurement values from new logs | Python |

All generated files are stored in the repository, so that each step can be
run without the preceding ones. Steps 1, 3, 5, and 6 run on Windows, Linux,
and macOS. Steps 2 and 4 call `g++` and were run on Linux; on Windows, they
run under WSL.

## Step 0: repository and Python environment

Windows (PowerShell):

```powershell
git clone https://github.com/mkamaldar/dual_cbf_esp32_experiments.git
cd dual_cbf_esp32_experiments
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` in that window and repeat
the line.

Linux and macOS:

```bash
git clone https://github.com/mkamaldar/dual_cbf_esp32_experiments.git
cd dual_cbf_esp32_experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs `dual-cbf-compiler` version 0.2.0 or later,
PyTorch, ONNX, NumPy, SciPy, and Matplotlib. Step 1 alone needs none of
these packages.

## Step 1: measurement block from the stored logs

```bash
python tools/collect.py paper_logs/*.log --sim sim_results.txt
```

The command prints one line `\Mdef{key}{value}` per measurement, which is the
block that the LaTeX source of the paper reads. The script expands the
wildcard itself, so that the same command runs in PowerShell. The keys are
described at the top of [`tools/collect.py`](tools/collect.py); for example,
`t.BICYCLE.dual_shared.ASM.EN.med` is the median assembly time of the shared
sweep for the bicycle with interrupts enabled.

## Step 2: host pre-flight

```bash
make host-test
```

For each project, the benchmark driver is built for the host, and all methods
are compared with the double-precision reference at 100,000 seeded states.
The expected output is `INFO,validation,PASS` for each of the six projects.
The lines `KINK` and `FLIP` count the states within `1e-4` of a ReLU
activation boundary and the gate flips at those states; they do not fail the
test.

```bash
make check-ref
```

compares the double-precision reference itself with `torch.autograd` in
double precision at 500 states per project.

## Step 3: retrain and regenerate

```bash
python 01_train_models.py       # writes <project>/cbf.pt and cbf.onnx
python 02_compile_headers.py    # writes <project>/main/dual_cbf.h and methods.h
```

`01_train_models.py` trains the six networks with the Adam optimizer, batches
of 1,024 states drawn uniformly from a box, and the random seed 0: 5,000
iterations at the learning rate `1e-3`, and, for the bicycle, 20,000
iterations, the last 6,000 at `2e-4`. `02_compile_headers.py` calls the
compiler on each ONNX file and generates the comparison methods and the
double-precision reference.

On Linux with PyTorch 2.15 (CPU), this step reproduces the stored
weights and the stored headers byte for byte. A different PyTorch version or
platform can change the trained weights, and therefore the numerical values,
within the variation of training. From the stored ONNX files,
`02_compile_headers.py` reproduces the stored headers byte for byte with
`dual-cbf-compiler` 0.2.0.

## Step 4: closed-loop simulations and figure

```bash
python 03_run_closed_loop.py
```

The script compiles each emitted header into a shared library with `g++` and
calls it from Python, so that the constraint coefficients of the simulations
come from the code that runs on the microcontroller. The plant is integrated
with a fourth-order Runge-Kutta scheme under a zero-order hold with the
period 0.02 s. For each system, an ideal case and a perturbed case, with
measurement noise and an input delay of one period, are simulated. The
script also evaluates the high-order CBF inequality of the pendulum barrier
on two grids and repeats the filtered trajectories with 10, 20, 40, and 80
Runge-Kutta substeps per period.

| Output | Content |
| --- | --- |
| `sim_results.txt` | `SIM,<key>,<value>` lines, read by `tools/collect.py` |
| `closed_loop_data.mat` | the trajectories, for `plot_closed_loop.m` |
| `closed_loop.pdf` | a Matplotlib preview of the three panels |

The stored `sim_results.txt` and `closed_loop_data.mat` are the outputs of
this script. These simulations illustrate the behavior of the filter; they
are not a verification of forward invariance.

The closed-loop figure of the paper is drawn in MATLAB from the stored
trajectories:

```matlab
plot_closed_loop
```

`plot_closed_loop.m` requires `tight_subplot.m` by Pekka Kumpulainen, which is
not redistributed with this repository; download it from the
[MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w)
and place it next to the script.

The number of timing states at which the quadratic program has no solution
and the fallback input is used is counted on the host by
[`tools/count_qp_fallback.cpp`](tools/count_qp_fallback.cpp); the build line
is at the top of that file.

## Step 5: build, flash, run, record

**Hardware.** An ESP32-S3-DevKitC-1 development board, connected through
its UART USB port.

**ESP-IDF.** The measurements of the paper were obtained with ESP-IDF v5.5.4,
which uses GCC 14.2.0.

- Windows: install ESP-IDF v5.5.4 with the Espressif installer, and open the
  "ESP-IDF PowerShell" shortcut that the installer creates. The commands
  below are typed in that window.
- Linux and macOS:

  ```bash
  mkdir -p ~/esp && cd ~/esp
  git clone -b v5.5.4 --recursive https://github.com/espressif/esp-idf.git
  cd esp-idf && ./install.sh esp32s3
  . ~/esp/esp-idf/export.sh        # in each new terminal
  ```

**Run.** From the repository folder, in the terminal in which ESP-IDF is
active:

```bash
python tools/run_experiments.py
```

For each of the six projects, the script builds from a clean state, flashes
the board, resets it, records its output from the `BEGIN` line to the `END`
line in `<project>/serial.log`, and appends the code-size report of the
linked program. One project takes several minutes; most of this time is the
validation at 10,000 states in emulated double precision.

| Option | Effect |
| --- | --- |
| `vdp_2_64_64_1` (a project name) | runs that project alone |
| `--port COM5`, `--port /dev/ttyUSB0` | names the serial port; without it, the script finds the port |
| `--force` | repeats a project whose log is complete |
| `--skip-build` | reuses the existing `build/` folder |
| `--capture-only` | resets the board and records, without build and flash |
| `--list-ports` | lists the serial ports |

A project whose log is complete is skipped on the next run. At the end, the
six logs are also written to `logs.zip`.

The same steps by hand, for one project, in a Linux or macOS terminal:

```bash
cd bicycle_4_32_32_1
rm -rf build sdkconfig
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor | tee serial.log   # Ctrl+] after "----- END"
python ../tools/size_report.py build/bicycle_cbf.elf >> serial.log
cd ..
```

## Step 6: collect

```bash
python tools/collect.py */serial.log --sim sim_results.txt
```

prints the measurement block of the new logs. With `--tex paper.tex`, the
block between `% BEGIN MEASUREMENTS` and `% END MEASUREMENTS` of that file is
rewritten, and the previous file is kept as `paper.tex.bak`. Execution times
vary between runs and between boards; the memory values and the code sizes
are determined by the build.

## What is measured, and how

Output lines of the firmware:

```
INFO,<key>,<value>
TIME,<example>,<method>,<window>,<irq>,<n>,<min_us>,<med_us>,<p99_us>,<max_us>,<mean_us>
COLD,<example>,<method>,<first_call_us>
MEM,<example>,<method>,<static_B>,<stack_B>,<heap_peak_B>,<heap_allocs>
ERR,<example>,<method>,<quantity>,<max_abs_err>,<max_scaled_err>
KINK,<example>,<tol>,<states closer than tol to a boundary>,<states>
ERRK,...   same as ERR, over the near-boundary states only
FLIP,<example>,<method>,<near-boundary states with scaled error > 1e-3>
SIZE,...   code size and static storage per method (tools/size_report.py)
```

- **Time.** Cycle counts from the `CCOUNT` register at 240 MHz, at 1,000
  seeded states after 16 warm-up calls. The window `ASM` is the constraint
  assembly; the window `STEP` adds the dynamics and the quadratic program.
  Each window is measured with interrupts enabled (`EN`) and with interrupts
  masked on the measuring core (`MASK`). The reported maxima are maxima over
  a sample and are not worst-case execution times.
- **Memory.** Static storage from the symbol table of the linked program;
  stack usage by filling the stack with a pattern and locating the deepest
  overwritten byte; heap usage and the number of allocations with the
  allocation hooks of ESP-IDF, which count the allocations made anywhere in
  the linked program.
- **Error.** Against an evaluation of the same network, with the same
  single-precision parameters, in double precision, at 10,000 seeded states.
  The scaled error is `|q - q_ref| / max(1, |q_ref|)`. For the ReLU
  networks, the states within `1e-4` of an activation boundary are counted
  and reported separately, because the single-precision gate can differ from
  the exact gate at those states.
- **Code size.** Size in flash of the functions of each method.

## Provenance of the reported numbers

| Item | Value |
| --- | --- |
| Compiler | `dual-cbf-compiler` 0.2.0 (first line of each `dual_cbf.h`) |
| Board | ESP32-S3, chip revision v0.2, 240 MHz, measuring core 0 |
| ESP-IDF | v5.5.4; the logs report the version string `v5.5.4-dirty` |
| C++ compiler | GCC 14.2.0, `-O2 -ffast-math -funroll-loops` |
| Timing states | 1,000 per method, seed 20261109, after 16 warm-up calls |
| Validation states | 10,000 per project, seed 20261110 |
| Memory states | seed 20261111 |
| Sampling | uniform on the sampling box of each project (`<project>/main/example.h`) |
| Simulation noise | seeds 1, 2, and 3 for the oscillator, the bicycle, and the pendulum |
| Training | seed 0 |

The seeds of the firmware are defined in
[`shared/bench_main.inc`](shared/bench_main.inc). The firmware prints the
timing and validation seeds, the versions of ESP-IDF and GCC, the flags, and
the chip revision in the `INFO` lines of each log. ESP-IDF appends the suffix
`-dirty` to its version string if tracked files of its installation differ
from the tagged release.

## Repository layout

| Path | Purpose |
| --- | --- |
| `01_train_models.py` | trains the six networks in PyTorch and exports them to ONNX |
| `02_compile_headers.py` | calls the compiler on each ONNX file (`dual_cbf.h`) and generates the comparison methods and the reference (`methods.h`) |
| `03_run_closed_loop.py` | closed-loop simulations on the host and the grid check of the pendulum barrier |
| `plot_closed_loop.m` | MATLAB script of the closed-loop figure |
| `closed_loop_data.mat`, `sim_results.txt` | stored outputs of `03_run_closed_loop.py` |
| `paper_logs/` | the six serial logs behind the numbers of the paper |
| `tools/gen_methods.py` | generator of `methods.h`: statically allocated reverse mode, forward-over-reverse, the three-component algebra, and the double-precision reference |
| `tools/check_reference.py` | compares the double-precision reference with `torch.autograd` in double precision |
| `tools/run_experiments.py` | builds, flashes, runs, and records all six projects |
| `tools/size_report.py` | reads the code size of each method from the linked program |
| `tools/collect.py` | parses the serial logs and writes the measurement block |
| `tools/count_qp_fallback.cpp` | counts the timing states at which the fallback input is used |
| `shared/bench_main.inc` | benchmark driver, common to all projects: validation, timing, stack probe, allocation hooks |
| `shared/methods_rd1.inc`, `shared/methods_rd2.inc` | the methods under test for relative degree one and two |
| `shared/reverse_mode_ad.h` | reverse mode with a heap-allocated cache (`rev_heap`) |
| `<project>/cbf.pt`, `<project>/cbf.onnx` | trained network |
| `<project>/main/dual_cbf.h` | emitted by the compiler |
| `<project>/main/methods.h` | emitted by `tools/gen_methods.py` |
| `<project>/main/example.h` | dynamics, sampling box, input set, nominal input, and class-K gains |
| `<project>/sdkconfig.defaults` | 240 MHz, `-O2`, allocation hooks enabled, task watchdog disabled |
| `Makefile` | shortcuts for the host-side steps (`make host-test`, `make check-ref`, `make sim`, `make collect-paper`) |

## Troubleshooting

- **`idf.py` is not recognized.** ESP-IDF is not active in the terminal. On
  Windows, use the "ESP-IDF PowerShell" shortcut; on Linux and macOS, run
  `. ~/esp/esp-idf/export.sh`.
- **`INFO,heap_hooks,0`, or `INFO,measured_cycles_per_us` near 160.** An old
  `sdkconfig` overrides `sdkconfig.defaults`. Delete the folder `build` and
  the file `sdkconfig` of the project and build again;
  `tools/run_experiments.py` does this by itself.
- **The flash step prints `Connecting....` without success.** Hold the BOOT
  button, press and release RST, release BOOT, and run the command again.
- **The serial port is busy.** Another monitor is attached to the port;
  `Ctrl+]` exits the ESP-IDF monitor.
- **`Permission denied: /dev/ttyUSB0`** on Linux: add the user to the group
  `dialout` (`sudo usermod -aG dialout $USER`), then log out and in.
- **The build fails in a folder whose path contains a space.** Move the
  repository to a short path without spaces, for example
  `C:\esp\dual_cbf_esp32_experiments`.
- **`INFO,validation,FAIL`.** A generated header does not match the network.
  Run `python 02_compile_headers.py` to the end and repeat the step.

## Status and intended use

These are benchmark and validation experiments for a research paper. They
measure the timing, the memory, and the numerical agreement of generated
code on a development board on a bench. The closed loops are simulated on a
host computer; no processor-in-the-loop, hardware-in-the-loop, or
physical-plant experiment is included. The barrier networks are trained by
regression onto analytic targets and carry no safety certificate. Nothing in
this repository is a flight controller, a vehicle controller, or a certified
safety component, and nothing in it has been evaluated against ISO 26262,
DO-178C, or IEC 61508. Anyone who adapts this code to a system is
responsible for the verification, validation, and certification of that
system.

## Citation

```bibtex
@misc{kamaldar2026dual,
  author = {Kamaldar, Mohammadreza},
  title  = {Extracting exact {L}ie derivatives without backpropagation:
            {A} dual compiler for neural control barrier functions},
  year   = {2026},
  url    = {https://arxiv.org/pdf/2604.23995}
}
```

## License

This repository is released under the MIT License; see [LICENSE](LICENSE).
The compiler, `dual-cbf-compiler`, is a separate project under the Apache
License 2.0.
