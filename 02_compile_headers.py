"""Translate the trained ONNX models into bare-metal C++ headers.

Run from the repository root after step 01:
    pip install dual-cbf-compiler
    python 02_compile_headers.py

This produces:
    bicycle_4_32_32_1/main/dual_cbf.h     (relative_degree=1)
    vdp_2_64_64_1/main/dual_cbf.h         (relative_degree=1)
    pendulum_2_32_32_1/main/dual_cbf.h    (relative_degree=2)

Each header is self-contained C++17, allocates zero dynamic memory,
embeds all weights as static const float arrays, and exposes the
public entry point evaluate_cbf (or evaluate_cbf_2nd_order for the
relative-degree-two pendulum). It is consumed verbatim by the ESP-IDF
project in <example>/main/.
"""

from __future__ import annotations

from pathlib import Path

import sys

import dual_cbf_compiler
from dual_cbf_compiler import emit_cpp_header, parse_onnx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from gen_methods import generate as generate_methods  # noqa: E402

EXAMPLES = [
    # (subdirectory,                relative_degree, number of inputs m)
    ("bicycle_4_32_32_1",            1, 2),
    ("vdp_2_64_64_1",                1, 1),
    ("pendulum_2_32_32_1",           2, 1),
    # Depth sweep: Van der Pol target, ReLU, width 32, 2/4/8 hidden layers.
    ("depth_2x32",                   1, 1),
    ("depth_4x32",                   1, 1),
    ("depth_8x32",                   1, 1),
]


def main() -> None:
    print(f"dual_cbf_compiler {dual_cbf_compiler.__version__}")
    for subdir, rd, m in EXAMPLES:
        onnx_path = ROOT / subdir / "cbf.onnx"
        out_path = ROOT / subdir / "main" / "dual_cbf.h"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"=== {subdir} (relative_degree={rd}) ===")
        if not onnx_path.exists():
            print(f"  ERROR: {onnx_path} not found; run 01_train_models.py first.")
            continue

        network = parse_onnx(onnx_path.as_posix(), relative_degree=rd)
        header = emit_cpp_header(network, relative_degree=rd, namespace="dual_cbf",
                                 n_inputs=m if rd == 1 else None)
        out_path.write_text(header)
        # Comparison methods (static reverse mode, shared-primal dual sweep,
        # exact second-order baseline, three-component algebra, reference)
        # that reuse the weight arrays of the compiled header.
        (out_path.parent / "methods.h").write_text(generate_methods(header, m))
        print(
            f"  depth={network.depth}  "
            f"widths={network.widths}  "
            f"max_width={max(network.widths)}  "
            f"-> {out_path.relative_to(ROOT)}"
        )

    print("\nAll headers and methods.h files generated. Build them with the ESP-IDF projects.")


if __name__ == "__main__":
    main()
