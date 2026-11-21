# Driver for the host-side steps of the ESP32-S3 experiments.
# Needs make and g++ (Linux, macOS, or WSL). On Windows without WSL, run the
# Python scripts directly; see README.md.
#
#   make install       Install the Python dependencies into the active venv.
#   make train         Train the six networks and export them to ONNX.
#   make compile       Emit dual_cbf.h and methods.h for each project.
#   make host-test     Pre-flight: build the benchmark driver for the host and
#                      compare all methods with the double-precision
#                      reference at 100000 seeded states (no board needed).
#   make check-ref     Compare that reference with torch.autograd (float64).
#   make sim           Closed-loop simulations (closed_loop.pdf,
#                      closed_loop_data.mat, sim_results.txt) and the grid
#                      check of the pendulum barrier.
#   make collect       Print the measurement block from */serial.log.
#   make collect-paper Print the measurement block from paper_logs/*.log,
#                      which are the logs behind the numbers of the paper.
#   make all           train, compile, host-test, sim (everything except the
#                      build and flash with ESP-IDF).
#   make clean         Remove generated files (the source code and the
#                      serial logs are kept).

PY := python3

PROJECTS := bicycle_4_32_32_1 vdp_2_64_64_1 pendulum_2_32_32_1 depth_2x32 depth_4x32 depth_8x32

.PHONY: all install train compile host-test check-ref sim collect collect-paper clean

all: train compile host-test sim
	@echo
	@echo "All host-side steps are complete. Build and flash the projects with"
	@echo "ESP-IDF (README.md, step 4), then run: make collect"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) 01_train_models.py

compile:
	$(PY) 02_compile_headers.py

host-test:
	@for e in $(PROJECTS); do \
	    echo "=== $$e host pre-flight ==="; \
	    ( cd $$e/main && \
	      g++ -O2 -ffast-math -funroll-loops -std=c++17 -DDUAL_CBF_HOST_TEST \
	          -I. -I../../shared main.cpp methods.cpp -o /tmp/$$e.host && \
	      /tmp/$$e.host > ../host.log; \
	      status=$$?; grep -E "^(KINK|FLIP|INFO,validation)" ../host.log; exit $$status ) || exit 1; \
	done
	@echo
	@echo "Host pre-flight PASS for all six projects (full output: <project>/host.log)."

# Independent check of the double-precision reference against torch.autograd.
check-ref:
	$(PY) tools/check_reference.py

sim:
	$(PY) 03_run_closed_loop.py

# With PAPER=path/to/paper.tex, the block between "% BEGIN MEASUREMENTS" and
# "% END MEASUREMENTS" of that file is rewritten; without it, the block is printed.
collect:
	$(PY) tools/collect.py */serial.log --sim sim_results.txt $(if $(PAPER),--tex $(PAPER))

collect-paper:
	$(PY) tools/collect.py paper_logs/*.log --sim sim_results.txt $(if $(PAPER),--tex $(PAPER))

clean:
	@for e in $(PROJECTS); do \
	    rm -f $$e/cbf.pt $$e/cbf.onnx $$e/main/dual_cbf.h $$e/main/methods.h; \
	    rm -f $$e/host.log $$e/sdkconfig $$e/sdkconfig.old; \
	    rm -rf $$e/build; \
	done
	rm -f closed_loop.pdf closed_loop_data.mat sim_results.txt logs.zip build_output.txt
