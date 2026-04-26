# Convenience driver for the ESP32-S3 experiment workflow.
#
#   make install       Install the Python dependencies into the active venv.
#   make train         Step 1: train the three neural CBFs and export ONNX.
#   make compile       Step 2: emit dual_cbf.h for each example.
#   make host-test     Optional pre-flight: compile and run host regression
#                      tests for all three examples (no MCU needed).
#   make sim           Step 3: closed-loop simulations for Figs 4 and 6.
#   make collect       Step 4: parse the captured serial logs and write
#                      paper_measurements.tex.
#   make all           Steps 1, 2, host-test, sim (everything except the
#                      ESP-IDF flash, which is interactive).
#   make clean         Remove generated artifacts (kept: source code).

PY := python3

.PHONY: all install train compile host-test sim collect clean

all: train compile host-test sim
	@echo
	@echo "All host-side steps complete. Now flash each example onto the"
	@echo "ESP32-S3 (see README.md, section 5), capture serial.log per"
	@echo "example, then run: make collect"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) 01_train_models.py

compile:
	$(PY) 02_compile_headers.py

host-test:
	@echo "=== bicycle host regression ==="
	cd bicycle_4_32_32_1/main && \
		g++ -O2 -std=c++17 -DDUAL_CBF_HOST_TEST -I. -I../../shared \
		    host_test.cpp -o /tmp/bicycle_host && \
		/tmp/bicycle_host
	@echo "=== vdp host regression ==="
	cd vdp_2_64_64_1/main && \
		g++ -O2 -std=c++17 -DDUAL_CBF_HOST_TEST -I. -I../../shared \
		    host_test.cpp -o /tmp/vdp_host && \
		/tmp/vdp_host
	@echo "=== pendulum host regression ==="
	cd pendulum_2_32_32_1/main && \
		g++ -O2 -std=c++17 -DDUAL_CBF_HOST_TEST -I. -I../../shared \
		    host_test.cpp -o /tmp/pend_host && \
		/tmp/pend_host
	@echo
	@echo "Host regression PASS for all three examples."

sim:
	$(PY) 03_run_closed_loop.py

collect:
	$(PY) 04_collect_results.py \
	    bicycle_4_32_32_1/serial.log \
	    vdp_2_64_64_1/serial.log \
	    pendulum_2_32_32_1/serial.log

clean:
	# Generated artifacts: trained nets, ONNX, headers, sim outputs.
	rm -f bicycle_4_32_32_1/cbf.{pt,onnx} bicycle_4_32_32_1/main/dual_cbf.h
	rm -f vdp_2_64_64_1/cbf.{pt,onnx} vdp_2_64_64_1/main/dual_cbf.h
	rm -f pendulum_2_32_32_1/cbf.{pt,onnx} pendulum_2_32_32_1/main/dual_cbf.h
	rm -f bicycle_closed_loop.pdf vdp_phase.pdf
	rm -f paper_measurements.tex
	# ESP-IDF build artifacts (the build/ directory is large)
	rm -rf bicycle_4_32_32_1/build vdp_2_64_64_1/build pendulum_2_32_32_1/build
	rm -f bicycle_4_32_32_1/sdkconfig vdp_2_64_64_1/sdkconfig pendulum_2_32_32_1/sdkconfig
