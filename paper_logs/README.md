# Serial logs behind the numbers of the paper

One log per project, recorded by `tools/run_experiments.py` on an ESP32-S3
(chip revision v0.2) with the firmware of this repository. Each log holds the
output of the board from the `BEGIN` line to the `END` line, followed by the
code-size report of the linked program (`SIZE` lines).

From the repository folder,

```bash
python tools/collect.py paper_logs/*.log --sim sim_results.txt
```

prints the measurement block of the paper. The line formats are described in
the main [README](../README.md#what-is-measured-and-how).
