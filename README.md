# TrustAgent-IIoT source data and reproducibility package

This repository contains the source data, executable simulation code, validation outputs, and reproducibility metadata associated with the TrustAgent-IIoT manuscript.

## Scope

The package reports results from an **executable calibrated simulation**. It does not contain raw Edge-IIoTset or TON_IoT records, a Hyperledger Fabric deployment, or measurements from a physical IIoT testbed. The dataset names identify calibrated synthetic regimes only. See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for the complete evidence boundary and validation summary.

## Reproduce the simulation

Python 3.11 or later is recommended.

```bash
python -m pip install -r requirements.txt
python simulate_trustagent.py --output reproduced_run
```

For available command-line options:

```bash
python simulate_trustagent.py --help
```

## Package contents

- `simulate_trustagent.py`: main executable simulator.
- `calibrate_parameters.py`: calibration utility.
- `VALIDATION_REPORT.md`: protocol, headline results, QA status, and interpretation boundary.
- `run_manifest.json`: file sizes and SHA-256 checksums for the generated package.
- `simulation_metadata.json`: execution environment, assumptions, and run metadata.
- CSV files: metrics, sensitivity analyses, event traces, ledger/network records, policy decisions, and QA results.

The original submitted ZIP archive has SHA-256 checksum:

```text
03fff68639cde163998d3d83dbe125c557c83dca53df2024599996f1de578e26
```

## Integrity and interpretation

Use `run_manifest.json` to verify individual files. The generated outputs support computational consistency and reproducibility under the declared synthetic assumptions; they should not be interpreted as direct evidence from the original benchmark records or a production deployment.

## License

No reuse license has been assigned yet. All rights are reserved unless a license is added by the repository owner.
