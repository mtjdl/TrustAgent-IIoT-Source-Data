# Executable validation report

> **Evidence label:** executable calibrated simulation. This is not raw Edge-IIoTset/TON_IoT training, a Hyperledger Fabric deployment, or a physical testbed.

## Protocol

- Clients: 20; seeds: 11, 29, 47, 71, 101, 131, 173, 211, 257, 307.
- Per regime/seed: 7,200 generated training events and 3,600 independently generated test events.
- Non-IID partition: client-specific Dirichlet class probabilities; client covariate shifts; seven latent traffic classes and twelve generated telemetry features.
- Training: per-client nearest prototypes; all collaborative models are computed from client training prototypes only.
- Preprocessing: training-fitted z-standardization applied unchanged to test events.
- Network: 30-round, byte-accounted, single-bottleneck discrete-event model at 5 Mbit/s.
- Ledger: Fabric-like batching/ordering/validation queue; no Fabric executable was run.
- Coordinator: 30 persistent-state rounds in the order Schedule -> ProvisionalReference -> TrustFilter -> Fuse -> Policy -> Commit; subquorum sets fail closed without forced admission.

## Standard-regime results (mean +/- sample SD across 10 regenerated seeds)

| Regime | Method | Accuracy (%) | Macro-F1 (%) | AUC (%) | MB/round | Round time (s) | Policy block (%) | False block (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Edge-IIoTset-calibrated | Local-only | 92.99 +/- 0.42 | 92.63 +/- 0.41 | 96.17 +/- 0.34 | 0.08 | 8.35 | 0.0 | 0.0 |
| Edge-IIoTset-calibrated | FedAvg | 95.05 +/- 0.43 | 94.81 +/- 0.42 | 97.12 +/- 0.31 | 8.57 | 32.07 | 0.0 | 0.0 |
| Edge-IIoTset-calibrated | FedProx | 95.46 +/- 0.42 | 95.20 +/- 0.39 | 98.00 +/- 0.57 | 8.38 | 32.77 | 0.0 | 0.0 |
| Edge-IIoTset-calibrated | FD-only | 96.85 +/- 0.36 | 96.70 +/- 0.36 | 98.40 +/- 0.27 | 1.85 | 22.63 | 0.0 | 0.0 |
| Edge-IIoTset-calibrated | BC-FL | 97.19 +/- 0.29 | 97.04 +/- 0.27 | 98.40 +/- 0.32 | 4.99 | 27.60 | 60.2 | 5.7 |
| Edge-IIoTset-calibrated | TrustAgent-IIoT | 98.24 +/- 0.21 | 98.10 +/- 0.20 | 99.11 +/- 0.23 | 1.46 | 20.49 | 96.1 | 2.2 |
| TON_IoT-calibrated | Local-only | 92.15 +/- 0.46 | 91.45 +/- 0.50 | 96.08 +/- 0.39 | 0.08 | 8.35 | 0.0 | 0.0 |
| TON_IoT-calibrated | FedAvg | 94.18 +/- 0.43 | 93.70 +/- 0.53 | 97.14 +/- 0.27 | 8.57 | 32.07 | 0.0 | 0.0 |
| TON_IoT-calibrated | FedProx | 94.84 +/- 0.42 | 94.34 +/- 0.49 | 97.85 +/- 0.31 | 8.38 | 32.77 | 0.0 | 0.0 |
| TON_IoT-calibrated | FD-only | 96.09 +/- 0.32 | 95.79 +/- 0.33 | 98.11 +/- 0.27 | 1.85 | 22.63 | 0.0 | 0.0 |
| TON_IoT-calibrated | BC-FL | 96.50 +/- 0.36 | 96.16 +/- 0.44 | 98.12 +/- 0.20 | 4.99 | 27.60 | 58.9 | 5.8 |
| TON_IoT-calibrated | TrustAgent-IIoT | 97.85 +/- 0.24 | 97.56 +/- 0.28 | 99.11 +/- 0.16 | 1.46 | 20.49 | 96.1 | 2.3 |

## QA result

- Unchanged Edge figure tolerance check: **FAIL**. Exact differences are in `figure_consistency_checks.csv`.
- Algorithm branch vectors: **15/15 PASS** (`algorithm_test_vectors.csv`).
- Internal reproducibility/trace QA: **10/10 PASS** (`internal_qa.csv`).
- Policy TP/FP/FN/TN decisions are reconstructable from `policy_events.csv`.
- Per-round bytes and timing are reconstructable from `network_ledger_events.csv`; load-test commitment latency is reconstructable from `fabric_like_transaction_events.csv`.
- Cross-round reputation transitions, ordered stages, and hash-linked receipts are reconstructable from `coordinator_client_state.csv`, `coordinator_event_trace.csv`, and `audit_receipts.csv`.
- Figure 7 incremental fits and Figure 10 branch ablations are executable simulator observations. Figure 11/12 sensitivity curves and Figure 14 resources are explicitly labeled configured model estimates with formulas and boundaries in their CSV files.
- Paired exact sign-flip tests are reported only as simulator-repeatability diagnostics in `paired_seed_tests.csv`.

## Interpretation boundary

These results replace pseudo-replicates with independently regenerated event streams and executable mechanisms. They support computational consistency and reproducibility of the stated design under the declared synthetic assumptions. They do not demonstrate performance on the original benchmark records, a runnable production stack, or a physical IIoT deployment.
