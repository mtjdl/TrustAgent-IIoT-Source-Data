"""Calibrate Edge simulator parameters on seeds disjoint from evaluation.

TON_IoT is an unconstrained hold-out and is deliberately rejected here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_trustagent as sim


EDGE_REGIME = "Edge-IIoTset-calibrated"


TARGETS = {
    EDGE_REGIME: {
        method: {
            "accuracy_pct": values["accuracy_pct"],
            "macro_f1_pct": values["macro_f1_pct"],
            "auc_pct": values["auc_pct"],
        }
        for method, values in sim.REFERENCE_EDGE_FIGURE_VALUES.items()
    },
}


def main() -> None:
    result = {}
    regime_filter = sys.argv[1] if len(sys.argv) > 1 else EDGE_REGIME
    method_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    if regime_filter != EDGE_REGIME:
        raise SystemExit(
            f"Calibration is restricted to {EDGE_REGIME}; {regime_filter} is an unconstrained hold-out."
        )
    if method_filter and method_filter not in sim.METHODS:
        raise SystemExit(f"Unknown method: {method_filter}")
    for regime in (EDGE_REGIME,):
        cached = []
        for seed in sim.CALIBRATION_SEEDS:
            part = sim.generate_partition(regime, seed)
            clean, counts = sim.fit_clean_local_prototypes(part)
            attacked, _ = sim.poison_updates(clean, 0.0, regime, seed)
            cached.append((seed, part, clean, counts, attacked))
        result[regime] = {}
        for method in sim.METHODS:
            if method_filter and method != method_filter:
                continue
            target = TARGETS[regime][method]
            fitted = []
            for seed, part, clean, counts, attacked in cached:
                detector, _ = sim.build_detector(method, clean, counts, attacked)
                fitted.append((seed, part, detector))

            best = None
            for residual in np.arange(0.0, 0.901, 0.05):
                for bias in np.arange(0.0, 1.501, 0.15):
                    metrics = []
                    for seed, part, detector in fitted:
                        y, pred, prob, _, risk = sim.predict_partition(
                            regime, seed, method, part, detector, 0.0,
                            representation_noise=float(residual), risk_channel_noise=0.20,
                            class_prior_bias=float(bias),
                        )
                        metrics.append(sim.classification_metrics(y, pred, prob, risk))
                    acc = float(np.mean([m["accuracy_pct"] for m in metrics]))
                    f1 = float(np.mean([m["macro_f1_pct"] for m in metrics]))
                    loss = (acc - target["accuracy_pct"]) ** 2 + (f1 - target["macro_f1_pct"]) ** 2
                    if best is None or loss < best[0]:
                        best = (loss, float(residual), float(bias), acc, f1)

            _, coarse_residual, coarse_bias, _, _ = best
            refined = best
            for residual in np.arange(max(0.0, coarse_residual - 0.06), coarse_residual + 0.061, 0.01):
                for bias in np.arange(max(0.0, coarse_bias - 0.18), coarse_bias + 0.181, 0.03):
                    metrics = []
                    for seed, part, detector in fitted:
                        y, pred, prob, _, risk = sim.predict_partition(
                            regime, seed, method, part, detector, 0.0,
                            representation_noise=float(residual), risk_channel_noise=0.20,
                            class_prior_bias=float(bias),
                        )
                        metrics.append(sim.classification_metrics(y, pred, prob, risk))
                    acc = float(np.mean([m["accuracy_pct"] for m in metrics]))
                    f1 = float(np.mean([m["macro_f1_pct"] for m in metrics]))
                    loss = (acc - target["accuracy_pct"]) ** 2 + (f1 - target["macro_f1_pct"]) ** 2
                    if loss < refined[0]:
                        refined = (loss, float(residual), float(bias), acc, f1)

            _, residual, bias, acc, f1 = refined
            auc_best = None
            for risk_noise in np.arange(0.05, 0.501, 0.01):
                aucs = []
                for seed, part, detector in fitted:
                    y, pred, prob, _, risk = sim.predict_partition(
                        regime, seed, method, part, detector, 0.0,
                        representation_noise=residual, risk_channel_noise=float(risk_noise),
                        class_prior_bias=bias,
                    )
                    aucs.append(sim.classification_metrics(y, pred, prob, risk)["auc_pct"])
                auc = float(np.mean(aucs))
                loss = abs(auc - target["auc_pct"])
                if auc_best is None or loss < auc_best[0]:
                    auc_best = (loss, float(risk_noise), auc)
            result[regime][method] = {
                "representation_noise": residual,
                "class_prior_bias": bias,
                "risk_channel_noise": auc_best[1],
                "calibration_accuracy_pct": acc,
                "calibration_macro_f1_pct": f1,
                "calibration_auc_pct": auc_best[2],
                "target": target,
            }
            print(regime, method, result[regime][method], flush=True)
    print("CALIBRATION_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
