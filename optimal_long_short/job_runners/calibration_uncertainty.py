"""Paired moving-block bootstrap for calibration and sizing uncertainty.

This job conditions on the empirical showcase's selected long/short roles,
endpoint EWM drift signal, and role-specific AAVE carry. It resamples paired,
contiguous blocks of the centered causal-EWM innovations; recalibrates only the
zero-log-mean residual shape; and propagates each estimate to a reference
survival probability and an explicitly liquidation-constrained health-buffer
selection.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from optimal_long_short.calibration import params_to_theta
from optimal_long_short.calibration_uncertainty import (
    PARAMETER_NAMES,
    calibration_bootstrap_record,
    liquidation_constrained_downstream,
    moving_block_bootstrap_indices,
    summarize_bootstrap_records,
)
from optimal_long_short.job_runners.calibrate_eth_btc import (
    CALIB_BOUNDS,
    compute_returns,
    load_aave_data,
    orient_residual_params,
    prepare_causal_innovations,
)
from optimal_long_short.job_runners.common import (
    DEFAULT_EMPIRICAL_PARAMS,
    RESULTS_DIR,
    load_calibrated_params,
)
from optimal_long_short.drift import with_expected_log_return_drift
from optimal_long_short.model_params import KouParams

DEFAULT_DETAIL_OUT = RESULTS_DIR / "calibration_uncertainty_bootstrap.csv"
DEFAULT_SUMMARY_OUT = RESULTS_DIR / "calibration_uncertainty_summary.csv"
DEFAULT_METADATA_OUT = RESULTS_DIR / "calibration_uncertainty_metadata.json"

DOWNSTREAM_METRICS = (
    "p_surv_at_reference_H0",
    "p_liq_at_reference_H0",
    "selected_H0",
    "selected_h0",
    "selected_initial_leverage",
    "selected_p_surv",
    "selected_p_liq",
)

DETAIL_FIELDS = (
    "replicate",
    "calibration_success",
    "selection_feasible",
    "objective",
    "n_iter",
    "error",
    *PARAMETER_NAMES,
    "reference_H0",
    *DOWNSTREAM_METRICS,
)

SUMMARY_FIELDS = (
    "metric",
    "point_estimate",
    "bootstrap_mean",
    "bootstrap_std",
    "ci_lower",
    "bootstrap_median",
    "ci_upper",
    "confidence_level",
    "n_finite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_EMPIRICAL_PARAMS,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=40)
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Observations per circular block; default is ceil(N^(1/3)).",
    )
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument("--pbar", type=float, default=0.10)
    parser.add_argument("--T", type=float, default=1.0 / 12.0)
    parser.add_argument("--H0-max", type=float, default=2.0)
    parser.add_argument("--H0-count", type=int, default=200)
    parser.add_argument("--detail-out", type=Path, default=DEFAULT_DETAIL_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    return parser.parse_args()


def _params_from_mapping(values: dict[str, Any]) -> KouParams:
    return KouParams(**{name: float(values[name]) for name in PARAMETER_NAMES})


def _apply_fixed_showcase_inputs(
    residual_market_order: KouParams,
    *,
    long_asset: str,
    short_asset: str,
    endpoint_trends: dict[str, float],
    long_carry: float,
    short_carry: float,
) -> KouParams:
    """Orient a residual fit and add the fixed point-showcase drift inputs."""

    if {long_asset, short_asset} != {"WETH", "WBTC"}:
        raise ValueError("The showcase must assign WETH and WBTC to opposite roles")
    residual_oriented = orient_residual_params(residual_market_order, long_asset)
    return with_expected_log_return_drift(
        residual_oriented,
        drift1=float(endpoint_trends[long_asset]) + float(long_carry),
        drift2=float(endpoint_trends[short_asset]) + float(short_carry),
    )


def _replicate_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker so multiprocessing works under spawn and fork."""

    replicate = int(task["replicate"])
    try:
        r1 = np.asarray(task["r1"], dtype=float)
        r2 = np.asarray(task["r2"], dtype=float)
        rng = np.random.default_rng(int(task["bootstrap_seed"]))
        indices = moving_block_bootstrap_indices(
            len(r1),
            int(task["block_length"]),
            rng,
        )
        # The point calibration treats directional location as a fixed EWM
        # input, not as part of the residual law. Recenter each resampled
        # innovation series under that same zero-log-mean restriction while
        # retaining the paired circular-block order.
        sample1 = r1[indices]
        sample2 = r2[indices]
        sample1 = sample1 - float(np.mean(sample1))
        sample2 = sample2 - float(np.mean(sample2))
        sample_indices = np.arange(len(indices), dtype=int)
        raw_point = _params_from_mapping(task["raw_point"])
        health_grid = np.asarray(task["health_grid"], dtype=float)

        return calibration_bootstrap_record(
            sample1,
            sample2,
            sample_indices,
            dt_years=float(task["dt_years"]),
            replicate=replicate,
            calibration_kwargs={
                "bounds": CALIB_BOUNDS,
                "theta0": params_to_theta(raw_point),
                "n_starts": int(task["n_starts"]),
                "seed": int(task["calibration_seed"]),
                "run_diagnostics": False,
                "max_moment_order": 4,
                "drift_mode": "zero_expected_log_return",
            },
            parameter_adjuster=lambda params: _apply_fixed_showcase_inputs(
                params,
                long_asset=str(task["long_asset"]),
                short_asset=str(task["short_asset"]),
                endpoint_trends=dict(task["endpoint_trends"]),
                long_carry=float(task["long_carry"]),
                short_carry=float(task["short_carry"]),
            ),
            downstream_evaluator=lambda params: liquidation_constrained_downstream(
                params,
                health_grid,
                reference_H0=float(task["reference_H0"]),
                pbar=float(task["pbar"]),
                b=float(task["b"]),
                T=float(task["T"]),
                S10=float(task["S10"]),
                S20=float(task["S20"]),
                ltv_max=float(task["ltv_max"]),
            ),
            max_moment_order=4,
        )
    except Exception as exc:  # Preserve failures for an auditable failure rate.
        return {
            "replicate": replicate,
            "calibration_success": False,
            "selection_feasible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_replicates < 2:
        raise ValueError("--bootstrap-replicates must be at least 2")
    if args.n_starts < 1:
        raise ValueError("--n-starts must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must lie in (0, 1)")
    if not 0.0 <= args.pbar <= 1.0:
        raise ValueError("--pbar must lie in [0, 1]")
    if not math.isfinite(args.T) or args.T <= 0.0:
        raise ValueError("--T must be finite and positive")
    if args.H0_count < 2:
        raise ValueError("--H0-count must be at least 2")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    started = time.perf_counter()

    payload = json.loads(args.params.read_text())
    meta = payload.get("meta", {})
    selection = payload.get("orientation_selection", {})
    preprocessing_meta = meta.get("returns_preprocessing", {})
    market_order = meta.get("market_calibration_order")
    if market_order != ["WETH", "WBTC"]:
        raise ValueError(
            "The uncertainty job expects residual calibration in market order "
            "['WETH', 'WBTC']"
        )
    if payload.get("ecf", {}).get("drift_mode") != "zero_expected_log_return":
        raise ValueError(
            "The empirical artifact must come from the shape-only, zero-log-mean "
            "ECF calibration"
        )
    if preprocessing_meta.get("method") != (
        "centered_lagged_normalized_ewm_innovations"
    ):
        raise ValueError(
            "The empirical artifact does not use centered causal-EWM innovations"
        )

    point_params, constraint = load_calibrated_params(args.params)
    raw_point = _params_from_mapping(payload["params_residual_ecf_market_order"])
    long_asset = str(selection["long_asset"])
    short_asset = str(selection["short_asset"])
    if (
        meta.get("asset1") != long_asset
        or meta.get("asset2") != short_asset
        or constraint.get("collateral_asset") != long_asset
        or constraint.get("debt_asset") != short_asset
    ):
        raise ValueError(
            "The saved parameter order, selected roles, and AAVE contract roles "
            "are inconsistent"
        )
    endpoint_trends = {
        asset: float(preprocessing_meta["endpoint_ewm_mean_annualized"][asset])
        for asset in market_order
    }
    long_carry = float(selection["long_supply_rate"])
    short_carry = float(selection["short_borrow_rate"])
    drift_components = payload.get("drift_components_by_asset", {})
    for asset, carry in ((long_asset, long_carry), (short_asset, short_carry)):
        if not math.isclose(
            float(drift_components[asset]["role_carry"]),
            carry,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(f"Saved role carry is inconsistent for {asset}")

    data, data_meta = load_aave_data()
    expected_sources = meta.get("source_files", {})
    loaded_sources = {
        "WBTC": data_meta["btc_file"],
        "WETH": data_meta["eth_file"],
    }
    if expected_sources and expected_sources != loaded_sources:
        raise ValueError(
            "The latest AAVE source files differ from the point-calibration files; "
            "re-run the point calibration or restore its source data first."
        )

    dt_years = float(meta.get("dt_years", data_meta["dt_years"]))
    if not math.isclose(
        dt_years,
        float(data_meta["dt_years"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("The current return frequency differs from the point fit")
    r_weth_raw, r_wbtc_raw, date0, date1 = compute_returns(data)
    half_life_years = float(preprocessing_meta["ewm_half_life_years"])
    r1, r2, preprocessing = prepare_causal_innovations(
        r_weth_raw,
        r_wbtc_raw,
        dt_years=dt_years,
        horizon_years=half_life_years,
    )
    if len(r1) != int(meta.get("n_obs", len(r1))):
        raise ValueError("The innovation sample length differs from the point fit")
    for asset in market_order:
        reproduced_trend = float(preprocessing[asset].mean_path[-1] / dt_years)
        if not math.isclose(
            reproduced_trend,
            endpoint_trends[asset],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"The current data do not reproduce the saved endpoint EWM trend "
                f"for {asset}"
            )

    block_length = (
        math.ceil(len(r1) ** (1.0 / 3.0))
        if args.block_length is None
        else int(args.block_length)
    )
    if block_length < 1 or block_length > len(r1):
        raise ValueError("--block-length must lie between 1 and N")

    H0_min = float(constraint["H0_min"])
    if not math.isfinite(args.H0_max) or args.H0_max <= H0_min:
        raise ValueError("--H0-max must be finite and above the protocol minimum")
    health_grid = np.linspace(H0_min, args.H0_max, args.H0_count)

    point_selection = liquidation_constrained_downstream(
        point_params,
        health_grid,
        reference_H0=H0_min,
        pbar=args.pbar,
        b=constraint["b"],
        T=args.T,
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
        ltv_max=constraint["ltv_max"],
    )
    if not point_selection["selection_feasible"]:
        raise ValueError("No point-estimate buffer is feasible on the requested grid")
    reference_H0 = float(point_selection["selected_H0"])
    point_downstream = liquidation_constrained_downstream(
        point_params,
        health_grid,
        reference_H0=reference_H0,
        pbar=args.pbar,
        b=constraint["b"],
        T=args.T,
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
        ltv_max=constraint["ltv_max"],
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(2 * args.bootstrap_replicates)
    tasks: list[dict[str, Any]] = []
    raw_point_dict = dataclasses.asdict(raw_point)
    for replicate in range(args.bootstrap_replicates):
        tasks.append(
            {
                "replicate": replicate,
                "r1": r1,
                "r2": r2,
                "dt_years": dt_years,
                "block_length": block_length,
                "bootstrap_seed": child_seeds[2 * replicate].generate_state(1)[0],
                "calibration_seed": child_seeds[2 * replicate + 1].generate_state(1)[0],
                "n_starts": args.n_starts,
                "raw_point": raw_point_dict,
                "long_asset": long_asset,
                "short_asset": short_asset,
                "endpoint_trends": endpoint_trends,
                "long_carry": long_carry,
                "short_carry": short_carry,
                "health_grid": health_grid,
                "reference_H0": reference_H0,
                "pbar": args.pbar,
                "b": constraint["b"],
                "T": args.T,
                "S10": constraint.get("S10", 1.0),
                "S20": constraint.get("S20", 1.0),
                "ltv_max": constraint["ltv_max"],
            }
        )

    records: list[dict[str, Any]] = []
    if args.workers == 1:
        records = [_replicate_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_replicate_worker, task) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda record: int(record["replicate"]))

    point_estimates = {
        name: float(getattr(point_params, name)) for name in PARAMETER_NAMES
    }
    point_estimates.update(
        {metric: float(point_downstream[metric]) for metric in DOWNSTREAM_METRICS}
    )
    summary = summarize_bootstrap_records(
        records,
        (*PARAMETER_NAMES, *DOWNSTREAM_METRICS),
        point_estimates=point_estimates,
        confidence_level=args.confidence_level,
    )

    _write_csv(args.detail_out, DETAIL_FIELDS, records)
    _write_csv(args.summary_out, SUMMARY_FIELDS, summary)

    elapsed = time.perf_counter() - started
    converged = sum(bool(record.get("calibration_success")) for record in records)
    selection_feasible = sum(
        bool(record.get("selection_feasible")) for record in records
    )
    metadata = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "method": "paired circular moving-block percentile bootstrap",
        "interpretation": (
            "Descriptive sampling uncertainty conditional on the Kou model, ECF "
            "criterion, centered causal-EWM preprocessing, point-selected asset "
            "roles, fixed endpoint EWM trends, fixed role carry, terminal-block "
            "protocol terms, block "
            "length, and discrete health-factor sizing grid; not a model-selection "
            "or out-of-sample forecast interval."
        ),
        "data": {
            "asset1": long_asset,
            "asset2": short_asset,
            "market_calibration_order": market_order,
            "date_range": [date0, date1],
            "n_obs": len(r1),
            "dt_years": dt_years,
            "returns_preprocessing": {
                "method": preprocessing_meta["method"],
                "ewm_half_life_years": half_life_years,
                "ewm_half_life_periods": float(
                    preprocessing_meta["ewm_half_life_periods"]
                ),
                "replicate_centering": True,
            },
            "source_files": loaded_sources,
        },
        "fixed_empirical_inputs": {
            "orientation_rule": selection["rule"],
            "long_asset": long_asset,
            "short_asset": short_asset,
            "endpoint_ewm_expected_log_drift": endpoint_trends,
            "long_supply_rate": long_carry,
            "short_borrow_rate": short_carry,
            "residual_drift_mode": "zero_expected_log_return",
            "orientation_reselected_per_replicate": False,
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "block_length_observations": block_length,
            "block_length_days": block_length * dt_years * 365.0,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "calibration_starts_per_replicate": args.n_starts,
            "workers": args.workers,
            "converged_replicates": converged,
            "selection_feasible_replicates": selection_feasible,
        },
        "downstream": {
            "rule": "maximize initial leverage subject to p_liq <= pbar",
            "pbar": args.pbar,
            "T": args.T,
            "reference_H0": reference_H0,
            "H0_min": H0_min,
            "H0_max": args.H0_max,
            "H0_count": args.H0_count,
        },
        "runtime_seconds": elapsed,
        "outputs": {
            "replicates": str(args.detail_out),
            "summary": str(args.summary_out),
        },
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2))

    print(
        f"Bootstrap: {converged}/{args.bootstrap_replicates} calibrations converged; "
        f"{selection_feasible}/{args.bootstrap_replicates} selections feasible."
    )
    print(
        f"Paired circular blocks: length={block_length} observations "
        f"({metadata['bootstrap']['block_length_days']:.2f} days)."
    )
    print(
        f"Fixed showcase orientation: long {long_asset} / short {short_asset}; "
        "endpoint EWM trends and role carry held fixed."
    )
    for metric in ("p_surv_at_reference_H0", "selected_H0"):
        row = next(item for item in summary if item["metric"] == metric)
        print(
            f"{metric}: point={row['point_estimate']:.6g}, "
            f"{args.confidence_level:.0%} percentile interval="
            f"[{row['ci_lower']:.6g}, {row['ci_upper']:.6g}]"
        )
    print(f"Replicates -> {args.detail_out}")
    print(f"Summary    -> {args.summary_out}")
    print(f"Metadata   -> {args.metadata_out}")
    print(f"Runtime    -> {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
