"""Measure WW3 agreement on the actual common times, cells and station spectra."""

import argparse
import json
from pathlib import Path

import numpy as np

from case_io import load_case, moments, read_config


def errors(prediction, reference, valid):
    """Float64 reductions; an identically zero reference has no relative metric."""
    prediction, reference = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    valid = np.broadcast_to(valid, prediction.shape)
    if not np.any(valid):
        return dict(count=0, relative_l2=None, peak_normalized=None, rmse=None, max_absolute=None)
    if not np.isfinite(prediction[valid]).all() or not np.isfinite(reference[valid]).all():
        raise ValueError("Nonfinite values in the declared comparison domain")
    delta = np.where(valid, prediction - reference, 0)
    norm = np.linalg.norm(reference[valid].ravel())
    peak = np.max(abs(reference[valid]))
    location = np.unravel_index(np.argmax(np.where(valid, abs(delta), -np.inf)), delta.shape)
    absolute = float(abs(delta[location]))
    return dict(count=int(valid.sum()), relative_l2=float(np.linalg.norm(delta.ravel()) / norm) if norm else None,
                peak_normalized=absolute / float(peak) if peak else None,
                rmse=float(np.sqrt(np.mean(delta[valid]**2))), max_absolute=absolute,
                max_absolute_index=[int(i) for i in location])


def circular_errors(prediction, reference, valid):
    difference = (prediction - reference + 180) % 360 - 180
    return errors(difference, np.zeros_like(difference), valid)


def centroid_trace(hs, case, coordinate, axis, periodic):
    """Energy-weighted pulse centre; circular longitude is unwrapped in time."""
    weights = np.where(case["wet"][None], hs**2, 0).astype(np.float64)
    if coordinate == "Spherical":
        weights *= np.cos(np.deg2rad(case["y"]))[None, None, :]
    values = case["x"] if axis == "x" else case["y"]
    marginal = weights.sum(axis=2 if axis == "x" else 1)
    if periodic:
        phase = np.deg2rad(values)
        circular = marginal @ np.exp(1j * phase)
        return np.unwrap(np.angle(circular)) * 6366197.7236758135
    total = marginal.sum(axis=1)
    centre = np.divide(marginal @ values, total, out=np.full_like(total, np.nan), where=total > 0)
    return centre * (np.pi / 180 * 6366197.7236758135) if coordinate == "Spherical" else centre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="run_case output directory")
    args = parser.parse_args()
    config = read_config(args.config)
    case = load_case(config, args.data_root)
    if json.loads((args.output / "run.json").read_text())["config"] != config:
        raise ValueError("Run and comparison configurations differ; rerun the declared case")
    with np.load(args.output / "prediction.npz") as loaded:
        predicted = dict(loaded)
    np.testing.assert_array_equal(predicted["time"], case["times"].astype(np.int64))
    for stored, original in (("x", "x"), ("y", "y"), ("theta_deg", "theta"), ("freq", "freq"), ("wet", "wet")):
        np.testing.assert_array_equal(predicted[stored], case[original])
    # Prescribed cells are displayed but excluded from the principal accuracy
    # metrics: boundary forcing is input, not an independently predicted answer.
    free = case["wet"] & ~case["boundary_mask"]
    swh = predicted["swh"]
    swh_steps = [errors(p, r, free) for p, r in zip(swh, case["fields"]["swh"])]
    # Direction validity is chosen solely from the reference wave height.
    min_height = config.get("direction_min_swh_m", 0.01)
    direction_valid = free[None] & (case["fields"]["swh"] >= min_height)
    bulk = dict(swh=errors(swh, case["fields"]["swh"], np.broadcast_to(free, swh.shape)),
                mwd_degrees=circular_errors(predicted["mwd"], (case["fields"]["mwd_from"] + 180) % 360,
                                            direction_valid))
    bulk["mwd_degrees"]["undefined_prediction_count"] = int(np.sum(
        direction_valid & (predicted["directional_resultant"] == 0)))
    if "mwp" in case["fields"]:
        bulk["mwp_seconds"] = errors(predicted["mwp"], case["fields"]["mwp"], direction_valid)
    common, pi, ri = np.intersect1d(case["times"], case["spectrum_times"], return_indices=True)
    station_prediction = predicted["spectra"][pi][:, case["station_ix"], case["station_iy"]]
    station_reference = case["station_spectra"][ri]
    station_free = free[case["station_ix"], case["station_iy"]]
    station_metrics = errors(station_prediction, station_reference,
                            np.broadcast_to(station_free[None, :, None, None], station_prediction.shape))
    station_steps = [errors(p, r, station_free[:, None, None])
                     for p, r in zip(station_prediction, station_reference)]
    pm = moments(station_prediction, case["theta"], case["freq"])
    rm = moments(station_reference, case["theta"], case["freq"])
    station_valid = station_free[None] & (rm["swh"] >= min_height)
    common_moments = dict(swh=errors(pm["swh"], rm["swh"], station_free[None]),
                          mwp_seconds=errors(pm["mwp"], rm["mwp"], station_valid),
                          mwd_degrees=circular_errors(pm["mwd"], rm["mwd"], station_valid))
    seconds = (case["times"] - case["times"][0]).astype("timedelta64[s]").astype(float)
    motion = None
    if config["case"] in ("tp1.1", "tp1.2"):
        axis = "x" if config["case"] == "tp1.1" else "y"
        centres = [centroid_trace(h, case, config["coordinate_type"], axis, axis == "x")
                   for h in (swh, case["fields"]["swh"])]
        fit = seconds <= 48 * 3600
        motion = dict(fit_hours=48, predicted_speed_m_s=float(np.polyfit(seconds[fit], centres[0][fit], 1)[0]),
                      ww3_speed_m_s=float(np.polyfit(seconds[fit], centres[1][fit], 1)[0]),
                      predicted_displacement_m=(centres[0] - centres[0][0]).tolist(),
                      ww3_displacement_m=(centres[1] - centres[1][0]).tolist())
    integral = predicted["integral"].astype(np.float64)
    report = dict(case=config["case"], status="MEASURED_NOT_THRESHOLD_CERTIFIED",
                  scope="Comparison against WW3 regression reference data",
                  frames=len(seconds), forecast_steps=len(seconds)-1, initial_frame_included=True,
                  score_domain="MAPSTA=1, prescribed MAPSTA=2 excluded",
                  direction_min_reference_swh_m=min_height,
                  bulk=bulk, common_quadrature_station_moments=common_moments,
                  station_spectrum=station_metrics, station_common_frames=len(common),
                  station_names=case["station_names"].tolist(),
                  station_forecast_steps=[dict(time=str(t), **m) for t, m in zip(common, station_steps)],
                  swh_forecast_steps=[dict(time=str(t), **m) for t, m in zip(case["times"], swh_steps)],
                  motion=motion,
                  budget=dict(final_to_initial_integral=float(integral[-1] / integral[0]),
                              cumulative_clipping_to_initial_integral=float(predicted["clipping_mass"].sum(dtype=np.float64) / integral[0]),
                              interpretation="Closed periodic no-source conservation only for tp1.1; other domains are open or prescribed"))
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("case", "status", "bulk", "station_spectrum", "budget")}, indent=2))


if __name__ == "__main__":
    main()
