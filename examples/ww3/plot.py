"""Render case-specific WW3 comparisons using the archived figure labels."""

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from case_io import load_case, read_config


def configure_fonts(font_path=None):
    """Use actual Times New Roman; do not silently substitute another serif."""
    if font_path:
        font_manager.fontManager.addfont(font_path)
    else:
        # WSL can use Windows fonts without copying or distributing them.
        for path in Path("/mnt/c/Windows/Fonts").glob("times*.ttf"):
            font_manager.fontManager.addfont(str(path))
    try:
        resolved = font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as error:
        raise ValueError("Install Times New Roman or pass --font-path /path/to/times.ttf") from error
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14,
                         "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "axes.linewidth": 1.35,
                         "xtick.major.width": 1.1, "ytick.major.width": 1.1,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return resolved


def read_experiment(config_path, data_root, output):
    config = read_config(config_path)
    case = load_case(config, data_root)
    with np.load(output / "prediction.npz") as values:
        prediction = dict(values)
    np.testing.assert_array_equal(prediction["time"], case["times"].astype(np.int64))
    return config, case, prediction


def profile_axes(axis, xlabel, title):
    axis.set(title=title, xlabel=xlabel)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))


def height_lines(axis):
    axis.set_ylabel("H (m)")
    ww3, = axis.plot([], [], color="red", linewidth=1.9, label="WW3_H")
    waden, = axis.plot([], [], color="blue", linewidth=1.5, linestyle="--", label="WADEN_H")
    axis.legend(loc="upper right")
    return ww3, waden


def propagation_figure(experiments):
    """Two wide one-dimensional panels: equatorial and meridional propagation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
    lines = []
    for axis, (config, case, prediction) in zip(axes, experiments):
        vertical = config["case"] == "tp1.2"
        coordinate = case["y"] if vertical else case["x"]
        profile_axes(axis, "Global Y (Lat)" if vertical else "Global X (Lon)",
                     "Wave Height Profile Comparison")
        axis.set_xlim(coordinate[0], coordinate[-1])
        axis.set_ylim(0, max(np.nanmax(case["fields"]["swh"]), np.max(prediction["swh"])) * 1.05)
        # Both archived packets are deep-water, monochromatic experiments.
        # Report phase and group speeds at their actual initial peak frequency.
        peak = int(np.argmax(case["initial"].sum(axis=(0, 1, 2))))
        frequency = float(case["freq"][peak])
        gravity = config["step_config"].get("gravity", 9.806)
        phase_speed = gravity / (2*np.pi*frequency)
        group_speed = phase_speed / 2
        spacing = float(abs(coordinate[1]-coordinate[0]))
        spacing_km = spacing * np.pi/180 * (4e7/(2*np.pi)) / 1000
        grid_text = (f"Δφ = {spacing:g}° (Δy = {spacing_km:.2f} km)" if vertical else
                     f"Δλ = {spacing:g}° (Δx = {spacing_km:.2f} km at the equator)")
        axis.text(0.015, 0.95,
                  f"{config['case']}   c = {phase_speed:.2f} m/s; Cg = {group_speed:.2f} m/s\n{grid_text}",
                  transform=axis.transAxes, va="top", fontsize=12,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
        lines.append(height_lines(axis))

    def update(index):
        for pair, (config, case, prediction) in zip(lines, experiments):
            vertical = config["case"] == "tp1.2"
            coordinate = case["y"] if vertical else case["x"]
            for line, field in zip(pair, (case["fields"]["swh"][index], prediction["swh"][index])):
                field = np.where(case["wet"], field, np.nan)
                values = field[len(case["x"])//2, :] if vertical else field[:, len(case["y"])//2]
                line.set_data(coordinate, values)
    return fig, axes, update


def refraction_figure(experiment):
    """Two wide panels: wave height and coming-from mean direction."""
    _, case, prediction = experiment
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
    for axis, title in zip(axes, ("Wave Height Profile Comparison", "Wave Direction Profile Comparison")):
        profile_axes(axis, "X (m)", title)
        axis.set_xlim(case["x"][0], case["x"][-1])
    axes[0].set_ylim(0, max(np.nanmax(case["fields"]["swh"]), np.max(prediction["swh"])) * 1.10)
    height = height_lines(axes[0])
    axes[0].legend(loc="lower left")
    axes[1].set(ylabel="Direction (deg)", ylim=(240, 330))
    ww3, = axes[1].plot([], [], color="red", linewidth=1.9, label="WW3_mwd")
    waden, = axes[1].plot([], [], color="green", linewidth=1.5, linestyle="--", label="WADEN_mwd")
    axes[1].legend(loc="upper left")
    middle = len(case["y"])//2
    depth_profile = np.where(case["wet"][:, middle], case["depth"][:, middle], np.nan)
    for axis in axes:
        depth_axis = axis.twinx()
        # Keep the auxiliary curve behind wave profiles and their legends.
        depth_axis.set_zorder(axis.get_zorder()-1)
        axis.patch.set_visible(False)
        axis.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
        depth_axis.plot(case["x"], depth_profile, color="0.4", linestyle=":",
                         linewidth=1.6, label="Depth", zorder=1)
        depth_axis.set_ylabel("Depth (m)", color="0.4")
        depth_axis.set_ylim(float(np.nanmax(depth_profile))*1.1, 0)
        depth_axis.tick_params(axis="y", colors="0.4", left=False, labelleft=False,
                               right=True, labelright=True)
        depth_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        depth_axis.legend(loc="lower right")

    def update(index):
        for line, field in zip(height, (case["fields"]["swh"][index], prediction["swh"][index])):
            line.set_data(case["x"], np.where(case["wet"], field, np.nan)[:, middle])
        valid = case["wet"] & (case["fields"]["swh"][index] >= 0.01)
        # Match the archived plot's coming-from convention without its clipping.
        for line, field in zip((ww3, waden), (case["fields"]["mwd_from"][index],
                                              (prediction["mwd"][index] + 180) % 360)):
            line.set_data(case["x"], np.where(valid, field, np.nan)[:, middle])
    return fig, axes, update


def cartesian_figure(experiment):
    """The archived tp2.1 style: only two filled-contour wave-height panels."""
    _, case, prediction = experiment
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), layout="constrained")
    maximum = max(float(np.nanmax(case["fields"]["swh"])), float(np.max(prediction["swh"])))
    levels = np.linspace(0, maximum, 11)
    contours = []
    for axis, title in zip(axes, ("WW3 Wave Height", "WADEN Wave Height")):
        axis.set(title=title, xlabel="X (m)", ylabel="Y (m)")
        axis.set_xlim(case["x"][0], case["x"][-1])
        axis.set_ylim(case["y"][0], case["y"][-1])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.ticklabel_format(style="plain", useOffset=False)
        initial = np.ma.array(case["fields"]["swh"][0].T, mask=~case["wet"].T)
        contour = axis.contourf(case["x"], case["y"], initial, levels=levels,
                               cmap="viridis", corner_mask=False)
        contours.append(contour)
        colorbar = fig.colorbar(contour, ax=axis, label="H (m)", ticks=np.linspace(0, 2.5, 6), pad=0.025)
        colorbar.outline.set_linewidth(1.35)

    def update(index):
        for panel, (axis, field) in enumerate(zip(axes, (case["fields"]["swh"][index],
                                                        prediction["swh"][index]))):
            if hasattr(contours[panel], "remove"):
                contours[panel].remove()
            else:  # Matplotlib 3.7 represents a contour set as separate collections.
                for collection in contours[panel].collections:
                    collection.remove()
            contours[panel] = axis.contourf(case["x"], case["y"],
                                           np.ma.array(field.T, mask=~case["wet"].T),
                                           levels=levels, cmap="viridis", corner_mask=False)
    return fig, axes, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Existing run_case output directory")
    parser.add_argument("--assets", type=Path, help="Image directory; defaults to the run directory")
    parser.add_argument("--companion-output", type=Path, help="tp1.1/tp1.2 partner run; defaults to its sibling directory")
    parser.add_argument("--max-step", type=int,
                        help="Display cap only; default tp1.4: 18000 s, other cases: step 100")
    parser.add_argument("--max-frames", type=int, default=101)
    parser.add_argument("--font-path", type=Path, help="Existing Times New Roman TTF if not installed")
    args = parser.parse_args()
    font = configure_fonts(str(args.font_path) if args.font_path else None)
    experiment = read_experiment(args.config, args.data_root, args.output)
    config, case, _ = experiment
    experiments = [experiment]
    if config["case"] in ("tp1.1", "tp1.2"):
        partner = "tp1_2" if config["case"] == "tp1.1" else "tp1_1"
        other = read_experiment(args.config.parent / f"{partner}.json", args.data_root,
                                args.companion_output or args.output.parent / partner)
        experiments = sorted([experiment, other], key=lambda entry: entry[0]["case"])
        # Calendar origins differ; these experiments share elapsed output time.
        first, second = (entry[1]["times"] for entry in experiments)
        np.testing.assert_array_equal(first-first[0], second-second[0])
        fig, axes, update = propagation_figure(experiments)
        stem = "tp1_1_tp1_2"
    elif config["case"] == "tp1.4":
        fig, axes, update = refraction_figure(experiment)
        stem = "tp1_4"
    else:
        fig, axes, update = cartesian_figure(experiment)
        stem = "tp2_1"
    seconds = (case["times"] - case["times"][0]).astype("timedelta64[s]").astype(float)
    max_step = args.max_step
    if max_step is None:
        max_step = int(np.searchsorted(seconds, 18000, side="right")-1) if config["case"] == "tp1.4" else 100
    last = min(max_step, len(seconds)-1)
    indices = np.unique(np.linspace(0, last, min(args.max_frames, last+1), dtype=int))
    heading = fig.suptitle("", fontsize=18)

    def draw(index, dpi):
        update(index)
        heading.set_text(f"Time: {seconds[index]:.2f} seconds")
        with io.BytesIO() as stream:
            fig.savefig(stream, format="png", dpi=dpi)
            stream.seek(0)
            with Image.open(stream) as frame:
                return frame.convert("RGB")

    destination = args.assets or args.output
    destination.mkdir(parents=True, exist_ok=True)
    frames = [draw(int(index), 150) for index in indices]
    durations = [100*int(b-a) for a, b in zip(indices[:-1], indices[1:])]
    durations.append(durations[-1] if durations else 100)
    frames[0].save(destination / f"{stem}.webp", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, lossless=True, method=4)
    snapshot = min(50, last)
    draw(snapshot, 300).save(destination / f"{stem}_overview.webp", lossless=True, method=4)
    record = dict(cases=[entry[0]["case"] for entry in experiments],
                  font_family="Times New Roman", font_file=font,
                  size_inches=fig.get_size_inches().tolist(), animation_dpi=150, snapshot_dpi=300,
                  displayed_steps=indices.tolist(), elapsed_seconds=seconds[indices].tolist(),
                  snapshot_step=snapshot, full_result_frames=len(seconds),
                  direction_convention="coming-from for tp1.4, matching the archived plot",
                  secondary_axes=[dict(ylabel=ax.get_ylabel(), limits=list(ax.get_ylim()))
                                  for ax in fig.axes if ax.get_ylabel() == "Depth (m)"],
                  axes=[dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel()) for ax in axes])
    (args.output / f"{stem}_visualization.json").write_text(json.dumps(record, indent=2) + "\n")
    plt.close(fig)
    print(f"Saved {stem}: {len(frames)} frames, steps 0–{last}, Times New Roman")


if __name__ == "__main__":
    main()
