"""Draw original paper panels or three compact conference display copies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator, PercentFormatter
from common import load_paper_config, calibration_conditions, uv_to_angles
from provenance import file_sha256, write_json
from run_paper import OUTPUT_ROOT, verify_stage, read_tree
from progress import TerminalProgress

SUMMARY_ROOT = OUTPUT_ROOT / "summary"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
COMPACT_FIGURE_ROOT = Path(__file__).resolve().parent.parent / "latex_v7" / "figures_compact"
COMPACT_LAYOUT = {
    "ellipse_padding_fraction": 0.05,
    "ellipse_minimum_padding_deg": 0.20,
    "ellipse_equal_aspect": True,
    "calibration_figsize_inches": [3.45, 2.25],
    "calibration_margins": {"left": 0.18, "right": 0.98, "bottom": 0.22, "top": 0.98},
    "calibration_legend_location": "lower right",
    "calibration_legend_columns": 2,
    "pdf_padding_inches": 0.04,
}
CHI2_95_DF2 = 5.991464547107979


def load_sealed_summary():
    completion = json.loads((SUMMARY_ROOT / "completion.json").read_text())
    path = SUMMARY_ROOT / "summary.json"
    if completion.get("status") != "complete" or completion["summary_sha256"] != file_sha256(path):
        raise ValueError("Summary completion marker does not match the summary")
    if completion["table_sha256"] != file_sha256(SUMMARY_ROOT / "table1.csv"):
        raise ValueError("Table completion marker does not match the table")
    for stage in ("calibration", "boundary"):
        marker = verify_stage(stage)
        if marker["artifact_sha256"] != completion["parents"][stage]:
            raise ValueError("Summary source experiment has changed")
    return json.loads(path.read_text()), completion


def load_representative_run():
    spec = load_paper_config()
    condition = spec["reporting"]["example_condition"]
    preferred = int(spec["reporting"]["example_repeat"])
    with h5py.File(OUTPUT_ROOT / "calibration/runs.h5", "r") as handle:
        cell = handle[condition]
        order = [preferred] + [i for i in range(spec["calibration"]["repeats"]) if i != preferred]
        for repeat in order:
            run = cell[f"r{repeat:04d}"]
            outer = read_tree(run["outer"])
            uq = read_tree(run["uq/actual_mask"])
            uv = outer["estimated_uv"]
            radii = np.linalg.norm(uv, axis=1)
            if not (outer["fit_converged"] and outer["correct_branch"] and uq["status"] == "valid"
                    and np.all((radii > 0) & (radii < 1))):
                continue
            representative = {"true_angles_deg": uv_to_angles(outer["true_uv"]),
                "estimated_angles_deg": uv_to_angles(uv), "estimated_uv": uv,
                "doa_covariance": uq["covariance"]}
            selection = {"condition": condition, "preferred_repeat": preferred,
                         "selected_repeat": repeat,
                         "rule": "predeclared_repeat_then_first_valid_in_fixed_order"}
            return representative, selection
    raise ValueError("No valid example is available at the predeclared condition")


def angle_covariances(estimated_uv: np.ndarray, covariance_uv: np.ndarray) -> np.ndarray:
    """Propagate (u,v) covariance to plotting coordinates (azimuth,elevation)."""
    output = np.empty_like(covariance_uv, dtype=float)
    degrees = 180.0 / np.pi
    for index, ((u_value, v_value), block) in enumerate(zip(estimated_uv, covariance_uv)):
        radius_squared = float(u_value**2 + v_value**2)
        radius = np.sqrt(radius_squared)
        if radius <= 0.0 or radius >= 1.0:
            raise ValueError("Angle Jacobian is singular at the representative estimate")
        scale = radius * np.sqrt(1.0 - radius_squared)
        theta_phi = degrees * np.asarray([
            [u_value / scale, v_value / scale],
            [-v_value / radius_squared, u_value / radius_squared],
        ])
        jacobian = theta_phi[[1, 0], :]
        propagated = jacobian @ block @ jacobian.T
        output[index] = 0.5 * (propagated + propagated.T)
    return output


def ellipse_parameters(covariance: np.ndarray) -> tuple[float, float, float]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if np.any(values <= 0.0):
        raise ValueError("Representative angle covariance is not positive definite")
    diameters = 2.0 * np.sqrt(CHI2_95_DF2 * values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    return float(diameters[0]), float(diameters[1]), float(angle)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _ellipse_panel_data(representative: dict[str, np.ndarray]) -> list[dict[str, np.ndarray | float]]:
    true_angles = representative["true_angles_deg"][:, [1, 0]]
    estimated_angles = representative["estimated_angles_deg"][:, [1, 0]]
    covariances = angle_covariances(
        representative["estimated_uv"], representative["doa_covariance"]
    )
    panels = []
    for truth, estimate, covariance in zip(true_angles, estimated_angles, covariances):
        width, height, angle = ellipse_parameters(covariance)
        panels.append({
            "truth": truth,
            "estimate": estimate,
            "width": width,
            "height": height,
            "angle": angle,
        })
    return panels


def plot_confidence_ellipses(panels: list[dict[str, np.ndarray | float]], path: Path,
                            *, compact: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(3.45, 2.80), constrained_layout=True)
    colors = ("#2f5d8a", "#b66c26", "#41806b")
    markers = ("o", "s", "^")
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    handles = []
    for index, (panel, color) in enumerate(zip(panels, colors), start=1):
        truth = np.asarray(panel["truth"])
        estimate = np.asarray(panel["estimate"])
        width, height = float(panel["width"]), float(panel["height"])
        angle = float(panel["angle"])
        radians = np.radians(angle)
        half_extent = 0.5 * np.asarray([
            np.hypot(width * np.cos(radians), height * np.sin(radians)),
            np.hypot(width * np.sin(radians), height * np.cos(radians)),
        ])
        lower = np.minimum(lower, np.minimum(truth, estimate - half_extent))
        upper = np.maximum(upper, np.maximum(truth, estimate + half_extent))
        axis.add_patch(Ellipse(estimate, width, height, angle=angle,
                             facecolor="none", edgecolor=color, linewidth=0.9, zorder=2))
        axis.plot(*truth, marker="x", color=color, markersize=5.0,
                  markeredgewidth=0.9, linestyle="none", zorder=6)
        axis.plot(*estimate, marker=markers[index - 1], color=color, markersize=3.6,
                  linestyle="none", zorder=5)
        handles.append(Line2D([], [], color=color, marker=markers[index - 1],
                              linewidth=0.9, label=f"Source {index}"))
    handles.extend([
        Line2D([], [], color="0.05", marker="x", linestyle="none", markersize=5.0,
               label="True DOA"),
    ])
    padding_fraction = COMPACT_LAYOUT["ellipse_padding_fraction"] if compact else 0.10
    minimum_padding = COMPACT_LAYOUT["ellipse_minimum_padding_deg"] if compact else 0.20
    padding = np.maximum(padding_fraction * (upper - lower), minimum_padding)
    axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axis.set_xlabel(r"Azimuth $\phi$ (°)")
    axis.set_ylabel(r"Elevation $\theta$ (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MultipleLocator(1.0))
    axis.yaxis.set_major_locator(MultipleLocator(1.0))
    axis.minorticks_off()
    axis.grid(color="0.92", linewidth=0.3)
    axis.legend(handles=handles, frameon=False, loc="upper left")
    save_options = {"pad_inches": COMPACT_LAYOUT["pdf_padding_inches"]} if compact else {}
    figure.savefig(path, format="pdf", bbox_inches="tight", **save_options)
    plt.close(figure)


def plot_calibration(summary, path, *, sweep, compact=False):
    conditions = [row for row in calibration_conditions() if row['sweep'] in (sweep, 'both')]
    variable = 'snr_db' if sweep == 'snr' else 'observation_rate'
    conditions.sort(key=lambda row: row[variable])
    x = np.asarray([row[variable] for row in conditions], float)
    figsize = COMPACT_LAYOUT["calibration_figsize_inches"] if compact else (3.45, 2.85)
    figure, axis = plt.subplots(figsize=figsize)
    if compact:
        figure.subplots_adjust(**COMPACT_LAYOUT["calibration_margins"])
    else:
        figure.subplots_adjust(left=.18, right=.98, bottom=.18, top=.80)
    styles = [('actual_mask', '#2f5d8a', 'o', '-', 'Proposed'),
              ('fitted_full_rate_scaled', '0.40', 's', '--', 'Rate-scaled'),
              ('bootstrap', '#a65f18', 'D', '-.', 'Parametric bootstrap')]
    plotted = []
    for method, color, marker, line, label in styles:
        cells = [summary['calibration']['methods'][method][row['id']] for row in conditions]
        # None means unestimable, never zero coverage or an interpolated value.
        y = np.asarray([cell['coverage'][-1] for cell in cells], float)
        intervals = np.asarray([cell['coverage95_interval'] for cell in cells], float)
        axis.plot(x, y, color=color, marker=marker, linestyle=line,
                  linewidth=.85, markersize=3.6, label=label)
        plotted.extend(y[np.isfinite(y)])
    axis.axhline(.95, color='0.55', linewidth=.7, label='Nominal 95%')
    axis.set_xlabel('SNR (dB)' if sweep == 'snr' else r'Observation rate $\rho$')
    axis.set_ylabel('Empirical coverage')
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    lower = min(plotted + [.90])
    axis.set_ylim(max(0, np.floor((lower - .02) * 20) / 20), 1)
    axis.set_xticks(x)
    axis.minorticks_off()
    axis.grid(color='0.92', linewidth=.3)
    legend_options = {"loc": COMPACT_LAYOUT["calibration_legend_location"]} if compact else {
        "loc": "upper center", "bbox_to_anchor": (.5, 1.28), "borderaxespad": 0,
    }
    legend_columns = COMPACT_LAYOUT["calibration_legend_columns"] if compact else 2
    axis.legend(frameon=False, ncol=legend_columns, fontsize=7, columnspacing=1, handlelength=1.8,
                handletextpad=.4, **legend_options)
    save_options = {"pad_inches": COMPACT_LAYOUT["pdf_padding_inches"]} if compact else {}
    figure.savefig(path, format='pdf', bbox_inches='tight', **save_options)
    plt.close(figure)


def plot_boundary(summary, path):
    rows = sorted(summary['boundary']['separations'], key=lambda row: row['separation'])
    x = np.asarray([row['separation'] for row in rows])
    def values(name):
        return np.asarray([row[name] for row in rows], float)
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True, constrained_layout=True)
    positive = x > 0
    axes[0].semilogy(x[positive], values('truth_smallest_singular_median')[positive], 'o-',
                     color='#2f5d8a', markersize=3.3, linewidth=.8, label='True model')
    axes[0].fill_between(x[positive], values('truth_smallest_singular_q25')[positive],
                          values('truth_smallest_singular_q75')[positive], color='#dbe9f6', linewidth=0)
    axes[0].semilogy(x[positive], values('estimated_smallest_singular_median')[positive], 's--',
                     color='0.4', markersize=3.1, linewidth=.8, label='Fitted model')
    zero = next((row for row in rows if row['separation'] == 0), None)
    if zero:
        value = zero['truth_smallest_singular_median']
        text = 'Coincident sources: true minimum = ' + (f'{value:.1e}' if value is not None else 'unavailable')
        axes[0].text(.03, .04, text, transform=axes[0].transAxes, fontsize=6.5)
    axes[0].set_ylabel('Smallest singular value')
    axes[0].legend(frameon=False, fontsize=7)
    for name, label, color, marker, line in [
        ('truth_regular_rate', 'True regularity', '#2f5d8a', 'o', '-'),
        ('uq_valid_rate', 'Fitted UQ valid', '#a65f18', 'D', '-.'),
        ('correct_branch_rate', 'Converged correct branch', '0.4', 's', '--')]:
        axes[1].plot(x, values(name), color=color, marker=marker, linestyle=line,
                     linewidth=.8, markersize=3, label=label)
    axes[1].set_ylabel('Run fraction')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_ylim(-.03, 1.03)
    axes[1].set_xlabel('Source separation in direction cosines')
    axes[1].legend(frameon=False, fontsize=6.5, loc='best')
    axes[1].set_xticks([0, .05, .10, .15, .20])
    for axis in axes:
        axis.grid(color='0.92', linewidth=.3)
        axis.minorticks_off()
    figure.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(figure)


def generate(*, output_dir=None, replace=False, compact=False):
    default_dir = COMPACT_FIGURE_ROOT if compact else FIGURE_ROOT
    output_dir = Path(output_dir if output_dir is not None else default_dir).resolve()
    # Compact display copies have a separate destination; original output routing is preserved.
    if compact:
        if output_dir != COMPACT_FIGURE_ROOT.resolve():
            raise ValueError('Compact figures must use latex_v7/figures_compact.')
        if replace:
            raise ValueError('Compact mode refuses replacement of existing display copies.')
    elif not output_dir.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Output must remain under data/paper_abc; paper replacement is a separate step.')
    names = ['doa_uncertainty.pdf', 'coverage_snr.pdf',
             'coverage_observation_rate.pdf']
    if not compact:
        names.append('information_boundary.pdf')
    outputs = [output_dir / name for name in names]
    marker = output_dir / ('completion.json' if compact else 'paper_figures.completion.json')
    if not replace and (marker.exists() or any(path.exists() for path in outputs)):
        if compact:
            raise FileExistsError('Compact display copies already exist; replacement is not enabled.')
        raise FileExistsError('New figures already exist; use --replace to redraw only these new outputs.')
    progress = TerminalProgress('Compact paper figures' if compact else 'Paper figures', len(outputs))
    progress.update(0, detail='stage=read_frozen_sources valid=0 failed=0', force=True)
    summary, completion = load_sealed_summary()
    representative, selection = load_representative_run()
    output_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        marker.unlink()
    _style()
    functions = [lambda p: plot_confidence_ellipses(_ellipse_panel_data(representative), p,
                                                  compact=compact),
                 lambda p: plot_calibration(summary, p, sweep='snr', compact=compact),
                 lambda p: plot_calibration(summary, p, sweep='observation_rate', compact=compact)]
    if not compact:
        functions.append(lambda p: plot_boundary(summary, p))
    for index, (function, output) in enumerate(zip(functions, outputs), 1):
        temporary = output.with_suffix('.pdf.incomplete')
        progress.update(index - 1, detail=f'stage={output.stem} valid={index-1} failed=0', force=True)
        try:
            function(temporary)
        except Exception:
            progress.update(index - 1, detail=f'stage={output.stem} valid={index-1} failed=1', force=True)
            raise
        temporary.replace(output)
        progress.update(index, detail=f'stage={output.stem} valid={index} failed=0', force=True)
    record = {'status': 'complete', 'parent_summary_sha256': completion['summary_sha256'],
        'script_sha256': file_sha256(Path(__file__)), 'representative_selection': selection,
        'figures': {path.name: file_sha256(path) for path in outputs},
        'cohort': 'ABC_common_valid', 'methods': ['Proposed', 'Rate-scaled', 'Parametric bootstrap']}
    if compact:
        record.update(layout=COMPACT_LAYOUT, source_parents=completion['parents'],
                      plotting_core_sha256=file_sha256(Path(__file__).with_name('common.py')))
    write_json(marker, record)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--compact', action='store_true',
                        help='Generate only three compact display copies in latex_v7/figures_compact.')
    arguments = parser.parse_args()
    for path in generate(output_dir=arguments.output_dir, replace=arguments.replace,
                         compact=arguments.compact):
        print(path)

if __name__ == '__main__':
    main()
