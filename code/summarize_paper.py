"""Validate completed stages and summarize paired run-level evidence."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from common import load_paper_config, calibration_conditions
from provenance import file_sha256, write_json
from progress import TerminalProgress
from run_paper import OUTPUT_ROOT, METHODS, read_tree, verify_stage


def calibration_error(coverage, levels):
    coverage, levels = np.asarray(coverage, float), np.asarray(levels, float)
    if coverage.shape != levels.shape:
        raise ValueError('Coverage and confidence levels must have identical shapes')
    return float(np.mean(np.abs(coverage - levels)))


def h2_interpretation(mean, lower, upper):
    if not np.all(np.isfinite([mean, lower, upper])):
        return 'not_estimable'
    if upper < 0:
        return 'direct_support'
    if mean < 0:
        return 'direction_consistent_evidence_insufficient'
    return 'comparative_claim_not_supported'


def interval_from_samples(samples):
    array = np.asarray(samples, float)
    array = array[np.isfinite(array)]
    return [float(np.quantile(array, .025)), float(np.quantile(array, .975))] if array.size else [np.nan, np.nan]


def run_summary(coverage, area, seconds, eligible, levels, indices):
    selected = np.flatnonzero(eligible)
    if not selected.size:
        return {'valid_runs': 0, 'coverage': [np.nan] * len(levels),
                'coverage95_interval': [np.nan, np.nan], 'calibration_error': np.nan,
                'median_area95': np.nan, 'mean_time': np.nan}
    mean = np.mean(coverage[selected], axis=0)
    resampled = []
    for draw in indices:
        accepted = draw[eligible[draw]]
        resampled.append(float(np.mean(coverage[accepted, -1])) if accepted.size else np.nan)
    return {'valid_runs': len(selected), 'coverage': mean,
            'coverage95_interval': interval_from_samples(resampled),
            'calibration_error': calibration_error(mean, levels),
            'median_area95': float(np.median(area[selected])),
            'mean_time': float(np.mean(seconds[selected]))}


def summarize():
    markers = {stage: verify_stage(stage) for stage in ('calibration', 'boundary')}
    spec = load_paper_config()
    levels = np.asarray(spec['reporting']['confidence_levels'])
    n = spec['calibration']['repeats']
    repetitions = spec['reporting']['paired_run_bootstrap_repetitions']
    rng = np.random.default_rng(spec['reporting']['summary_seed'])
    # Joint repeat resampling preserves shared latent quantities across conditions and methods.
    indices = rng.integers(0, n, size=(repetitions, n))
    conditions = calibration_conditions(spec)
    progress = TerminalProgress('Paper summary', len(conditions) + len(spec['boundary']['separations']))
    calibration = {'methods': {method: {} for method in METHODS}, 'cohort': 'ABC_common_valid',
                   'confidence_levels': levels, 'planned_runs_per_condition': n}
    paired_cells = []
    table_rows = []
    with h5py.File(OUTPUT_ROOT / 'calibration/runs.h5', 'r') as handle:
        if set(handle) != {condition['id'] for condition in conditions}:
            raise ValueError('Calibration conditions do not match the protocol')
        calibration['bootstrap_budget'] = int(handle.attrs['bootstrap_budget'])
        for ci, condition in enumerate(conditions):
            cell = handle[condition['id']]
            if set(cell) != {f'r{i:04d}' for i in range(n)}:
                raise ValueError('Calibration run count mismatch')
            coverage = np.full((n, 3, len(levels)), np.nan)
            areas, seconds = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
            valid = np.zeros((n, 3), bool)
            status_counts = [Counter() for _ in METHODS]
            fit_seconds = np.empty(n)
            for repeat in range(n):
                run = cell[f'r{repeat:04d}']
                if not bool(run.attrs.get('committed', False)):
                    raise ValueError('Uncommitted calibration run')
                outer = read_tree(run['outer'])
                fit_seconds[repeat] = outer['initialization_time'] + outer['fit_time']
                for mi, method in enumerate(METHODS):
                    uq = read_tree(run['uq'][method])
                    if not bool(uq.get('committed', False)):
                        raise ValueError('Uncommitted UQ result')
                    status_counts[mi][str(uq['status'])] += 1
                    seconds[repeat, mi] = uq['time']
                    valid[repeat, mi] = bool(outer['fit_converged'] and outer['correct_branch'] and
                                            uq['status'] == 'valid' and uq['metrics'] is not None)
                    if valid[repeat, mi]:
                        coverage[repeat, mi] = np.mean(uq['metrics']['ellipse_coverage'], axis=0)
                        areas[repeat, mi] = np.median(uq['metrics']['ellipse_area'][:, -1])
            common = np.all(valid, axis=1)
            paired_cells.append((coverage, common))
            for mi, method in enumerate(METHODS):
                row = run_summary(coverage[:, mi], areas[:, mi], seconds[:, mi], common, levels, indices)
                row['own_valid'] = run_summary(coverage[:, mi], areas[:, mi], seconds[:, mi], valid[:, mi], levels, indices)
                row.update(planned_runs=n, valid_rate=float(valid[:, mi].mean()),
                           common_valid_rate=float(common.mean()), status_counts=dict(status_counts[mi]),
                           all_planned_coverage=np.sum(np.nan_to_num(coverage[:, mi]), axis=0) / n,
                           mean_uq_time_all_planned=float(np.mean(seconds[:, mi])),
                           mean_shared_fit_time=float(np.mean(fit_seconds)),
                           mean_end_to_end_time_common=float(np.mean(fit_seconds[common] + seconds[common, mi])) if common.any() else np.nan)
                calibration['methods'][method][condition['id']] = row
                if condition['id'] in spec['reporting']['anchors']:
                    table_rows.append({'condition': condition['id'], 'method': method,
                        'planned_runs': n, 'common_valid_runs': row['valid_runs'],
                        'coverage95': row['coverage'][-1], 'coverage95_lower': row['coverage95_interval'][0],
                        'coverage95_upper': row['coverage95_interval'][1],
                        'median_area95': row['median_area95'], 'mean_uq_seconds': row['mean_time'],
                        'mean_end_to_end_seconds': row['mean_end_to_end_time_common'],
                        'own_valid_rate': row['valid_rate']})
            progress.update(ci + 1, detail=f'condition={condition["id"]} common-valid={int(common.sum())} failed={n-int(common.sum())}')
    comparisons = {}
    for other in (1, 2):
        deltas = []
        for draw in indices:
            cell_deltas = []
            for coverage, eligible in paired_cells:
                accepted = draw[eligible[draw]]
                if not accepted.size:
                    break
                means = np.mean(coverage[accepted], axis=0)
                cell_deltas.append(calibration_error(means[0], levels) - calibration_error(means[other], levels))
            deltas.append(float(np.mean(cell_deltas)) if len(cell_deltas) == len(paired_cells) else np.nan)
        observed = []
        for coverage, eligible in paired_cells:
            if eligible.any():
                means = np.mean(coverage[eligible], axis=0)
                observed.append(calibration_error(means[0], levels) - calibration_error(means[other], levels))
        delta = float(np.mean(observed)) if len(observed) == len(paired_cells) else np.nan
        lower, upper = interval_from_samples(deltas)
        comparisons[METHODS[other]] = {'delta_calibration_error': delta, 'interval95': [lower, upper],
            'valid_resamples': int(np.isfinite(deltas).sum()), 'repetitions': repetitions,
            'interpretation': h2_interpretation(delta, lower, upper),
            'note': 'A zero-containing interval does not establish equivalence.'}
    calibration['paired_comparisons'] = comparisons
    boundary_rows = []
    with h5py.File(OUTPUT_ROOT / 'boundary/runs.h5', 'r') as handle:
        if set(handle) != {f's{i:02d}' for i in range(len(spec['boundary']['separations']))}:
            raise ValueError('Boundary conditions do not match the protocol')
        for si, separation in enumerate(spec['boundary']['separations']):
            cell = handle[f's{si:02d}']
            if set(cell) != {f'r{i:04d}' for i in range(spec['boundary']['repeats'])}:
                raise ValueError('Boundary run count mismatch')
            runs = []
            for key in sorted(cell):
                if not bool(cell[key].attrs.get('committed', False)):
                    raise ValueError('Uncommitted boundary result')
                runs.append(read_tree(cell[key]))
            truth = np.asarray([r['truth_diagnostics'].get('smallest_identifiable_singular_value', np.nan) for r in runs])
            fitted = np.asarray([r['estimated_regularity_diagnostics'].get('smallest_identifiable_singular_value', np.nan) for r in runs])
            def quantile(values, q):
                finite = np.asarray(values, float)
                finite = finite[np.isfinite(finite)]
                return float(np.quantile(finite, q)) if finite.size else np.nan
            boundary_rows.append({'separation': separation, 'planned_runs': len(runs),
                'truth_smallest_singular_median': quantile(truth, .5),
                'truth_smallest_singular_q25': quantile(truth, .25),
                'truth_smallest_singular_q75': quantile(truth, .75),
                'estimated_smallest_singular_median': quantile(fitted, .5),
                'truth_regular_rate': float(np.mean([r['truth_regularity_status'] == 'valid' for r in runs])),
                'fit_converged_rate': float(np.mean([r['fit']['status'] == 'converged' for r in runs])),
                'correct_branch_rate': float(np.mean([r['fit']['status'] == 'converged' and r['correct_branch'] for r in runs])),
                'uq_valid_rate': float(np.mean([r['uq_status'] == 'valid' for r in runs])),
                'angular_rmse_deg': float(np.sqrt(np.mean([r['angular_rmse_deg'] ** 2 for r in runs]))),
                'median_area95': quantile([np.median(r['metrics']['ellipse_area'][:, -1])
                                          for r in runs if r['metrics'] is not None], .5),
                'status_counts': dict(Counter(r['uq_status'] for r in runs))})
            progress.update(len(conditions) + si + 1, detail=f'separation={separation}')
    destination = OUTPUT_ROOT / 'summary'
    destination.mkdir(parents=True, exist_ok=True)
    summary = {'calibration': calibration, 'boundary': {'separations': boundary_rows},
               'parents': {stage: marker['artifact_sha256'] for stage, marker in markers.items()},
               'statistics': 'Targets averaged within runs; shared repeat IDs resampled jointly across conditions.'}
    write_json(destination / 'summary.json', summary)
    with (destination / 'table1.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    write_json(destination / 'completion.json', {'status': 'complete',
        'summary_sha256': file_sha256(destination / 'summary.json'),
        'table_sha256': file_sha256(destination / 'table1.csv'),
        'parents': summary['parents'], 'script_sha256': file_sha256(Path(__file__))})
    return destination / 'summary.json'

if __name__ == '__main__':
    print(summarize())
