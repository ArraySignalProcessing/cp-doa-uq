"""Single stage-based entry point for the new paper experiments (user-run only)."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import time
import h5py
import numpy as np

from common import (DATA_ROOT, load_config, load_paper_config, calibration_conditions,
                    cp_tensor)
from protocol import draw_trial_latents, fixed_scene, materialize_condition
from experiment_core import evaluate_outer_instance, evaluate_boundary_run, _uq_result
from baselines.uq_core import (gauge_restricted_doa_covariance,
                               fitted_full_rate_scaled_doa_covariance, UQContractError)
from baselines.conditional_bootstrap import extend_prefix, prefix_covariance
from metrics import ellipse_diagnostics
from provenance import (execution_manifest, require_manifest_hash, runtime_metadata,
                        file_sha256, write_json)
from progress import TerminalProgress

OUTPUT_ROOT = DATA_ROOT / 'paper_abc'
METHODS = ('actual_mask', 'fitted_full_rate_scaled', 'bootstrap')


def write_tree(group, values):
    for name, value in values.items():
        if name == 'initializations':
            continue
        if isinstance(value, dict):
            write_tree(group.create_group(name), value)
        elif value is None:
            group.attrs[name] = '<none>'
        elif np.isscalar(value):
            group.attrs[name] = value
        else:
            group.create_dataset(name, data=np.asarray(value))


def read_tree(group):
    values = {}
    for name, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        values[name] = None if isinstance(value, str) and value == '<none>' else value
    for name, value in group.items():
        values[name] = read_tree(value) if isinstance(value, h5py.Group) else value[...]
    return values


@contextmanager
def stage_file(stage):
    directory = OUTPUT_ROOT / stage
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'running.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'{lock} 已存在；确认没有同阶段进程后再手动移除该锁。')
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        with h5py.File(directory / 'runs.h5', 'a') as handle:
            manifest = execution_manifest()
            if 'manifest' in handle.attrs:
                require_manifest_hash(handle.attrs['manifest'])
                recorded_runtime = json.loads(handle.attrs['runtime'])
                if recorded_runtime != runtime_metadata():
                    raise ValueError('运行环境或线程设置改变，不能继续同一正式结果。')
            else:
                handle.attrs['manifest'] = manifest['sha256']
                handle.attrs['manifest_json'] = json.dumps(manifest)
                handle.attrs['runtime'] = json.dumps(runtime_metadata())
                handle.attrs['config'] = json.dumps(load_paper_config())
                handle.attrs['elapsed_seconds'] = 0.0
            marker = directory / 'completion.json'
            if marker.exists():
                marker.unlink()
            started = time.perf_counter()
            try:
                yield handle
            finally:
                handle.attrs['elapsed_seconds'] = float(handle.attrs['elapsed_seconds']) + time.perf_counter() - started
                handle.flush()
    finally:
        lock.unlink(missing_ok=True)


def commit_result(parent, name, result):
    # A partial outer fit is recomputed, never mistaken for a committed fit.
    if name in parent:
        del parent[name]
    group = parent.create_group(name)
    write_tree(group, result)
    group.file.flush()
    group.attrs['committed'] = True
    group.file.flush()
    return group


def cached(parent, name):
    return name in parent and bool(parent[name].attrs.get('committed', False))


def seal(stage, expected):
    path = OUTPUT_ROOT / stage / 'runs.h5'
    write_json(OUTPUT_ROOT / stage / 'completion.json', {
        'status': 'complete', 'stage': stage, 'expected_runs': expected,
        'artifact_sha256': file_sha256(path), 'manifest': execution_manifest(),
        'completed_at_utc': datetime.now(timezone.utc).isoformat()})


def verify_stage(stage):
    directory = OUTPUT_ROOT / stage
    if (directory / 'running.lock').exists():
        raise ValueError(f'{stage} 正在运行，不能读取为完成结果')
    marker = json.loads((directory / 'completion.json').read_text())
    if marker['status'] != 'complete' or marker['artifact_sha256'] != file_sha256(directory / 'runs.h5'):
        raise ValueError(f'{stage} 未完成或文件与完成标记不一致')
    require_manifest_hash(marker['manifest']['sha256'])
    return marker


def calibration_inputs(spec, condition, repeat):
    section = spec['calibration']
    shape, rank = tuple(spec['shape']), int(spec['rank'])
    latents = draw_trial_latents(shape, rank, spec['powers'], master_seed=section['seed'],
                                 scene_index=0, repeat_index=repeat)
    scene = fixed_scene(shape, spec['geometry']['fixed_angles_deg'],
                        maximum_spatial_factor_condition=spec['geometry']['maximum_spatial_factor_condition'],
                        master_seed=section['seed'])
    clean = cp_tensor(scene['ax'], scene['ay'], latents['signals'])
    full, mask, _ = materialize_condition(clean, latents['standard_noise'], latents['mask_uniform'],
                                         observation_rate=condition['observation_rate'],
                                         snr_db=condition['snr_db'])
    return latents, scene, clean, np.where(mask, full, 0.0), mask


def outer_fit(spec, global_config, condition, inputs):
    latents, scene, clean, _, _ = inputs
    return evaluate_outer_instance(clean_tensor=clean, standard_noise=latents['standard_noise'],
        mask_uniform=latents['mask_uniform'], true_uv=scene['true_uv'], condition=condition,
        initialization_seed=latents['initialization_seed'], rank=spec['rank'],
        optimizer_config=global_config['optimizer'], method_config=global_config['methods']['proposed'],
        confidence_levels=spec['reporting']['confidence_levels'],
        correct_branch_max_uv_error=spec['reporting']['correct_branch_max_uv_error'],
        )


def bootstrap_result(prefix, outer, budget, spec):
    samples, valid = prefix['samples'][...], prefix['valid'][...]
    refit_seconds = float(np.sum(prefix['seconds'][:budget]))
    setup_seconds = float(prefix.attrs.get('setup_seconds', 0.0))
    started = time.perf_counter()
    covariance, status = prefix_covariance(samples, valid, budget,
                                spec['calibration']['bootstrap']['minimum_valid_fraction'])
    uq_seconds = refit_seconds + setup_seconds + time.perf_counter() - started
    metrics = None
    if covariance is not None:
        covariance = covariance[np.asarray(outer['target_permutation'], dtype=int)]
        try:
            metrics = ellipse_diagnostics(outer['estimated_uv'] - outer['true_uv'], covariance,
                                           spec['reporting']['confidence_levels'])
        except UQContractError as error:
            covariance, status = None, error.status
    return {'covariance': covariance, 'status': status, 'metrics': metrics,
            'time': uq_seconds, 'requested': budget, 'valid_refits': int(np.sum(valid[:budget])),
            'failure_reason': '' if status == 'valid' else status}


def run_calibration():
    if (OUTPUT_ROOT / 'calibration/completion.json').exists():
        verify_stage('calibration')
        print('Calibration 已完成，复用原结果。')
        return
    spec, global_config = load_paper_config(), load_config()['global']
    budget = int(spec['calibration']['bootstrap']['repetitions'])
    if budget < 2:
        raise ValueError('Bootstrap repetitions must be at least two')
    conditions = calibration_conditions(spec)
    total = len(conditions) * spec['calibration']['repeats']
    with stage_file('calibration') as handle:
        if 'bootstrap_budget' in handle.attrs and int(handle.attrs['bootstrap_budget']) != budget:
            raise ValueError('Bootstrap 预算改变，不能继续同一正式实验')
        handle.attrs['bootstrap_budget'] = budget
        done = sum(bool(r.attrs.get('committed', False)) for c in handle.values() for r in c.values())
        progress = TerminalProgress('Calibration ABC', total, initial_completed=done,
                                     prior_elapsed=float(handle.attrs['elapsed_seconds']))
        prior_runs = [r for c in handle.values() for r in c.values() if bool(r.attrs.get('committed', False))]
        valid = sum(bool(r['outer'].attrs['fit_converged']) and bool(r['outer'].attrs['correct_branch'])
                    and all(str(r['uq'][m].attrs['status']) == 'valid' for m in METHODS) for r in prior_runs)
        completed, failed = done, done - valid
        progress.update(done, detail=f"stage=shared-fit-ABC common-valid={valid} failed={failed}", force=True)
        for ci, condition in enumerate(conditions):
            cell = handle.require_group(condition['id'])
            cell.attrs['parameters'] = json.dumps(condition)
            for repeat in range(spec['calibration']['repeats']):
                key = f'r{repeat:04d}'
                if cached(cell, key):
                    continue
                run = cell.require_group(key)
                inputs = calibration_inputs(spec, condition, repeat)
                if not cached(run, 'outer'):
                    commit_result(run, 'outer', outer_fit(spec, global_config, condition, inputs))
                outer = read_tree(run['outer'])
                uq = run.require_group('uq')
                order = METHODS[repeat % 3:] + METHODS[:repeat % 3]
                for method in order:
                    if cached(uq, method):
                        continue
                    if not outer['fit_converged']:
                        value = {'status': 'fit_not_converged', 'failure_reason': 'Outer fit failed',
                                 'covariance': None, 'metrics': None, 'time': 0.0}
                    elif method == 'bootstrap':
                        prefix = run.require_group('bootstrap_prefix')
                        extend_prefix(prefix, factors=outer['fit']['factors'], observed=inputs[3], mask=inputs[4],
                            optimizer=global_config['optimizer'], fit_converged=True, budget=budget,
                            seed=spec['calibration']['bootstrap_seed'], condition_index=ci, repeat_index=repeat,
                            progress=progress, outer_completed=completed,
                            detail=f'condition={ci+1}/{len(conditions)} repeat={repeat+1}/{spec["calibration"]["repeats"]}')
                        value = bootstrap_result(prefix, outer, budget, spec)
                    else:
                        function = gauge_restricted_doa_covariance if method == 'actual_mask' else fitted_full_rate_scaled_doa_covariance
                        value = _uq_result(function, outer['fit']['factors'], inputs[3], inputs[4],
                                           global_config['methods']['proposed'])
                        value['metrics'] = None
                        if value['covariance'] is not None:
                            value['covariance'] = value['covariance'][np.asarray(outer['target_permutation'], dtype=int)]
                            try:
                                value['metrics'] = ellipse_diagnostics(outer['estimated_uv'] - outer['true_uv'],
                                    value['covariance'], spec['reporting']['confidence_levels'])
                            except UQContractError as error:
                                value.update(status=error.status, failure_reason=str(error), covariance=None)
                    commit_result(uq, method, value)
                run.file.flush()
                run.attrs['committed'] = True
                run.file.flush()
                common_valid = bool(outer['correct_branch'] and outer['fit_converged'] and
                                    all(str(uq[m].attrs['status']) == 'valid' for m in METHODS))
                valid += int(common_valid)
                failed += int(not common_valid)
                completed += 1
                progress.update(completed, detail=f'common-valid={valid} failed={failed}')
        progress.update(total, detail=f'common-valid={valid} failed={failed}', force=True)
    seal('calibration', total)


def run_boundary():
    if (OUTPUT_ROOT / 'boundary/completion.json').exists():
        verify_stage('boundary')
        print('Boundary 已完成，复用原结果。')
        return
    spec, global_config = load_paper_config(), load_config()['global']
    boundary = spec['boundary']
    total = len(boundary['separations']) * boundary['repeats']
    with stage_file('boundary') as handle:
        done = sum(bool(r.attrs.get('committed', False)) for c in handle.values() for r in c.values())
        progress = TerminalProgress('Regularity boundary', total, initial_completed=done,
                                     prior_elapsed=float(handle.attrs['elapsed_seconds']))
        valid = sum(str(r.attrs['uq_status']) == 'valid' for c in handle.values() for r in c.values()
                    if bool(r.attrs.get('committed', False)))
        completed, failed = done, done - valid
        progress.update(done, detail=f"stage=true-and-fitted-information valid={valid} failed={failed}", force=True)
        for si, separation in enumerate(boundary['separations']):
            cell = handle.require_group(f's{si:02d}')
            cell.attrs['separation'] = separation
            for repeat in range(boundary['repeats']):
                key = f'r{repeat:04d}'
                if cached(cell, key):
                    continue
                latent = draw_trial_latents(tuple(spec['shape']), spec['rank'], spec['powers'],
                                             master_seed=boundary['seed'], scene_index=0, repeat_index=repeat)
                result = evaluate_boundary_run(separation=separation, geometry=boundary['geometry'],
                    signals=latent['signals'], standard_noise=latent['standard_noise'], mask_uniform=latent['mask_uniform'],
                    initialization_seed=latent['initialization_seed'], shape=tuple(spec['shape']), rank=spec['rank'],
                    powers=spec['powers'], observation_rate=boundary['observation_rate'], snr_db=boundary['snr_db'],
                    noise_reference_power=boundary['noise_reference_power'], optimizer_config=global_config['optimizer'],
                    method_config=global_config['methods']['proposed'], confidence_levels=spec['reporting']['confidence_levels'],
                    correct_branch_max_uv_error=spec['reporting']['correct_branch_max_uv_error'])
                commit_result(cell, key, result)
                completed += 1
                valid += int(result['uq_status'] == 'valid')
                failed += int(result['uq_status'] != 'valid')
                progress.update(completed, detail=f'separation={si+1}/9 repeat={repeat+1}/{boundary["repeats"]} valid={valid} failed={failed}')
        progress.update(total, force=True)
    seal('boundary', total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibration', 'boundary', 'summary', 'plot'])
    args = parser.parse_args()
    if args.stage == 'calibration':
        run_calibration()
    elif args.stage == 'boundary':
        run_boundary()
    elif args.stage == 'summary':
        from summarize_paper import summarize
        print(summarize())
    else:
        from plot_paper import generate
        for path in generate():
            print(path)

if __name__ == '__main__':
    main()
