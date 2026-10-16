"""Original planning, subprocess execution, and completion tracking."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .aggregation import _v1_aggregate
from .io import _v1_atomic_json
from .source_builder import _v1_prepare_source
from .sources import _V1_SOURCE_CELLS


_ABLATION_VARIANTS = {
    'adaptive_ewc': ['full_v1', 'fixed_ewc', 'no_ewc'],
    'rollback': ['full_v1', 'no_rollback'],
    'episodic_replay': ['full_v1', 'uniform_replay', 'no_episodic_replay'],
}


def run_neurone_v1_ablations(configuration):
    cfg = copy.deepcopy(configuration)
    seeds = cfg['seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Declare exactly three distinct integer seeds in [0, 2**32).')
    if not cfg['cases'] or any(case not in _ABLATION_VARIANTS for case in cfg['cases']):
        raise ValueError('Cases must be adaptive_ewc, rollback, and/or episodic_replay.')
    for key in ['task_steps', 'batch_size', 'target_update_steps', 'rollback_interval',
                'after_task_episodes', 'pre_task_episodes', 'final_episodes']:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key + ' must be a positive integer.')
    if cfg['batch_size'] >= 10000:
        raise ValueError('Batch size must be below the original replay capacity of 10000.')
    if not 0 <= cfg['gamma'] <= 1 or cfg['device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Invalid gamma or device.')
    variants = list(dict.fromkeys(v for case in cfg['cases'] for v in _ABLATION_VARIANTS[case]))
    plan = [dict(variant=v, seed=s) for s in seeds for v in variants]
    print(f'{len(variants)} configurations × 3 seeds = {len(plan)} independent runs.')
    print('Cases:', ', '.join(cfg['cases']), '| Seeds:', seeds)
    if cfg['dry_run']:
        for row in plan:
            print(row['variant'], 'seed=' + str(row['seed']))
        return plan
    missing = [name for name in ['torch', 'numpy', 'pandas', 'scipy'] if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError('Install missing dependencies in this kernel: ' + ', '.join(missing))
    root = Path(cfg['output_dir']).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(json.dumps(_V1_SOURCE_CELLS, ensure_ascii=False).encode()).hexdigest()
    identity = {k: v for k, v in cfg.items() if k not in {'output_dir', 'skip_completed', 'dry_run', 'make_plots', 'continue_on_error'}}
    identity['source_sha256'] = source_hash
    manifest_path = root / 'configuration.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError('Output contains a different configuration. Declare a fresh OUTPUT_DIR.')
    _v1_atomic_json(manifest_path, identity)
    _v1_atomic_json(root / 'plan.json', plan)
    for index, row in enumerate(plan, 1):
        directory = root / 'runs' / row['variant'] / ('seed_' + str(row['seed']))
        status_file = directory / 'status.json'
        if directory.exists() and any(directory.iterdir()):
            previous = json.loads(status_file.read_text()) if status_file.exists() else {}
            if cfg['skip_completed'] and previous.get('status') == 'complete':
                print(f'[{index}/{len(plan)}] Reusing {row["variant"]} seed={row["seed"]}', flush=True)
                continue
            raise FileExistsError(f'Incomplete/existing run: {directory}. Use a fresh OUTPUT_DIR or preserve and move this run directory.')
        directory.mkdir(parents=True, exist_ok=True)
        _v1_atomic_json(directory / 'configuration.json', dict(identity, **row))
        worker = directory / 'run_v1_variant.py'
        worker.write_text(_v1_prepare_source(_V1_SOURCE_CELLS, cfg, row['variant'], row['seed']), encoding='utf-8')
        started = time.perf_counter()
        _v1_atomic_json(status_file, dict(**row, status='running'))
        print(f'[{index}/{len(plan)}] Training {row["variant"]} seed={row["seed"]}; log: {directory / "run.log"}', flush=True)
        try:
            with (directory / 'run.log').open('w', encoding='utf-8') as log:
                env = dict(os.environ, PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, '-u', str(worker)], cwd=directory, stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    returncode = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if returncode:
                raise RuntimeError(f'Worker exited with code {returncode}; see {directory / "run.log"}')
            import pandas as pd
            matrix = pd.read_csv(directory / 'checkpoints' / 'metrics.csv')
            if matrix.shape != (7, 7):
                raise RuntimeError('The original V1 run did not complete all seven task phases.')
            _v1_atomic_json(status_file, dict(**row, status='complete', elapsed_seconds=time.perf_counter()-started))
        except BaseException as error:
            _v1_atomic_json(status_file, dict(**row, status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                            elapsed_seconds=time.perf_counter()-started, error=str(error)))
            _v1_aggregate(root, plan, cfg)
            if not cfg['continue_on_error'] or isinstance(error, KeyboardInterrupt):
                raise
            print('Failed:', error, flush=True)
        _v1_aggregate(root, plan, cfg)
        print(f'[{index}/{len(plan)}] Saved results.', flush=True)
    result = _v1_aggregate(root, plan, cfg)
    print('Finished. Tables:', root / 'tables')
    return result
