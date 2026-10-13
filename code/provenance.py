"""Code fingerprints and atomic records for independently checkpointed experiments."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
ACTIVE_SOURCES = (
    'config.yaml', 'experiments/paper.yaml', 'common.py', 'protocol.py', 'metrics.py',
    'experiment_core.py', 'baselines/cp_core.py', 'baselines/uq_core.py',
    'baselines/conditional_bootstrap.py', 'run_paper.py', 'provenance.py',
)

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def execution_manifest() -> dict:
    files = {name: file_sha256(CODE_ROOT / name) for name in ACTIVE_SOURCES}
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'files': files}

def require_manifest_hash(recorded: str) -> None:
    if str(recorded) != execution_manifest()['sha256']:
        raise ValueError('配置或实验代码已改变，不能混用已有检查点；请先检查差异。')

def runtime_metadata() -> dict:
    import numpy, scipy, h5py, yaml
    return {'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'h5py': h5py.__version__, 'yaml': yaml.__version__,
            'threads': {key: os.environ.get(key, 'unset') for key in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}}

def strict_json_value(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return strict_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(strict_json_value(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
