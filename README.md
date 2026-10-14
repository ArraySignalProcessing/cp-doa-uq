# cp-doa-uq

Gauge-invariant local uncertainty quantification for 2-D DOA estimation
from incomplete complex CP observations.

## Setup

Run from the repository root with Python 3.11:

```bash
python3.11 -m venv code/.venv
source code/.venv/bin/activate
python -m pip install -r code/requirements.txt
```

## Reproduce the experiments

```bash
python code/run_paper.py calibration
python code/run_paper.py boundary
python code/run_paper.py summary
python code/plot_paper.py
```

Results are saved under `data/paper_abc/`.
The scripts report progress and support checkpoint resumption.

The plotting script generates three figures under
`data/paper_abc/figures/`. Running it again updates the figures
from the existing experiment results.
