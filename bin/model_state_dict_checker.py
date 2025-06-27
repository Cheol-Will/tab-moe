import sys
from pathlib import Path
if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib
# path = "exp/moe-sparse-shared-piecewiselinear/covtype2/0-tuning/checkpoint.pt"
model = 0
path = "exp/moe-sparse-shared-piecewiselinear/churn/0-evaluation/0/"
ckpt = lib.load_checkpoint(path)
model.load_state_dict(lib.load_checkpoint(path)['model'])

for k, v  in ckpt.items():
    print(k)
    