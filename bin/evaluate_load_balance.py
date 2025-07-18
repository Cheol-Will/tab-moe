import argparse
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from loguru import logger

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib

DEFAULT_N_SEEDS = 1


def main(
    path: str | Path,
    n_seeds: int = DEFAULT_N_SEEDS,
    function: None | str = None,
    *,
    force: bool = False,
):
    path = Path(path).resolve()
    assert path.is_dir()


    # load repost.json
    if path.name.endswith('-tuning'):
        assert function is None
        assert (path / 'DONE').exists()
        from_tuning = True
        tuning_report = lib.load_report(path)
        function_qualname = tuning_report['config']['function'] # "bin.model.main"
        template_config = tuning_report['best']['config']
        evaluation_dir = path.with_name(path.name.replace('tuning', 'evaluation'))
        evaluation_dir.mkdir(exist_ok=True)

    elif path.name.endswith('-evaluation'):
        assert function is not None
        from_tuning = False
        function_qualname = function
        evaluation_dir = path
        template_config = lib.load_config(evaluation_dir / '0')
    
    elif path.name.endswith('-balance'):
        assert function is not None
        from_tuning = False
        function_qualname = function
        evaluation_dir = path
        template_config = lib.load_config(evaluation_dir / '0')
    
    else:
        raise ValueError(f'Bad input path: {path}')
    del path

    # train and evaluate with given configs
    # function_ = lib.import_(function_qualname)
    function_ = lib.import_("bin.model_load_balance.main")
    for seed in range(n_seeds):
        output = evaluation_dir / str(seed)
        config_path = output.with_suffix('.toml') 
        print(output)
        print(config_path)
        done = (output / 'DONE').exists()

        config = deepcopy(template_config)
        config['seed'] = seed
        if 'catboost' in function_qualname:
            if config['model']['task_type'] == 'GPU':
                config['model']['task_type'] = 'CPU'  # this is crucial for good results
                thread_count = config['model'].get('thread_count', 1)
                config['model']['thread_count'] = max(thread_count, 4)

        try:
            if (from_tuning or seed > 0) and not (config_path.exists() and done):
                lib.dump_config(output, config)
            # function_(config, output, force=force) # bin.model.main: train-eval-test
            function_(config, output, force=force) # call bin.model_load_balance.main()


        except Exception:
            if from_tuning or seed > 0:
                config_path.unlink(True)
            shutil.rmtree(output, True)
            raise


if __name__ == '__main__':
    lib.configure_libraries()

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--n_seeds', type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument('--function')
    parser.add_argument('--force', action='store_true')
    main(**vars(parser.parse_args(sys.argv[1:])))
