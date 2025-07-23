# Deep models on Tabular Data

# Setup environment
With Conda:

```
conda create -f environment.yaml -n tabm
conda activate tabm
```

# Running the code

`bin/go.py` performs hyperparameter tuning, evaluation with best hyperparameter on 15 different seeds, and reports ensemble result via bin/tune.py, bin/evaluate.py, and bin/ensemble.py, respectively.

> Our main focus is the average metric from best hyperparameter on 15 different seeds (result of bin/evaluate.py)

You can run bin/go.py with the specified model in arch_list on data_list (6 dataset) as follows.

```
bash _run_go_subset.sh
```


For example, If the specified model is "qtabformer-query-4-key-k-value-ky-mqa" and current dataset is adult, then it finds below 0-tuning.toml

```
exp/
  <qtabformer-query-4-key-k-value-ky-mqa>/
    <adult>/       # Or why/<dataset> or tabred/<dataset>
      0-tuning.toml  # The hyperparameter tuning config
      0-tuning/      # The result of the hyperparameter tuning
      0-evaluation/  # The evaluation under multiple random seeds
```

If the dataset is one of why or tabred, then it finds one more depth.

```
exp/
  <qtabformer-query-4-key-k-value-ky-mqa>/
    <why>/       # Or why/<dataset> or tabred/<dataset>
      <classif-num-medium-0-credit>/
        0-tuning.toml  # The hyperparameter tuning config
        0-tuning/      # The result of the hyperparameter tuning
        0-evaluation/  # The evaluation under multiple random seeds
```
With 0-tuning.toml, hyperparameter tuning is performed. 

To setup the 0-tuning.toml for all dataset, set your experiment directory in _copy_toml.sh, and run _copy_toml.sh

From src_type, it will copy all necessary configs to dest_type. Configs include model, data path, hyperparameters. However, you need to modify your experiement's hyperparameters.
```
src_type="qtabformer-query-4-key-k-value-ky-mqa"
dest_type="qtabformer-exp1"
```

```
bash _copy_toml.sh
```

In 0-tuning.toml, the model and its script code is specified as follows.

```
function = "bin.qtabformer.main"
```

You can choose to tune or not as follows:

```
momentum = 0.999
d_main = [
    "_tune_",
    "int-power-of-two",
     6,
     9,
]
num_heads = [
  "_tune_",
  "int",
  4,
 8,
 4,
]
dropout0 = [
    "_tune_",
    "?uniform",
    0.0,
    0.0,
    0.6,
]
```
- momentum: fixed to 0.999.
- d_main: sampled from [64, 128, 256, 512].
- num_heads: is sampled from [4, 8], three arguments are min, max, and step, respectively.
If step is not given, it is set to 1 in default.
- dropout0: sampled from [0.0, 0.6]. when "?uniform" is used, it can choose NO Dropout (eqaul to 0.0 (first argument is default)).


## Experiment Result
To load the experiment result and rank statistics, add your model in exp_list in summary.py, and run summary.py.
It shows all metrics of models specified in exp_list and benchmarks and rank statistics. 

```
exp_list = [
  ...,
  YOUR_MODEL_NAME,
]
```

```
python summary.py
```

The outputs as as follows:
```
          OnlineNewsPopularity ↓ churn ↑ credit ↑ ecom-offers ↑ medical_charges ↓ sberbank-housing ↓
method                                                                                                                                
CatBoost                0.8532  0.8582   0.7734        0.5596            0.0816             0.2482
Excel-plugins           0.8605  0.8618   0.7724        0.5759            0.0817             0.2533
FT-T                    0.8629  0.8593   0.7745        0.5775            0.0814             0.2440
LightGBM                0.8546  0.8600   0.7686        0.5758            0.0820             0.2468
MLP                     0.8643  0.8553   0.7735        0.5989            0.0816             0.2529
MLP-piecewiselinear     0.8585  0.8580   0.7758        0.5949            0.0812             0.2383
MNCA                    0.8651  0.8595   0.7739        0.5765            0.0811             0.2593
MNCA-periodic           0.8647  0.8606   0.7734        0.5758            0.0809             0.2448
SAINT                   0.8600  0.8603   0.7739        0.5812            0.0814             0.2467
YOUR_MODEL              0.8600  0.8603   0.7739        0.5812            0.0814             0.2467
...

method                                              RANK
tabm-piecewiselinear                            1.833333
tabm-mini-piecewiselinear                       2.166667
tabm                                            2.833333
MLP-piecewiselinear                             2.833333
CatBoost                                        3.500000
XGBoost                                         3.666667
LightGBM                                        3.833333
MNCA-periodic                                   3.833333
TabR-periodic                                   4.000000
SAINT                                           4.166667
FT-T                                            4.166667
T2G                                             4.166667
YOUR_MODEL                                      4.333333
```

## Code overview

| Code              | Comment                                                   |
| :---------------- | :-------------------------------------------------------- |
| `bin/model.py`    | **The implementation of TabM** and the training pipeline  |
| `bin/tune.py`     | Hyperparameter tuning                                     |
| `bin/evaluate.py` | Evaluating a model under multiple random seeds            |
| `bin/ensemble.py` | Evaluate an ensemble of models                            |
| `bin/go.py`       | `bin/tune.py` + `bin/evaluate.py` + `bin/ensemble.py`     |
| `lib`             | Common utilities used by the scripts in `bin`             |
| `exp`             | Hyperparameters and metrics of the models on all datasets |
| `tools`           | Additional technical tools                                |

The `exp` directory is structured as follows:

```
exp/
  <model>/
    <dataset>/       # Or why/<dataset> or tabred/<dataset>
      0-tuning.toml  # The hyperparameter tuning config
      0-tuning/      # The result of the hyperparameter tuning
      0-evaluation/  # The evaluation under multiple random seeds
```

<details>
<summary>Show</summary>

- `bin/model.py` takes one TOML config as the input and produces a directory next to the config as the output.
  For example, the command `python bin/model.py exp/hello/world.toml` will produce the directory `exp/hello/world`.
  The `report.json` file in the output directory is the main result of the run:
  it contains all metrics and hyperparameters.
- The same applies to `bin/tune.py`.
- Some scripts support the `--continue` flag to continue the execution of an interrupted run.
- Some scripts support the `--force` flag to **overwrite the existing result**
  and run the script from scratch.
- The layout in the `exp` directory can be arbitrary;
  the current layout is just our convention.

</details>


## Hyperparameter tuning

Use `bin/tune.py` to tune hyperparameters for `bin/model.py` or `bin/qtabformer.py`.
For example, the following commands reproduce the hyperparameter runing of TabM on the California Housing dataset
(this takes around one hour on NVIDIA A100):

```
mkdir -p exp/reproduce/tabm/california
cp exp/tabm/california/0-tuning.toml exp/reproduce/tabm/california
python bin/tune.py exp/reproduce/tabm/california/0-tuning.toml --continue
```

## Evaluation

Use `bin/evaluate.py` to train a model under multiple random seeds.
For example, the following command evaluates the tuned TabM from the previous section:

```
python bin/evaluate.py exp/reproduce/tabm/california/0-tuning
```

To evaluate a manually composed config for `bin/model.py`,
create a directory with a name ending with `-evaluation`,
and put the config with the name `0.toml` in it.
Then, pass the directory as the argument to `bin/evaluate.py`.
For example:

```
# The config is stored at exp/<any/path>/0-evaluation/0.toml
python bin/evaluate.py exp/<any/path>/0-evaluation --function "bin.model.main"
```

## Automating all of the above

Use `bin/go.py` to run hyperparameter tuning, evaluation and ensembling with a single command.
For example, all the above steps can be implemented as follows:

# How to cite

```
@inproceedings{gorishniy2024tabm,
    title={{TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling}},
    author={Yury Gorishniy and Akim Kotelnikov and Artem Babenko},
    booktitle={ICLR},
    year={2025},
}
```
