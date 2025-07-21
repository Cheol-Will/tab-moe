import math
import statistics
from pathlib import Path
from typing import Any, Literal, Optional, Union, Type

from dataclasses import dataclass

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import rtdl_num_embeddings
import scipy
import torch.utils.tensorboard
from loguru import logger
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

import lib
import lib.data
import lib.deep
import lib.reformer
import lib.env
from lib import KWArgs, PartKey


# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<


def compute_label_bins(y: Tensor, n_bins: int):
    bins = [
        q.unique()
        for q in torch.quantile(
            y, torch.linspace(0.0, 1.0, n_bins + 1).to(y), dim=0
        )
    ]
    rtdl_num_embeddings._check_bins(bins)

    return bins



class ScaledDotProductDistance(nn.Module):
    def __init__(self, scale: float, temperature: float):
        super(ScaledDotProductDistance, self).__init__()
        self.scale = scale
        self.temperature = temperature

    def forward(self, query, key):
        # Scaled dot product
        similarities = torch.mm(query, key.T) * self.scale  # (B, N)
        distances = -similarities  
        distances = distances / self.temperature  
        return distances


class L2Distance(nn.Module):
    def __init__(self, temperature: float):
        super(L2Distance, self).__init__()
        self.temperature = temperature

    def forward(self, query, key):
        # L2 distance (Euclidean distance)
        # distances = torch.cdist(query, candidate_k, p=2)  # (B, N)
        # inplace operation -> grad error
        query_squared = torch.sum(query**2, dim=-1, keepdim=True)
        key_squared = torch.sum(key**2, dim=-1, keepdim=True)
        cross_term = torch.mm(query, key.T)

        distances = query_squared + key_squared.T - 2 * cross_term
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances)
        distances = distances / self.temperature  

        return distances


class CosineSimilarity(nn.Module):
    def __init__(
        self, 
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, key):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        similarities = torch.mm(query, key.T) # (B, N)
        distances = -similarities  
        distances = distances / self.temperature  
        return distances        


class DistanceMetric(nn.Module):
    def __init__(
        self,
        scale: None | float = None,
        temperature: float = 0.2,
        distance_name: str = 'sdp',
    ):
        super(DistanceMetric, self).__init__()

        assert distance_name in ['sdp', 'l2', 'cossim']

        if distance_name == 'sdp':
            assert scale is not None
            print(f"Initialize ScaledDotProductDistance with scale={scale} and temperature={temperature}")
            self.distance = ScaledDotProductDistance(scale=scale, temperature=temperature)
        elif distance_name == 'l2':
            print(f"Initialize L2Distance with temperature={temperature}")
            self.distance = L2Distance(temperature=temperature)
        elif distance_name == 'cossim':
            print(f"Initialize CosineSimilarity with temperature={temperature}")
            self.distance = CosineSimilarity(temperature=temperature)

    def forward(self, query, key):
        return self.distance(query, key)


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        bins: None | list[Tensor],
        #
        num_embeddings: None | dict = None,
        momentum: float, # momentum
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        # predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['dropout0']],
        dropout0: float,
        dropout1: Union[bool, Literal['dropout0']],
        normalization: str, # BatchNorm or LayerNorm
        activation: str,
        queue_size: int, 
        is_classification: bool,
        use_mlp_head: bool = False,
        distance_metric: str = 'sdp',
        temperature: float = 0.2,
        candidate_encoding_batch_size: None | int = None,
    ):
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        
        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features

        else:
            if bins is None:
                self.num_module = lib.deep.make_module(
                    **num_embeddings, n_features=n_num_features
                )
            else:
                assert num_embeddings['type'].startswith('PiecewiseLinearEmbeddings')
                self.num_module = lib.deep.make_module(**num_embeddings, bins=bins)

            # Adjust d_num for piecewise linear embedding and periodic embedding.    
            d_num = n_num_features * num_embeddings['d_embedding']
        
        # >>> Categorical features
        self.cat_module = (
            lib.deep.OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        d_cat = sum(cat_cardinalities)      

        # >>> Embedding
        d_in = d_num + d_cat # input dimension after numerical embedidng and one hot encoding 
        self.encoder_query = lib.reformer.BaseEncoder(d_in, d_main, d_multiplier, dropout0, encoder_n_blocks, skip_connection=False)
        self.encoder_key = lib.reformer.BaseEncoder(d_in, d_main, d_multiplier, dropout0, encoder_n_blocks, skip_connection=False)

        for param_q, param_k in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(queue_size, d_main))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))    

        self.distance = DistanceMetric(d_main ** -0.5, temperature, distance_metric)

        d_out = 1 if n_classes is None else n_classes
        self.transformation = nn.Linear(d_main, d_main) # key transformation
        if n_classes is None:
            # Naive linear embeddign for regression
            self.label_encoder = nn.Linear(1, d_main) 
        else:
            # Embedding lookup table for classification
            self.label_encoder = nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2)) # (B, 1, d_main) -> (B, d_main)
            )

        if use_mlp_head:
            self.head = nn.Sequential(*[
                nn.Linear(d_main, d_main),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(d_main, d_out),
            ])
        else:
            self.head = nn.Linear(d_main, d_out)

        self.d_out = d_out
        self.n_classes = n_classes
        self.scale = d_main ** -0.5
        self.momentum = momentum
        self.queue_size = queue_size
        self.search_index = None
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.temperature = temperature
        self.distance_metric = distance_metric
        self.reset_parameters()

    def reset_parameters(self):
        pass
    def _encode(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        """
        Feature processing including one-hot encoding and numerical embedding.        
        """
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x]) # (B, F)

        return x
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # print(f"Current batch size: {batch_size}")
        # print(f"Queue pointer: {ptr}")

        if ptr + batch_size > self.queue_size:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:self.queue_size, :]  = keys[:self.queue_size - ptr, :]
            self.queue[0:overflow, :] = keys[self.queue_size - ptr:, :]
        else:
            self.queue[ptr:ptr + batch_size, :] = keys
            
        ptr = (ptr + batch_size) % self.queue_size # adjust pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def calculate_key(
        self, 
        x_num: None | Tensor = None, 
        x_cat: None | Tensor = None, 
        y: None | Tensor = None,
    ):
        """
        Before evaluation, calculate keys for whole train dataset.
        """
        x = self._encode(x_num, x_cat)
        x = self.encoder_key(x)
        x = x + self.label_encoder(y.unsqueeze(-1))
       
        return x

    def forward(
        self,
        *,
        x_num: None | Tensor = None, 
        x_cat: None | Tensor = None,
        y: None | Tensor = None, 
        candidate_k: Tensor,
        candidate_y: Tensor,
        is_train: bool,
    ) -> Tensor:
        """
        candidate_k: pre-computed keys for evalaution.
        candidate_y: y label of trainset for evalaution.
        """
        x = self._encode(x_num, x_cat) # preprocessing
        query = self.encoder_query(x)
       
        if is_train:
            with torch.no_grad():
                key_x = self.encoder_key(x)
                context_y = self.label_encoder(y.unsqueeze(-1))
                key_x = key_x + context_y
        else:
            key_x = None

        if self.training:
            # During trainig, use memory queue.
            assert candidate_k is None
            assert candidate_y is None
            candidate_k = self.queue
        else:
            # During evaluation, use pre-computed candidates set.
            if is_train:
                # During eval on train set, add itself and remove later. 
                assert y is not None
                candidate_k = torch.cat([key_x, candidate_k])
        distances = self.distance(query, candidate_k) # scaled dot product or L2 distance
        if is_train and not self.training:
            # During eval on train set, remove the label of training index. 
            distances = distances.fill_diagonal_(torch.inf)  
        
        weights = F.softmax(-distances, dim=-1) # (B, N)        

        # Add context information to query.
        context = self.transformation(candidate_k)
        context = torch.mm(weights, context)
        query = query + context
        query = self.head(query) 

        if self.training:
            self._dequeue_and_enqueue(key_x) # enqueue current batch
            
        if query.ndim == 2:
            query = query.unsqueeze(1) # 
        return query
  

class Config(TypedDict):
    seed: int
    data: KWArgs
    bins: NotRequired[KWArgs]
    label_bins: NotRequired[int]
    model: KWArgs
    head_selection: NotRequired[bool]
    optimizer: KWArgs
    n_lr_warmup_epochs: NotRequired[int]
    batch_size: int
    eval_batch_size: NotRequired[int]
    patience: int
    n_epochs: int
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
    amp: NotRequired[bool]  # torch.autocast
    compile: NotRequired[bool]  # torch.compile

DEFAULT_SHARE_TRAINING_BATCHES = True

def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | lib.JSONDict:
    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(output, force=force):
        return None

    lib.print_config(config)  # type: ignore[code]
    delu.random.seed(config['seed'])
    device = lib.get_device()
    report = lib.create_report(main, config)

    # >>> Data
    batch_size = config['batch_size']
    dataset = lib.data.build_dataset(**config['data'])
    if dataset.task.is_regression:
        print("Standardize labels for regression task")
        dataset.data['y'], regression_label_stats = lib.data.standardize_labels(
            dataset.data['y']
        )
        print(dataset.data['y'])
    else:
        regression_label_stats = None

    # Convert binary features to categorical features.
    if dataset.n_bin_features > 0:
        x_bin = dataset.data.pop('x_bin')
        # Remove binary features with just one unique value in the training set.
        # This must be done, otherwise, the script will fail on one specific dataset
        # from the "why" benchmark.
        n_bin_features = x_bin['train'].shape[1]
        good_bin_idx = [
            i for i in range(n_bin_features) if len(np.unique(x_bin['train'][:, i])) > 1
        ]
        if len(good_bin_idx) < n_bin_features:
            x_bin = {k: v[:, good_bin_idx] for k, v in x_bin.items()}

        if dataset.n_cat_features == 0:
            dataset.data['x_cat'] = {
                part: np.zeros((dataset.size(part), 0), dtype=np.int64)
                for part in x_bin
            }
        for part in x_bin:
            dataset.data['x_cat'][part] = np.column_stack(
                [dataset.data['x_cat'][part], x_bin[part].astype(np.int64)]
            )
        del x_bin
    dataset = dataset.to_torch(device)
    Y_train = dataset.data['y']['train'].to(
        torch.long if dataset.task.is_classification else torch.float
    )
    # >>> Model
    if 'bins' in config:
        # Compute the bins for PiecewiseLinearEncoding and PiecewiseLinearEmbeddings.
        compute_bins_kwargs = (
            {
                'y': Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                'regression': dataset.task.is_regression,
                'verbose': True,
            }
            if 'tree_kwargs' in config['bins']
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset.data['x_num']['train'], **config['bins'], **compute_bins_kwargs
        )
        logger.info(f'Bin counts: {[len(x) - 1 for x in bin_edges]}')
    else:
        bin_edges = None

    # Compute quantiles for label embedding if regressison task.
    if 'label_bins' in config:
        if dataset.task.is_regression:
            label_bins = compute_label_bins(dataset.data['y']['train'], n_bins=config['label_bins'])
        else:
            print("Skip computing labels bins since current taks is classification.")
        
    # branching for custom model
    meta_data = {
        "n_num_features": dataset.n_num_features,
        "cat_cardinalities": dataset.compute_cat_cardinalities(),
        "n_classes": dataset.task.try_compute_n_classes(),
        "bins": bin_edges,
    }

    model_args = config['model'].copy()
    model_args.pop('share_training_batches', None)
    queue_ratio = model_args.pop('queue_ratio', None)
    queue_size_ = min(dataset.size('train'), queue_ratio * batch_size)
    queue_size = (queue_size_ // batch_size) * batch_size
    model_args['queue_size'] = queue_size
    print(f"Init qtabformerv2 with {model_args}" )
    model = Model(
        **meta_data,
        **model_args,
        is_classification=dataset.task.is_classification
    )
    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'
    model.to(device)
    if lib.is_dataparallel_available():
        model = nn.DataParallel(model)

    # >>> Training
    step = 0
    report['epoch_size'] = epoch_size = math.ceil(dataset.size('train') / batch_size)
    eval_batch_size = config.get(
        'eval_batch_size',
        # With torch.compile,
        # the largest possible evaluation batch size is noticeably smaller.
        2048 if config.get('compile', False) else 32768,
    )
    chunk_size = None  # Currently, not used.
    share_training_batches = config['model'].get(
        'share_training_batches', DEFAULT_SHARE_TRAINING_BATCHES
    )

    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    gradient_clipping_norm = config.get('gradient_clipping_norm')
    _loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
        # else nn.functional.nll_loss
    )

    # Keep the train_indices for Retrieval-based Model
    train_size = dataset.size('train')
    train_indices = torch.arange(train_size, device=device)
    # eval_candidate_key = torch.randn(train_size, config['model']['d_main'])

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return _loss_fn(
            y_pred.flatten(0, 1),
            (
                y_true.repeat_interleave(y_pred.shape[1])
                if share_training_batches
                else y_true
            ),
        )

    # The following generator is used only for creating training batches,
    # so the random seed fully determines the sequence of training objects.
    batch_generator = torch.Generator(device).manual_seed(config['seed'])
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')
    parameter_statistics = config.get('parameter_statistics', config['seed'] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    # Only bfloat16 was tested as amp_dtype.
    # However, float16 is supported as a fallback.
    # To enable float16, uncomment the two lines below.
    amp_dtype = (
        torch.bfloat16
        if config.get('amp', False)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        # else torch.float16
        # if config.get('amp', False) and and torch.cuda.is_available()
        else None
    )
    amp_enabled = amp_dtype is not None
    # For FP16, the gradient scaler must be used.
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore[code]
    logger.info(f'AMP enabled: {amp_enabled}')

    if config.get('compile', False):
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument,
        # because it caused issues with training.
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    def get_Xy(part: str, idx: Tensor):
        x_num = dataset.data['x_num'][part][idx] if 'x_num' in dataset.data else None
        x_cat = dataset.data['x_cat'][part][idx] if 'x_cat' in dataset.data else None
        y = dataset.data['y'][part][idx]
        y = y.to(
            torch.long if dataset.task.is_classification else torch.float
        )

        return x_num, x_cat, y 

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def freeze_key():
        candidate_k = []
        candidate_y = []
        for idx in torch.arange(dataset.size("train"), device=device).split(batch_size):
            x_num, x_cat, y = get_Xy('train', idx)
            key = model.calculate_key(x_num, x_cat, y)
            # print(key.shape)
            candidate_k.append(key)
            candidate_y.append(y)

        candidate_k = torch.cat(candidate_k)
        candidate_y = torch.cat(candidate_y)
        
        return candidate_k, candidate_y

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(
        part: PartKey, 
        idx: Tensor, 
        training: bool, 
        candidate_k: Tensor = None,
        candidate_y: Tensor = None,
    ) -> Tensor:
        """
        Note that candidate_k is pre-computed and passed during evaluation.
        """
        if not training:
            assert candidate_k is not None
            assert candidate_y is not None

        x_num, x_cat, y = get_Xy(part, idx)
        is_train = part == 'train'

        if training:
            # During training, use queue.
            cand_k = None
            cand_y = None
        else:
            candidate_idx = train_indices
            if is_train:
                # eval on train
                candidate_idx = train_indices[~torch.isin(train_indices, idx)]
            cand_k = candidate_k[candidate_idx]
            cand_y = candidate_y[candidate_idx]

        pred = model(
            x_num=x_num,
            x_cat=x_cat,
            y=y if is_train else None,
            candidate_k=cand_k,
            candidate_y=cand_y,
            is_train=is_train,
        )
        return pred.squeeze(-1).float()


    @evaluation_mode()
    def evaluate(
        parts: list[PartKey], eval_batch_size: int
    ) -> tuple[
        dict[PartKey, Any], dict[PartKey, np.ndarray], dict[PartKey, np.ndarray], int
    ]:
        model.eval()
        # Before evaluation, calculate key
        # 
        candidate_k, candidate_y = freeze_key()

        head_predictions: dict[PartKey, np.ndarray] = {}
        for part in parts:
            while eval_batch_size:
                try:
                    head_predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False, candidate_k, candidate_y)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        if dataset.task.is_regression:
            assert regression_label_stats is not None
            head_predictions = {
                k: v * regression_label_stats.std + regression_label_stats.mean
                for k, v in head_predictions.items()
            }
        else:
            head_predictions = {
                k: scipy.special.softmax(v, axis=-1)
                for k, v in head_predictions.items()
            }
            if dataset.task.is_binclass:
                head_predictions = {k: v[..., 1] for k, v in head_predictions.items()}

        predictions = {k: v.mean(1) for k, v in head_predictions.items()}
        metrics = (
            dataset.task.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': lib.WORST_SCORE} for x in predictions}
        )
        return metrics, predictions, head_predictions, eval_batch_size

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'batch_generator': batch_generator.get_state(),
                'random_state': delu.random.get_state(),
                'early_stopping': early_stopping,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            }
            | (
                {} if grad_scaler is None else {'grad_scaler': grad_scaler.state_dict()}
            ),
        )
        lib.dump_report(output, report)
        lib.backup_output(output)

    print()
    timer.run()
    while config['n_epochs'] == -1 or step // epoch_size < config['n_epochs']:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        batches = (
            torch.randperm(
                dataset.size('train'),
                generator=batch_generator,
                device=device,
            ).split(batch_size)
            if share_training_batches
            else [
                x.transpose(0, 1).flatten()
                for x in torch.rand(
                    (config['model']['k'], dataset.size('train')),
                    generator=batch_generator,
                    device=device,
                )
                .argsort(dim=1)
                .split(batch_size, dim=1)
            ]
        )
        for batch_idx in tqdm(batches, desc=f'Epoch {step // epoch_size} Step {step}'):
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx, True), Y_train[batch_idx])

            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

            if parameter_statistics and (
                step % epoch_size == 0  # The first batch of the epoch.
                or step // epoch_size == 0  # The first epoch.
            ):
                for k, v in lib.deep.compute_parameter_stats(model).items():
                    writer.add_scalars(k, v, step, timer.elapsed())
                    del k, v

            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            model._momentum_update_key_encoder()
            step += 1
            epoch_losses.append(loss.detach())

        epoch_losses = torch.stack(epoch_losses).tolist()
        mean_loss = statistics.mean(epoch_losses)
        metrics, predictions, _, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )

        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer.elapsed()}
        )
        lib.print_metrics(mean_loss, metrics)
        writer.add_scalars('loss', {'train': mean_loss}, step, timer.elapsed())
        for part in metrics:
            writer.add_scalars(
                'score', {part: metrics[part]['score']}, step, timer.elapsed()
            )

        if (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        ):
            print('🌸 New best epoch! 🌸')
            report['best_step'] = step
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(output, predictions)

        early_stopping.update(metrics['val']['score'])
        if early_stopping.should_stop() or not lib.are_valid_predictions(predictions):
            break

        print()
    report['time'] = str(timer)

    # >>>
    if lib.get_checkpoint_path(output).exists():
        model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, head_predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    save_checkpoint()

    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run(main)
