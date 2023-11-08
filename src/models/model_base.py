import dataclasses
from dataclasses import dataclass, field
import json
import time
from typing import List, Literal

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src import utils


@dataclass
class ModelMetadata:
    # Architecture type: https://github.com/huggingface/pytorch-image-models/issues/1162
    arch_family: Literal['mlp', 'cnn', 'transformer', 'hybrid']
    arch: str
    model_id: str
    # Merged (30M) - EVA: ImageNet-22K, CC12M, CC3M, Object365, COCO (train), ADE20K (train)
    # Merged (38M) - EVA: ImageNet-22K, CC12M, CC3M, Object365, COCO (train), ADE20K (train), OpenImages
    training_data: Literal[
        'ImageNet-1k (1.2M)',
        'ImageNet-21k (14M)',
        'Merged (30M)',
        'Merged (38M)',
        'Flickr YFCC (90M)',
        'JFT (300M)',
        'WIT (400M)',
        'LAION-400M (400M)',
        'Instagram (940M)',
        'RandomInternetImages (2B)',
        'LAION-2B (2.3B)',
        'Instagram (3.6B)'
    ]
    source: Literal['torchvision.models', 'torch.hub', 'timm', 'huggingface']
    eval_batch_size: int
    num_params: int = -1
    # List of extra annotations:
    # Distillation
    # Semi-Supervised Learning
    # Self-Supervised Learning
    # Adversarial Training
    # CLIP Training
    extra_annotations: List[str] = field(default_factory=list)


class ModelBase:

    def __init__(self, model, transform_fn, classify_batch_fn, model_metadata: ModelMetadata):
        model = model.eval()
        self.model = model
        self.transform_fn = transform_fn
        self.classify_batch_fn = classify_batch_fn
        self.model_metadata = model_metadata
        self.model_metadata.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.device = utils.get_device()

    @torch.no_grad()
    def compute_accuracy(self, dataset: str):
        logger.info(f'Compute accuracy for model: {self.model_metadata.model_id}')
        torch.cuda.empty_cache()
        model = self.model.to(self.device)
        model.eval()

        dset = get_dataset(dataset=dataset, transform=self.transform_fn)
        data_loader = DataLoader(dset, batch_size=self.model_metadata.eval_batch_size, shuffle=False, num_workers=4)
        indices_to_1k = dset.get_indices_in_1k()

        num_total, num_correct, num_correct_subset = 0, 0, 0
        with tqdm(data_loader, ncols=150, leave=False) as pbar:
            for _, inputs, target, default_target in pbar:
                inputs = inputs.to(self.device)
                probs = self.classify_batch_fn(model, inputs)
                preds = probs.argmax(-1).cpu()
                assert inputs.size(0) == probs.size(0) == preds.numel() == target.numel() == default_target.numel()
                num_total += preds.numel()
                num_correct += (preds == target).sum().item()
                if len(indices_to_1k) < 1000:
                    probs_subset = probs[:, indices_to_1k]
                    preds_subset = probs_subset.argmax(-1).cpu()
                    num_correct_subset += (preds_subset == default_target).sum().item()
                pbar.set_postfix(acc=100.0 * num_correct / num_total)

        logger.info(f'correct = {num_correct}, total = {num_total}, acc = {100.0 * num_correct / num_total:.2f}')
        if len(indices_to_1k) < 1000:
            logger.info('Subset: correct = {}, total = {}, acc = {:.2f}'.format(
                num_correct_subset, num_total, 100.0 * num_correct_subset / num_total
            ))
        self.model.to('cpu')

    @torch.no_grad()
    def save_preds(self, dataset: str):
        # self.compute_accuracy(dataset)
        # return

        artefacts_dataset_path = utils.get_artefacts_path() / dataset
        if not artefacts_dataset_path.is_dir():
            artefacts_dataset_path.mkdir()
        artefacts_model_dir = utils.get_or_create_path(
            artefacts_dataset_path / self.model_metadata.model_id.replace('/', '--')
        )

        metadata_path = artefacts_model_dir / 'metadata.json'
        model_metadata = dataclasses.asdict(self.model_metadata)

        # Compute and cache predictions (if not already computed)
        samples_info_path = artefacts_dataset_path / 'samples_info.npz'
        preds_path = artefacts_model_dir / 'preds.npz'
        if preds_path.is_file():
            assert metadata_path.is_file()
            with open(metadata_path, 'r') as f:
                model_metadata_file_contents = json.load(f)
            model_metadata_file_contents.update(model_metadata)
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata_file_contents, f)
            return

        logger.info(f'Compute predictions for model: {self.model_metadata.model_id}')

        torch.cuda.empty_cache()
        model = self.model.to(self.device)
        model.eval()

        dset = get_dataset(dataset=dataset, transform=self.transform_fn)
        paths_prefix = dset.root + '/'
        data_loader = DataLoader(dset, batch_size=self.model_metadata.eval_batch_size, shuffle=False, num_workers=4)
        indices_to_1k = dset.get_indices_in_1k()

        num_total, num_correct, num_correct_subset = 0, 0, 0
        sample_files_list = []
        targets_list = []
        top1_preds_list = []
        top5_preds_list = []
        probs_list = []
        start_time = time.perf_counter()
        with tqdm(data_loader, ncols=150, leave=False) as pbar:
            for paths, inputs, targets, default_targets in pbar:
                inputs = inputs.to(self.device)
                probs = self.classify_batch_fn(model, inputs).detach()
                preds = probs.argmax(-1).cpu()
                assert inputs.size(0) == probs.size(0) == preds.numel() == targets.numel() == default_targets.numel()
                num_total += preds.numel()
                num_correct += (preds == targets).sum().item()
                if len(indices_to_1k) < 1000:
                    probs_subset = probs[:, indices_to_1k]
                    preds_subset = probs_subset.argmax(-1).cpu()
                    num_correct_subset += (preds_subset == default_targets).sum().item()
                pbar.set_postfix(acc=100.0 * num_correct / num_total)

                paths = list(paths)
                assert all(p.startswith(paths_prefix) for p in paths)
                paths = [p[len(paths_prefix):] for p in paths]
                sample_files_list.extend(paths)

                targets_list.append(targets)
                top1_preds_list.append(preds)
                top5_preds_list.append(torch.topk(probs, 5, dim=1).indices.cpu())
                probs_list.append(probs.cpu())
        elapsed_time = time.perf_counter() - start_time

        logger.info(f'correct = {num_correct}, total = {num_total}, acc = {100.0 * num_correct / num_total:.2f}')
        if len(indices_to_1k) < 1000:
            logger.info('Subset: correct = {}, total = {}, acc = {:.2f}'.format(
                num_correct_subset, num_total, 100.0 * num_correct_subset / num_total
            ))

        model_metadata['top1_val_acc'] = 100.0 * num_correct / num_total
        if len(indices_to_1k) < 1000:
            model_metadata[f'top1_val_acc_subset_{len(indices_to_1k)}'] = 100.0 * num_correct_subset / num_total
        model_metadata['preds_comp_time'] = elapsed_time

        targets = torch.cat(targets_list).numpy()
        assert len(dset) == len(sample_files_list) == len(targets) == num_total

        # Save / check samples info (files and targets)
        if not samples_info_path.is_file():
            np.savez_compressed(
                samples_info_path,
                sample_files=np.array(sample_files_list),
                targets=targets
            )
        else:
            loaded = np.load(samples_info_path)
            assert np.array_equal(np.array(sample_files_list), loaded['sample_files'])
            assert np.array_equal(targets, loaded['targets'])

        # Save model metadata
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f)

        # Save model predictions
        np.savez_compressed(
            preds_path,
            top1=torch.cat(top1_preds_list).numpy(),
            top5=torch.cat(top5_preds_list).numpy(),
            probs=torch.cat(probs_list).numpy()
        )
        logger.info(f'Predictions saved in: {preds_path}')

        # Move model back to CPU.
        self.model.to('cpu')
