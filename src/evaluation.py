import argparse
import collections
from enum import Enum
import json
import os
import statistics
from typing import Dict, List, Optional, Set, Union

import clip
from loguru import logger
import pandas as pd
from PIL import Image
import tensorflow_datasets as tfds
import torch
from tqdm import tqdm

from src import datasets, utils, utils_clip

tqdm.pandas(ncols=150, leave=False)

ERROR_ANALYSIS_DF_FILE_NAME = 'errors_analysis.pkl'
MODEL_METADATA_FILE_NAME = 'metadata.json'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluation / error analysis pipeline')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'imagenet-a'],
                        help="Dataset on which to run the evaluation")
    parser.add_argument('--perform_error_analysis', action='store_true', default=False,
                        help='Perform error analysis: run the pipeline and save the error types in a df for each model')
    parser.add_argument('--collect_results', action='store_true', default=False,
                        help='Collect and organize the results for all models on the non-problematic samples')
    parser.add_argument('--count_spurious_correlation_pairs', action='store_true', default=False,
                        help="Count the spurious correlations pairs that appear as a mistake and found by our method")
    parser.add_argument('--list_most_common_errors', action='store_true', default=False,
                        help="List the samples that occur the most often as mistakes")
    parser.add_argument('--error_type', type=str, help="For which error type to list the most common error samples",
                        choices=[
                            'overlapping_classes',
                            'multi_label',
                            'fg',
                            'fg-OOV',
                            'non-proto',
                            'spurious_correlation',
                            'model_failure'
                        ])
    return parser.parse_args()


class ErrorClsTypes(str, Enum):
    correct = 'correct'

    # Error types relevant to problematic samples only:
    correct_but_wrong_label = 'correct (top-1=target) but target is wrong/unclear'
    empty_multi_label_correct = 'empty (problematic) multi-label, correct (top-1=target)'
    empty_multi_label_correct_collapsed_mapping = 'empty (problematic) multi-label, correct wrt collapsed mappings'
    empty_multi_label_wrong = 'empty (problematic) multi label, wrong (top-1 != target)'
    OOV_similar_to_target = 'fine-grained/OOV, same superset as target, but target not in multi-label'
    common_co_occurrences_with_target = \
        'common co-occurrences with target (spurious correlations), but target not in multi-label'

    correct_collapsed_mapping = 'correct (collapsed mapping)'
    correct_multi_label = 'correct wrt multi label (ambiguous/multi-object)'
    same_superclass = 'same superclass (fine-grained)'
    OOV_detected_by_clip = 'potentially OOV, detected by CLIP'
    common_co_occurrences = 'common co-occurrences (spurious correlations)'
    non_prototypical = 'non-prototypical (DoughBagel)'
    not_classified = 'error not classified'

    @staticmethod
    def filter_error_type(df: pd.DataFrame, error_type):
        return df.apply(lambda row: error_type in row.error_types, axis=1)


class EvalManager:

    def __init__(self, dataset: str = 'imagenet'):
        logger.info('Initialize EvalManager')
        self.dataset = dataset

        # Target/class id to class descriptions:
        self.target_desc = utils.load_labels_map()

        # Initial DataFrame:
        self.init_df = utils.create_initial_samples_df(self.target_desc, dataset=self.dataset)
        self.init_df = utils_clip.add_clip_visual_similarity_info_to_df(self.init_df, dataset=self.dataset)
        self.init_df = utils.add_multi_labels_to_df(self.init_df, self.target_desc, dataset=self.dataset)

        # WordNet id and target/class id mappings:
        # Treat all the ImageNet target labels as in-vocabulary
        self.present_wnet_ids: List[str] = sorted(self.init_df['wnet_id'].unique().tolist())
        with open(utils.get_artefacts_path() / 'imagenet_wnet_id_to_target.json', 'r') as class_to_idx_file:
            self.imagenet_wnet_id_to_target = json.load(class_to_idx_file)
        assert len(self.imagenet_wnet_id_to_target) == utils.NUM_CLASSES
        self.imagenet_wnet_ids = sorted(list(self.imagenet_wnet_id_to_target))
        self.in_voc_wnet_offsets: Set[int] = {int(wnet_id[1:]) for wnet_id in self.imagenet_wnet_ids}

        # Filter allowed labels (targets) per group:
        self.allowed_labels_per_group = {
            'all': set(),
            'organism': set(),
            # 'organism_minus_dog': set(),
            # 'dog': set(),
            # 'food': set(),
            'artifact': set(),
            'other': set()  # all - organism - artifact
        }
        for target, wnet_id in enumerate(self.imagenet_wnet_ids):
            synsets_superset = utils.get_synsets_superset(wnet_id)

            for group in self.allowed_labels_per_group:
                if group == 'all':
                    self.allowed_labels_per_group[group].add(target)
                elif group == 'organism_minus_dog':
                    if utils.lemma_names_in_synsets_superset(['organism', 'dog'], synsets_superset) == [True, False]:
                        self.allowed_labels_per_group[group].add(target)
                elif group == 'other':
                    if not any(utils.lemma_names_in_synsets_superset(['organism', 'artifact'], synsets_superset)):
                        self.allowed_labels_per_group[group].add(target)
                else:
                    if all(utils.lemma_names_in_synsets_superset([group], synsets_superset)):
                        self.allowed_labels_per_group[group].add(target)

        for group, allowed_labels in self.allowed_labels_per_group.items():
            logger.info('Group {}: # of ImageNet allowed labels = {}, # of allowed labels in {} dataset = {}'.format(
                group, len(allowed_labels), self.dataset,
                sum((self.imagenet_wnet_id_to_target[wnet_id] in allowed_labels) for wnet_id in self.present_wnet_ids)
            ))

        self.collapsed_mappings = utils.dough_bagel_collapsed_class_definitions()

        logger.info('=== Organize original ImageNet classes in superclasses ===')
        self.class_id_to_supercls, self.wnet_id_to_supercls = utils.define_superclasses(
            self.imagenet_wnet_id_to_target, self.target_desc
        )
        for c in range(utils.NUM_CLASSES):
            wnet_id = self.imagenet_wnet_ids[c]
            if not self.class_id_to_supercls[c]:
                self.class_id_to_supercls[c] = {wnet_id}
                self.wnet_id_to_supercls[wnet_id] = {wnet_id}
            assert self.class_id_to_supercls[c] == self.wnet_id_to_supercls[wnet_id]
        self.supercls_to_class_ids: Dict[str, Set[int]] = collections.defaultdict(set)
        for class_id, supercls_set in self.class_id_to_supercls.items():
            for supercls in supercls_set:
                self.supercls_to_class_ids[supercls].add(class_id)
        superclass_sizes = []
        num_singleton_superclasses = 0
        for class_ids in self.supercls_to_class_ids.values():
            class_ids_len = len(class_ids)
            assert class_ids_len > 0
            if class_ids_len == 1:
                num_singleton_superclasses += 1
            else:
                superclass_sizes.append(class_ids_len)
        logger.info(f'Number of     singleton superclasses: {num_singleton_superclasses}')
        logger.info(f'Number of non-singleton superclasses: {len(superclass_sizes)}')
        logger.info(f'Min    superclass size: {min(superclass_sizes)}')
        logger.info(f'Max    superclass size: {max(superclass_sizes)}')
        logger.info(f'Median superclass size: {statistics.median(superclass_sizes)}')
        logger.info(f'Avg    superclass size: {statistics.mean(superclass_sizes)}')
        logger.info(f'Number of pairs in a same superclass: {self.same_superclass_pairs_count()}')
        logger.info('DONE')

        self.train_files, _, _ = utils_clip.get_train_clip_embeddings()
        self.init_df['clip_top10_train_files'] = self.init_df['clip_top10_train_indices'].map(
            lambda topk_train_indices: [self.train_files[idx] for idx in topk_train_indices]
        )
        self.init_df['clip_top10_supercls'] = self.init_df['clip_top10_train_targets'].map(
            lambda topk_targets: [supercls for t in topk_targets for supercls in self.class_id_to_supercls[t]]
        )
        self.device, self.clip_model, self.clip_preprocess = utils_clip.get_clip_model()

        logger.info('=== Identify common co-occurence pairs ===')
        self.common_co_occurrence_pairs_set = utils.common_co_occurrences(self.class_id_to_supercls)
        logger.info(f'DONE: {len(self.common_co_occurrence_pairs_set)} pairs in total. (a,b) and (b,a) both counted')

        self.non_prototypical_samples = utils.get_dough_bagel_non_prototypical_samples()
        logger.info(f'=== DoughBagel non-prototypical samples: {len(self.non_prototypical_samples)} in total ===')

    def get_description(self, by: Union[int, str]) -> str:
        if isinstance(by, int):
            assert 0 <= by < utils.NUM_CLASSES
            return self.target_desc[by]
        elif isinstance(by, str):
            assert by in self.imagenet_wnet_id_to_target
            target = self.imagenet_wnet_id_to_target[by]
            assert 0 <= target < utils.NUM_CLASSES
            return self.target_desc[target]
        else:
            raise TypeError('Expected an integer or a string.')

    def prepare_and_eval_model(self, model_dir_id: str, classify_errors: bool = True) -> pd.DataFrame:
        df = self.init_df.copy()
        # 1. Add model predictions to the dataframe
        df = utils.add_model_predictions_to_df(df, model_dir_id, dataset=self.dataset, labels_map=self.target_desc)
        # 2. Define/filter the evaluation set: imagenet2012_multilabel
        df = df[df['has_defined_multi_labels']]
        df = df.drop(columns=['has_defined_multi_labels'])
        if classify_errors:
            # 3. Classify the errors
            df['error_types'] = df.progress_apply(lambda row: self.classify_error(row), axis=1)
        return df

    def debug_oov(self, file_name: str, model_dir_id: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        if df is None:
            assert model_dir_id is not None
            error_analysis_file = utils.get_artefacts_path() / self.dataset / model_dir_id / ERROR_ANALYSIS_DF_FILE_NAME
            if error_analysis_file.is_file():
                df = pd.read_pickle(error_analysis_file)
            else:
                df = self.prepare_and_eval_model(model_dir_id)

        df = df[~df['is_problematic']]
        df = df[df['file_name'] == file_name]
        # df = df[ErrorClsTypes.filter_error_type(df, ErrorClsTypes.OOV_detected_by_clip)]

        print(f'Filtered: {df.shape[0]} rows, debugging only the first one')
        print()
        for _, row in df.iterrows():
            self.classify_error(row, debug_oov=True)
            break

    def same_superclass_pairs_count(self) -> int:
        return sum(
            self.labels_in_same_superclass(label1, label2)
            for label1 in range(utils.NUM_CLASSES) for label2 in range(label1 + 1, utils.NUM_CLASSES)
        )

    def labels_in_same_superclass(self, label1: int, label2: int) -> bool:
        return not self.class_id_to_supercls[label1].isdisjoint(self.class_id_to_supercls[label2])

    def common_co_occurrence_labels(self, label1: int, label2: int) -> bool:
        return (label1, label2) in self.common_co_occurrence_pairs_set

    def classify_error(self, row, debug_oov: bool = False) -> List[str]:
        error_types = []
        if not row.multi_label:
            # row.multi_label == []
            assert row.is_problematic
            if row.target == row.top1:
                error_types.append(ErrorClsTypes.empty_multi_label_correct.value)
            elif row.top1 in self.collapsed_mappings.get(row.target, []):
                error_types.append(ErrorClsTypes.empty_multi_label_correct_collapsed_mapping.value)
            else:
                error_types.append(ErrorClsTypes.empty_multi_label_wrong.value)
        else:
            if row.target == row.top1:
                if row.target in row.multi_label:
                    error_types.append(ErrorClsTypes.correct.value)
                    return error_types
                else:
                    assert row.is_problematic
                    error_types.append(ErrorClsTypes.correct_but_wrong_label.value)
            else:
                if row.top1 in self.collapsed_mappings.get(row.target, []):
                    error_types.append(ErrorClsTypes.correct_collapsed_mapping.value)
                    return error_types
                if row.top1 in row.multi_label:
                    error_types.append(ErrorClsTypes.correct_multi_label.value)
                    return error_types

        same_superclass = [
            str(c) for c in row.multi_label if self.labels_in_same_superclass(c, row.top1)
        ]

        if same_superclass:
            error_types.append(ErrorClsTypes.same_superclass.value)
            error_types.append(f'[{",".join(list(self.class_id_to_supercls[row.top1]))}]')
            error_types.append(','.join(same_superclass))
            return error_types

        # Relevant to the problematic samples only.
        if row.target not in row.multi_label:
            assert row.is_problematic
            if row.target == row.top1 or self.labels_in_same_superclass(row.target, row.top1):
                error_types.append(ErrorClsTypes.OOV_similar_to_target.value)
                error_types.append(f'[{",".join(list(self.class_id_to_supercls[row.top1]))}]')
                return error_types
            elif self.common_co_occurrence_labels(row.target, row.top1):
                error_types.append(ErrorClsTypes.common_co_occurrences_with_target.value)
                return error_types

        if debug_oov:
            print(f'img_rel_path: {row.img_rel_path}')
            print(f'target: {row.target}, multi_label: {row.multi_label}, multi_desc: {row.multi_desc}')
            print(f'top1: {row.top1}, top1_desc: {row.top1_desc}')
            print(row.error_types)
            print()
            print('clip_top10_train_files:')
            print('\n'.join(row.clip_top10_train_files))
            print()
            print('clip_top10_supercls:', row.clip_top10_supercls)
            print('top-1 prediction superclasses:', self.class_id_to_supercls[row.top1])
            print('Overlap: {} / {}'.format(
                sum(row.clip_top10_supercls.count(supercls) for supercls in self.class_id_to_supercls[row.top1]),
                len(row.clip_top10_supercls)
            ))
            print()

        if sum(row.clip_top10_supercls.count(supercls) for supercls in self.class_id_to_supercls[row.top1]) >= 1:
            if debug_oov: print('Enter Fine-Grained OOV analysis\n')

            proposals = utils_clip.get_wnet_proposals(
                self.imagenet_wnet_ids[row.top1], row.wnet_id, self.wnet_id_to_supercls, debug_oov
            )

            TEMPLATE = 'A photo of a {}'
            text_prompts = [TEMPLATE.format(desc) for desc, _ in proposals]

            image = self.clip_preprocess(
                Image.open(os.path.join(datasets.get_dataset_root(dataset=self.dataset), row.wnet_id, row.file_name))
            ).unsqueeze(0).to(self.device)
            text = clip.tokenize(text_prompts).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text)

            # target_prompts_cnt = sum((proposal[1] == int(self.wnet_ids[row.target][1:])) for proposal in proposals)

            # mean = text_features[target_prompts_cnt:].mean(dim=0, keepdim=True)
            # image_features = image_features - mean
            # text_features = text_features - mean

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # cos_sim = (image_features @ text_features.T)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(dim=0)

            # Check if the most similar according to CLIP is out of ImageNet vocabulary:
            proposal_idx = torch.argmax(similarity).item()
            _, top5_proposal_idx = torch.topk(similarity, min(5, len(proposals)))
            top5_proposal_idx = top5_proposal_idx.cpu().tolist()
            top5_prop_summary = [
                (proposals[idx][1], proposals[idx][0], f'{100.0 * similarity[idx]:.2f}%')
                for idx in top5_proposal_idx
            ]
            if proposals[proposal_idx][1] not in self.in_voc_wnet_offsets:
                error_types.append(ErrorClsTypes.OOV_detected_by_clip.value)
                error_types.append(f'Top-5 alternative proposals: {top5_prop_summary} | out of {len(proposals)}')
                if debug_oov: print('Error classified as fine-grained OOV')
                return error_types
            elif debug_oov:
                print(top5_prop_summary)
                print(f'{proposals[proposal_idx][1]} is in-vocabulary, hence error is not OOV')
        elif debug_oov:
            print('No superclass overlap')

        if debug_oov: print('Fine-grained OOV rejected')

        common_co_occurences = [
            str(c) for c in row.multi_label if self.common_co_occurrence_labels(c, row.top1)
        ]
        if common_co_occurences:
            error_types.append(ErrorClsTypes.common_co_occurrences.value)
            error_types.append(','.join(common_co_occurences))
            return error_types

        if row.file_name in self.non_prototypical_samples:
            error_types.append(ErrorClsTypes.non_prototypical.value)
            return error_types

        error_types.append(ErrorClsTypes.not_classified.value)
        return error_types


ARCH_TO_ARCH_GROUP = {
    # ResNet
    'resnet': 'ResNet',
    'resnext': 'ResNet',
    'wide_resnet': 'ResNet',
    'ecaresnet': 'ResNet',
    'tresnet': 'ResNet',
    'seresnet': 'ResNet',
    'lambda_resnet': 'ResNet',
    'skresnet': 'ResNet',
    'seresnext': 'ResNet',
    'res2net': 'ResNet',
    'resnest': 'ResNet',
    'halonet': 'ResNet',

    # RegNet
    'regnet': 'RegNet',

    # ConvNeXt
    'convnext': 'ConvNeXt',

    # EfficientNet
    'efficientnet': 'EfficientNet',
    'mixnet': 'EfficientNet',
    'mnasnet': 'EfficientNet',
    'fbnet': 'EfficientNet',
    'mobilenet': 'EfficientNet',
    'hardcorenas': 'EfficientNet',
    'lcnet': 'EfficientNet',
    'tinynet': 'EfficientNet',
    'shufflenet': 'EfficientNet',
    'squeezenet': 'EfficientNet',
    'rexnet': 'EfficientNet',

    # ViT
    'vit': 'ViT',
    'deit': 'ViT',
    'cait': 'ViT',
    'coat': 'ViT',
    'flexivit': 'ViT',
    'crossvit': 'ViT',
    'twins': 'ViT',
    'volo': 'ViT',
    'convit': 'ViT',
    'davit': 'ViT',
    'nest': 'ViT',
    'mvit': 'ViT',
    'gcvit': 'ViT',
    'pit': 'ViT',

    # Other notable ViT models
    'eva': 'EVA',
    'beit': 'BEIT',
    'maxvit': 'MaxViT',
    'coatnet': 'MaxViT',
    'xcit': 'XCiT',
    'swin': 'Swin',

    # Metaformer (mlp, cnn and hybrid)
    'caformer': 'MetaFormer',
    'convformer': 'MetaFormer',
    'poolformer': 'MetaFormer',

    # Various CNNs
    'alexnet': 'CNN',
    'densenet': 'CNN',
    'googlenet': 'CNN',
    'inception': 'CNN',
    'vgg': 'CNN',
    'hrnet': 'CNN',
    'dla': 'CNN',
    'dpn': 'CNN',
    'xception': 'CNN',
    'nfnet': 'CNN',
    'focalnet': 'CNN',
    'darknet': 'CNN',
    'convmixer': 'CNN',

    # Various MLPs
    'resmlp': 'MLP',
    'mlp-mixer': 'MLP',
    'gmlp': 'MLP',
    'gmixer': 'MLP',

    # Various Hybrids
    'mobilevit': 'Hybrid',
    'levit': 'Hybrid',
    'pvt': 'Hybrid',
    'efficientformer': 'Hybrid',
    'edgenext': 'Hybrid',
}


def compute_mla(df: pd.DataFrame, method: str = 'no_error') -> float:
    # Adapted from: https://www.tensorflow.org/datasets/catalog/imagenet2012_multilabel
    assert method in ['no_error', 'minor_error', 'explainable_error']

    num_correct_per_class = {}
    num_images_per_class = {}

    for _, row in df.iterrows():
        # The label of the image in ImageNet
        cur_class = row.target

        # If we haven't processed this class yet, set the counters to 0
        if cur_class not in num_correct_per_class:
            assert cur_class not in num_images_per_class
            num_correct_per_class[cur_class] = 0
            num_images_per_class[cur_class] = 0

        num_images_per_class[cur_class] += 1

        # We count a prediction as correct if it is marked as correct or unclear
        # (i.e., we are lenient with the unclear labels)
        # if cur_pred in row.multi_label:
        error_types_correct = [
            ErrorClsTypes.correct,
            ErrorClsTypes.correct_collapsed_mapping,
            ErrorClsTypes.correct_multi_label
        ]
        count_as_correct = False
        if any(error_type in row.error_types for error_type in error_types_correct):
            count_as_correct = True

        if method in ['minor_error', 'explainable_error']:
            if ErrorClsTypes.same_superclass in row.error_types or \
                    ErrorClsTypes.OOV_detected_by_clip in row.error_types:
                assert not count_as_correct
                count_as_correct = True

        if method == 'explainable_error':
            if ErrorClsTypes.non_prototypical in row.error_types or \
                    ErrorClsTypes.common_co_occurrences in row.error_types:
                assert not count_as_correct
                count_as_correct = True

        num_correct_per_class[cur_class] += count_as_correct

    assert len(num_correct_per_class) == len(num_images_per_class)
    num_classes = len(num_correct_per_class)

    # Compute the per-class accuracies and then average them
    final_avg = 0
    for cid in num_correct_per_class:
        assert cid in num_correct_per_class
        assert cid in num_images_per_class
        final_avg += num_correct_per_class[cid] / num_images_per_class[cid]
    final_avg /= num_classes
    return 100.0 * final_avg


def load_imagenet_m() -> Dict[str, List[int]]:
    imagenet_m_dict = {}
    ds = tfds.load('imagenet2012_multilabel', split='imagenet_m')
    for example in ds:
        file_name = example['file_name'].numpy().decode("utf-8")
        assert file_name not in imagenet_m_dict
        imagenet_m_dict[file_name] = \
            example['correct_multi_labels'].numpy().tolist() + example['unclear_multi_labels'].numpy().tolist()
    return imagenet_m_dict


if __name__ == '__main__':
    utils.set_reproducibility()
    args = get_args()

    model_dir_ids = utils.list_model_dir_ids(dataset=args.dataset)
    imagenet_m_dict = load_imagenet_m()  # Include in evaluation only if args.dataset == 'imagenet'

    eval_manager = EvalManager(dataset=args.dataset)

    if args.perform_error_analysis:
        for i, model_dir_id in enumerate(model_dir_ids):
            logger.info(f'[{i + 1:>3}/{len(model_dir_ids)}] Analyse {model_dir_id}')
            df = eval_manager.prepare_and_eval_model(model_dir_id)
            df.to_pickle(utils.get_artefacts_path() / args.dataset / model_dir_id / ERROR_ANALYSIS_DF_FILE_NAME)

    if args.collect_results:
        # Non-problematic samples statistics:
        non_problematic_stats = {k: [] for k in eval_manager.allowed_labels_per_group}

        for model_dir_id in tqdm(model_dir_ids, ncols=150, leave=False):
            error_analysis_file = utils.get_artefacts_path() / args.dataset / model_dir_id / ERROR_ANALYSIS_DF_FILE_NAME
            if not error_analysis_file.is_file():
                continue

            with open(utils.get_artefacts_path() / args.dataset / model_dir_id / MODEL_METADATA_FILE_NAME, 'r') as f:
                model_metadata = json.load(f)
            with open(utils.get_artefacts_path() / 'imagenet' / model_dir_id / MODEL_METADATA_FILE_NAME, 'r') as f:
                imagenet_model_metadata = json.load(f)
                imagenet_top1_val_acc = imagenet_model_metadata['top1_val_acc']
                imagenet_mla_val_acc = imagenet_model_metadata['multi_label_val_acc']

            val_acc_dict = {k: v for k, v in model_metadata.items() if k.startswith('top1_val_acc')}
            val_acc_dict['imagenet_top1_val_acc'] = imagenet_top1_val_acc
            val_acc_dict['imagenet_mla_val_acc'] = imagenet_mla_val_acc

            model_df = pd.read_pickle(error_analysis_file)
            model_df = model_df[~model_df['is_problematic']]
            mla_all = compute_mla(model_df)
            mla_all_minor_error = compute_mla(model_df, method='minor_error')
            mla_all_explainable_error = compute_mla(model_df, method='explainable_error')

            num_all_top1_errors = (model_df['top1'] != model_df['target']).sum()

            for k in non_problematic_stats:
                # Each sample goes into the group if *any* of its multi-labels is an allowed label for that group.
                # => a given sample may fall in more than one group.
                df = model_df[
                    model_df['multi_label'].map(
                        lambda multi_label: any(l in eval_manager.allowed_labels_per_group[k] for l in multi_label)
                    )
                ]
                mla_group = compute_mla(df)
                mla_group_minor_error = compute_mla(df, method='minor_error')
                mla_group_explainable_error = compute_mla(df, method='explainable_error')

                imagenet_m_df = df[df['file_name'].isin(imagenet_m_dict)]
                for _, row in imagenet_m_df.iterrows():
                    assert row.multi_label == imagenet_m_dict[row.file_name]
                assert df.apply(lambda row: row.target in row.multi_label, axis=1).all()

                # Model type / metadata:
                results_summary = {
                    'model_dir_id': model_dir_id,
                    'num_params': model_metadata['num_params'],
                    'training_data': model_metadata['training_data'],
                    'arch': model_metadata['arch'],
                    'arch_family': model_metadata['arch_family'],
                    'arch_group': ARCH_TO_ARCH_GROUP[model_metadata['arch']],
                    'extra_annotations': model_metadata['extra_annotations'],
                }

                # top1 / multi-label validation accuracies (incl. on original ImageNet)
                results_summary.update(val_acc_dict)

                results_summary.update({
                    'num_all_samples': model_df.shape[0],
                    'num_group_samples': df.shape[0],
                    'num_group_correct': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.correct).sum(),
                    'num_all_top1_errors': num_all_top1_errors,
                    'num_group_top1_errors': (df['top1'] != df['target']).sum(),

                    'mla_all': mla_all,
                    'mla_all_minor_error': mla_all_minor_error,
                    'mla_all_explainable_error': mla_all_explainable_error,
                    'mla_group': mla_group,
                    'mla_group_minor_error': mla_group_minor_error,
                    'mla_group_explainable_error': mla_group_explainable_error,

                    # To be plotted:
                    'num_collapsed_mappings': \
                        ErrorClsTypes.filter_error_type(df, ErrorClsTypes.correct_collapsed_mapping).sum(),
                    'num_ambiguous': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.correct_multi_label).sum(),
                    'num_same_superclass': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.same_superclass).sum(),
                    'num_common_co_occ': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.common_co_occurrences).sum(),
                    'num_non_prototypical': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.non_prototypical).sum(),
                    'num_OOV_clip': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.OOV_detected_by_clip).sum(),
                    'num_not_classified': ErrorClsTypes.filter_error_type(df, ErrorClsTypes.not_classified).sum(),
                })

                assert results_summary['num_group_samples'] == \
                    results_summary['num_group_correct'] + results_summary['num_group_top1_errors']

                assert (results_summary['num_group_top1_errors'] ==
                        results_summary['num_collapsed_mappings'] +
                        results_summary['num_ambiguous'] +
                        results_summary['num_same_superclass'] +
                        results_summary['num_common_co_occ'] +
                        results_summary['num_non_prototypical'] +
                        results_summary['num_OOV_clip'] +
                        results_summary['num_not_classified'])

                # ImageNet-M stats
                if args.dataset == 'imagenet':
                    results_summary.update({
                        'num_all_imagenet-m_samples': 68,
                        'num_group_imagenet-m_samples': imagenet_m_df.shape[0],
                        'num_imagenet-m_errors': \
                            imagenet_m_df.apply(lambda row: row.top1 not in row.multi_label, axis=1).sum()
                    })

                non_problematic_stats[k].append(results_summary)

        results_dir = utils.get_or_create_path(utils.get_root_path() / f'stats-{args.dataset}')
        for k, s in non_problematic_stats.items():
            results_df = pd.DataFrame.from_records(s)
            results_df.to_csv(results_dir / f'non_problematic_stats_{k}.csv', index=False)

    if args.count_spurious_correlation_pairs:
        appearing_spur_corr_pairs = set()
        example_images = collections.defaultdict(int)
        for model_dir_id in tqdm(model_dir_ids, ncols=150, leave=False):
            model_df = pd.read_pickle(
                utils.get_artefacts_path() / args.dataset / model_dir_id / ERROR_ANALYSIS_DF_FILE_NAME
            )
            model_df = model_df[~model_df['is_problematic']]
            model_df = model_df[ErrorClsTypes.filter_error_type(model_df, ErrorClsTypes.common_co_occurrences)]
            for _, row in model_df.iterrows():
                for c in row.multi_label:
                    if eval_manager.common_co_occurrence_labels(c, row.top1):
                        appearing_spur_corr_pairs.add((min(c, row.top1), max(c, row.top1)))
                        example_images[(row.img_rel_path, row.target, row.top1, c)] += 1
        print('Spurious correlations pairs that appear as a mistake:', len(appearing_spur_corr_pairs))

    if args.list_most_common_errors:
        error_type_mapping = {
            'overlapping_classes': ErrorClsTypes.correct_collapsed_mapping,
            'multi_label': ErrorClsTypes.correct_multi_label,
            'fg': ErrorClsTypes.same_superclass,
            'fg-OOV': ErrorClsTypes.OOV_detected_by_clip,
            'non-proto': ErrorClsTypes.non_prototypical,
            'spurious_correlation': ErrorClsTypes.common_co_occurrences,
            'model_failure': ErrorClsTypes.not_classified
        }
        img_to_multi_labels = {}
        img_pred_to_error_types = {}
        example_images = collections.defaultdict(int)
        for model_dir_id in tqdm(model_dir_ids, ncols=150, leave=False):
            model_df = pd.read_pickle(
                utils.get_artefacts_path() / args.dataset / model_dir_id / ERROR_ANALYSIS_DF_FILE_NAME
            )
            model_df = model_df[~model_df['is_problematic']]
            model_df = model_df[ErrorClsTypes.filter_error_type(model_df, error_type_mapping[args.error_type])]
            for _, row in model_df.iterrows():
                example_images[(row.img_rel_path, row.target, row.top1)] += 1
                if row.img_rel_path not in img_to_multi_labels:
                    img_to_multi_labels[row.img_rel_path] = row.multi_label
                if (row.img_rel_path, row.top1) not in img_pred_to_error_types:
                    img_pred_to_error_types[(row.img_rel_path, row.top1)] = row.error_types
        example_images = [(k, example_images[k]) for k in example_images]
        example_images = sorted(example_images, key=lambda x: x[1], reverse=True)
        printed_samples = set()
        for i, (k, v) in enumerate(example_images[:1000]):
            if k[0] in printed_samples:
                continue
            printed_samples.add(k[0])
            print(f'{i + 1}) cnt mistakes = {v}, file = {k[0]}')
            print(f'target = {k[1]}, {eval_manager.imagenet_wnet_ids[k[1]]}: {eval_manager.get_description(k[1])}')
            print('multi_labels:')
            for l in img_to_multi_labels[k[0]]:
                print(f'{l}, {eval_manager.imagenet_wnet_ids[l]}: {eval_manager.get_description(l)}')
            print(f'pred = {k[2]}, {eval_manager.imagenet_wnet_ids[k[2]]}: {eval_manager.get_description(k[2])}')
            if args.error_type == 'fg-OOV':
                print(img_pred_to_error_types[(k[0], k[2])])
            print()
