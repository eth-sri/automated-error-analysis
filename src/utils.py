import collections
import itertools
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple
import urllib.request

from loguru import logger
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

NUM_SAMPLES = {
    'imagenet': 50000,
    'imagenet-a': 7500,
}
NUM_CLASSES = 1000


# Device:

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Randomness & reproducibility:

def set_reproducibility(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Files and paths:

def get_root_path() -> Path:
    """Returns the path to automatic-blindspot-detection/."""
    return Path(__file__).resolve().parent.parent


def get_or_create_path(p: Path) -> Path:
    """Create a folder if it does not already exist."""
    if not p.is_dir():
        p.mkdir()
    return p


def get_artefacts_path() -> Path:
    """Returns the path to automatic-blindspot-detection/artefacts/."""
    artefacts_path = get_root_path() / 'artefacts'
    return get_or_create_path(artefacts_path)


def list_model_dir_ids(dataset: str = 'imagenet') -> List[str]:
    artefacts_path = get_artefacts_path() / dataset
    models_data_folders = sorted(list(artefacts_path.glob('**')))
    return [p.parts[-1] for p in models_data_folders if p.is_dir() and p.parent == artefacts_path]


# Eval utils:

def load_labels_map() -> List[str]:
    with open(get_artefacts_path() / 'labels_map.json') as json_file:
        label_map = json.load(json_file)
    return [label_map[str(i)] for i in range(NUM_CLASSES)]


def load_samples_info(dataset: str = 'imagenet'):
    samples_info_path = get_artefacts_path() / dataset / 'samples_info.npz'
    assert samples_info_path.is_file()
    loaded = np.load(samples_info_path)
    sample_files, targets = loaded['sample_files'], loaded['targets']
    sample_files = sample_files.tolist()
    assert len(sample_files) == NUM_SAMPLES[dataset]
    assert targets.ndim == 1 and targets.shape[0] == NUM_SAMPLES[dataset]
    return sample_files, targets


def create_initial_samples_df(labels_map: Optional[List[str]] = None, dataset: str = 'imagenet') -> pd.DataFrame:
    sample_files, targets = load_samples_info(dataset)
    assert all(0 <= t < 1000 for t in targets)
    if labels_map is None:
        labels_map = load_labels_map()
    assert all(len(img_rel_path.split('/')) == 2 for img_rel_path in sample_files)
    df = pd.DataFrame({
        'wnet_id': [img_rel_path.split('/')[0] for img_rel_path in sample_files],
        'file_name': [img_rel_path.split('/')[1] for img_rel_path in sample_files],
        'img_rel_path': sample_files,
        'target': targets,  # ImageNet-indexed (0 to 1000) class ids / targets
        'target_desc': [labels_map[t] for t in targets],
    })
    return df


def add_multi_labels_to_df(df: pd.DataFrame, labels_map: Optional[List[str]] = None, dataset: str = 'imagenet') \
        -> pd.DataFrame:
    assert dataset in ['imagenet', 'imagenet-a']

    # The input dataframe df should have a "file_name" column.
    img_to_multi_labels = collections.defaultdict(list)
    img_to_wrong_multi_labels = collections.defaultdict(list)
    img_to_is_problematic = collections.defaultdict(bool)

    if dataset == 'imagenet':
        # TensorFlow dataset: imagenet2012_multilabel
        # Evaluating Machine Accuracy on ImageNet, ICML 2020
        # When does dough become a bagel? Analyzing the remaining mistakes on ImageNet, NeurIPS 2022
        logger.info('Load multilabels from imagenet2012_multilabel')
        ds = tfds.load('imagenet2012_multilabel', split='validation')
        for example in ds:
            file_name = example['file_name'].numpy().decode("utf-8")

            assert file_name not in img_to_multi_labels
            assert file_name not in img_to_wrong_multi_labels
            assert file_name not in img_to_is_problematic

            img_to_multi_labels[file_name] = \
                example['correct_multi_labels'].numpy().tolist() + example['unclear_multi_labels'].numpy().tolist()
            img_to_wrong_multi_labels[file_name] = example['wrong_multi_labels'].numpy().tolist()
            img_to_is_problematic[file_name] = example['is_problematic'].numpy()
    elif dataset == 'imagenet-a':
        # "... ensure images do not fall into more than one of the several hundred classes"
        for _, row in df.iterrows():
            img_to_multi_labels[row.file_name] = [row.target]
            img_to_is_problematic[row.file_name] = False
    else:
        raise ValueError(f'Multi-labels for the {dataset} dataset are not supported')

    df['has_defined_multi_labels'] = df['file_name'].map(lambda file_name: file_name in img_to_multi_labels)
    df['multi_label'] = df['file_name'].map(img_to_multi_labels)
    df['wrong_multi_label'] = df['file_name'].map(img_to_wrong_multi_labels)
    df['is_problematic'] = df['file_name'].map(img_to_is_problematic)

    if labels_map is None:
        labels_map = load_labels_map()
    df['multi_desc'] = df['multi_label'].map(lambda multi_labels: [f'({labels_map[t]})' for t in multi_labels])
    df['wrong_multi_desc'] = df['wrong_multi_label'].map(
        lambda wrong_multi_labels: [f'({labels_map[t]})' for t in wrong_multi_labels]
    )

    return df


def load_model_preds(dataset: str, model_dir_id: str, labels_map: Optional[List[str]] = None):
    # Load the predictions for the whole ImageNet evaluation dataset (not only imagenet2012_multi)!
    preds_path = get_artefacts_path() / dataset / model_dir_id / 'preds.npz'
    assert preds_path.is_file()
    loaded = np.load(preds_path)
    top1, top5, probs = loaded['top1'], loaded['top5'], loaded['probs']
    assert top1.ndim == 1 and top1.shape[0] == NUM_SAMPLES[dataset]
    assert top5.ndim == 2 and top5.shape == (NUM_SAMPLES[dataset], 5)
    assert probs.ndim == 2 and probs.shape == (NUM_SAMPLES[dataset], NUM_CLASSES)

    if labels_map is None:
        labels_map = load_labels_map()
    top1_desc = [labels_map[t] for t in top1]

    return top1, top1_desc, top5, probs


def load_dough_bagel_logits_as_dict(path):
    logits_array = np.load(tf.io.gfile.GFile(path, 'rb'))
    ret = {}
    for i in range(len(logits_array)):
        filename = f'ILSVRC2012_val_{i+1:08}.JPEG'
        ret[filename] = logits_array[i]
    return ret


def add_model_predictions_to_df(
        df: pd.DataFrame, model_dir_id: str, dataset: str = 'imagenet', labels_map: Optional[List[str]] = None
) -> pd.DataFrame:
    if model_dir_id not in ['vit3b', 'greedysoups']:
        top1, top1_desc, _, _ = load_model_preds(dataset, model_dir_id, labels_map)
        df['top1'] = top1.tolist()
        df['top1_desc'] = top1_desc
    else:
        assert dataset == 'imagenet'
        model_logits_file = get_or_create_path(get_artefacts_path() / 'imagenet') / f'{model_dir_id}.npz'
        if not model_logits_file.is_file():
            urllib.request.urlretrieve(
                f'https://storage.googleapis.com/brain-car-datasets/imagenet-mistakes/logits/{model_dir_id}.npz',
                model_logits_file
            )
        model_logits = load_dough_bagel_logits_as_dict(model_logits_file)
        df['top1'] = df['file_name'].map(lambda file_name: model_logits[file_name].argmax())
        if labels_map is None:
            labels_map = load_labels_map()
        df['top1_desc'] = df['top1'].map(lambda t: labels_map[t])
    return df


def common_co_occurrences(class_id_to_superclass: Dict[int, Set[str]]) -> Set[Tuple[int, int]]:
    # TODO: Think about lifting it to spurious correlations between the superclasses?
    # Define spurious correlations based on co-occurence counts/probabilities
    # (computed on the examples *not* in imagenet2012_multilabel):
    file_names_in_multilabel_dset = set()
    imagenet_multilabel_ds = tfds.load('imagenet2012_multilabel', split='validation')
    for example in imagenet_multilabel_ds:
        file_name = example['file_name'].numpy().decode("utf-8")
        file_names_in_multilabel_dset.add(file_name)
    label_cnt = [0] * NUM_CLASSES
    co_occurence_cnt = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    # TensorFlow dataset: imagenet2012_real
    # Are we done with ImageNet, arXiv 2020
    logger.info('Load multilabels from imagenet2012_real')
    imagenet_real_ds = tfds.load('imagenet2012_real', split='validation')

    real_ds_size = 0
    overlap_with_multi_label_ds = 0
    no_multi_labels = 0
    single_multi_label = 0
    extracted_from_num_samples = 0

    unique_pairs = set()
    tot_non_unique_pairs = 0
    for example in imagenet_real_ds:
        real_ds_size += 1

        file_name = example['file_name'].numpy().decode("utf-8")
        multi_labels = example['real_label'].numpy().tolist()

        if file_name in file_names_in_multilabel_dset:
            overlap_with_multi_label_ds += 1
            continue

        if not multi_labels:
            no_multi_labels += 1
            continue

        assert len(multi_labels) == len(set(multi_labels))
        original_label = example['original_label'].numpy()
        if original_label not in multi_labels:
            multi_labels.append(original_label)

        for a in multi_labels:
            label_cnt[a] += 1

        if len(multi_labels) == 1:
            single_multi_label += 1
            continue

        assert len(multi_labels) > 1
        extracted_from_num_samples += 1
        for a, b in itertools.combinations(multi_labels, 2):
            co_occurence_cnt[a][b] += 1
            co_occurence_cnt[b][a] += 1
            tot_non_unique_pairs += 1
            unique_pairs.add((min(a, b), max(a, b)))

    logger.debug('Common co-occurrence pairs:')
    logger.debug(f'real_ds_size = {real_ds_size}')
    logger.debug(f'overlap_with_multi_label_ds = {overlap_with_multi_label_ds}')
    logger.debug(f'no_multi_labels = {no_multi_labels}')
    logger.debug(f'single_multi_label = {single_multi_label}')
    logger.debug(f'extracted from {extracted_from_num_samples} samples')

    logger.debug(
        f'Total number of non-unique pairs in the ReaL (but not multi_label) dataset: {tot_non_unique_pairs}'
    )
    logger.debug(
        'Total number of unique pairs in the ReaL (but not multi_label) dataset ' +
        f'(before same superclass and count >= 2 filtering): {len(unique_pairs)}'
    )
    logger.debug(f'Extracted from {extracted_from_num_samples} ReaL (but not multi_label) images.')

    unique_pairs = {(a, b) for a, b in unique_pairs if class_id_to_superclass[a].isdisjoint(class_id_to_superclass[b])}
    logger.debug(f'After filtering for a shared superclass: {len(unique_pairs)} pairs')

    unique_pairs = {(a, b) for a, b in unique_pairs if co_occurence_cnt[a][b] >= 2}
    logger.debug(f'Appearing more than once: {len(unique_pairs)} pairs')

    ret_pairs = set()
    for a in range(NUM_CLASSES):
        for b in range(a, NUM_CLASSES):
            if class_id_to_superclass[a].isdisjoint(class_id_to_superclass[b]) and co_occurence_cnt[a][b] >= 2:
                ret_pairs.add((a, b))
                ret_pairs.add((b, a))
    assert len(ret_pairs) == 2 * len(unique_pairs)

    return ret_pairs


def get_synset_from_wnet_id(wnet_id: str):
    return wn.synset_from_pos_and_offset(wnet_id[0], int(wnet_id[1:]))


def print_wnet_paths(wnet_id: str, wnet_desc: Optional[str] = None):
    synset = get_synset_from_wnet_id(wnet_id)
    if wnet_desc is not None:
        print(wnet_id, synset, wnet_desc)
    else:
        print(wnet_id, synset)
    hypernym_paths = synset.hypernym_paths()
    for path in hypernym_paths:
        print('*', path)


def get_wnet_distance(wnet_id1: str, wnet_id2: str) -> float:
    synset1 = get_synset_from_wnet_id(wnet_id1)
    synset2 = get_synset_from_wnet_id(wnet_id2)
    return synset1.path_similarity(synset2)


def get_synsets_superset(wnet_id: str):
    wn_synset = get_synset_from_wnet_id(wnet_id)
    hypernym_paths = wn_synset.hypernym_paths()
    return [element for sublist in hypernym_paths for element in sublist]  # flatten lists.


def lemma_names_in_synsets_superset(lemma_predicate: List[str], synsets_superset) -> List[bool]:
    return [
        any(lemma_name in synset.lemma_names() for synset in synsets_superset)
        for lemma_name in lemma_predicate
    ]


def define_superclasses(imagenet_wnet_id_to_target: Dict[str, int], target_desc_list: List[str]) \
        -> Tuple[Dict[int, Set[str]], Dict[str, Set[str]]]:
    logger.info('Define superclasses')

    rows = [(wnet_id, target, target_desc_list[target]) for wnet_id, target in imagenet_wnet_id_to_target.items()]

    # The order of the superclass definitions is no longer important, so the superclass definitions can be optionally
    # written in a single json dictionary.
    superclass_definitions = []
    with open(get_artefacts_path() / 'superclasses.json', 'r') as f:
        for line in f:
            superclass_definitions.append(json.loads(line))

    wnet_id_to_superclass = collections.defaultdict(set)
    class_id_to_superclass = collections.defaultdict(set)
    superclass_size = collections.defaultdict(int)
    without_superclass = 0
    for wnet_id, target, target_desc in rows:
        synsets_superset = get_synsets_superset(wnet_id)

        for superclass_def in superclass_definitions:
            assert len(superclass_def) == 1
            superclass_def_list = list(superclass_def.items())
            assert len(superclass_def_list) == 1

            superclass_id, lemma_predicates = superclass_def_list[0]

            any_predicate_contained_in_superclass = any(
                all(lemma_names_in_synsets_superset(lemma_predicate, synsets_superset))
                for lemma_predicate in lemma_predicates
            )

            if any_predicate_contained_in_superclass:
                wnet_id_to_superclass[wnet_id].add(superclass_id)
                class_id_to_superclass[target].add(superclass_id)
                superclass_size[superclass_id] += 1
                # break  # the break is not important anymore

        assert len(wnet_id_to_superclass[wnet_id]) == len(class_id_to_superclass[target])
        if len(wnet_id_to_superclass[wnet_id]) == 0:
            without_superclass += 1

    with open(get_artefacts_path() / 'superclasses.txt', 'w') as superclasses_txt_file:
        min_superclass_size = 1000
        max_superclass_size = -1
        for superclass_def in superclass_definitions:
            superclass_def_list = list(superclass_def.items())
            superclass_id, _ = superclass_def_list[0]
            superclasses_txt_file.write(f'{superclass_id}: {superclass_size[superclass_id]} wnet classes\n')
            min_superclass_size = min(min_superclass_size, superclass_size[superclass_id])
            max_superclass_size = max(max_superclass_size, superclass_size[superclass_id])
            for wnet_id, _, target_desc in rows:
                if superclass_id in wnet_id_to_superclass[wnet_id]:
                    superclasses_txt_file.write(f'{wnet_id} {target_desc}\n')
            superclasses_txt_file.write('\n')

        superclasses_txt_file.write(f'no superclass: {without_superclass} wnet classes\n')
        for wnet_id, _, target_desc in rows:
            if len(wnet_id_to_superclass[wnet_id]) == 0:
                superclasses_txt_file.write(f'{wnet_id} {target_desc}\n')

    def left_matching(matching, not_matching):
        left = []
        for wnet_id, _, target_desc in rows:
            if wnet_id in wnet_id_to_superclass:
                continue
            wn_synset = wn.synset_from_pos_and_offset(wnet_id[0], int(wnet_id[1:]))
            hypernym_paths = wn_synset.hypernym_paths()
            synsets_superset = [element for sublist in hypernym_paths for element in sublist]

            if not matching or all(lemma_names_in_synsets_superset(matching, synsets_superset)):
                if not not_matching or not any(lemma_names_in_synsets_superset(not_matching, synsets_superset)):
                    left.append((wnet_id, target_desc, hypernym_paths))
        return left

    left_wnet_ids = left_matching([], [])
    if left_wnet_ids:
        logger.debug(f'Total left wnet ids: {len(left_wnet_ids)}')
        for wnet_id, target_desc, hypernym_paths in left_wnet_ids:
            logger.debug(f'{wnet_id} {target_desc}')
            for hypernym_path in hypernym_paths:
                logger.debug(f'* {hypernym_path}')
            logger.debug('')

    return class_id_to_superclass, wnet_id_to_superclass


def get_dough_bagel_non_prototypical_samples() -> Set[str]:
    imagenet_mistakes_metadata_path = get_root_path() / 'imagenet-mistakes' / 'metadata'
    with open(imagenet_mistakes_metadata_path / 'greedy_soups_mistakes.json', 'r') as json_file:
        greedy_soups_mistakes = json.load(json_file)
    with open(imagenet_mistakes_metadata_path / 'vit3b_mistakes.json', 'r') as json_file:
        vit3b_mistakes = json.load(json_file)

    FAILURE_CATEGORY = 'failure category'
    NON_PROTOTYPICAL = 'Non-prototypical'
    return set(
        k for k, mistake_metadata in itertools.chain(greedy_soups_mistakes.items(), vit3b_mistakes.items())
        if mistake_metadata[FAILURE_CATEGORY] == NON_PROTOTYPICAL
    )


def dough_bagel_collapsed_class_definitions() -> Dict[int, List[int]]:
    # Any of the keys in the collapsed mapping dictionary appear in multi labels: 1116 times.
    # From those cases, (all of) their allowed predictions also appear in the corresponding multi labels: 1103 times,
    # i.e., 98.8% of the times.
    return {
        # All siberian huskies and malamutes are also eskimo dogs:
        250: [248], 249: [248],
        # Sunglass and sunglasses are the same class (bidirectional):
        836: [837], 837: [836],
        # Indian and African elephants are also tuskers:
        385: [101], 386: [101],
        # A coffee mug is also a cup:
        504: [968],
        # Maillot and maillot, tanksuit are the same class (bidirectional):
        638: [639], 639: [638],
        # Missile and projectile missile are the same class (bidirectional):
        657: [744], 744: [657],
        # Notebook computer and laptop are the same class (bidirectional):
        620: [681], 681: [620],
        # Monitor and screen are the same class (bidirectional):
        664: [782], 782: [664],
        # A cassette player is also a tape player:
        482: [848],
        # Weasel, polecats, black-footed ferrets, and minks are all the same class:
        356: [357, 358, 359], 357: [356, 358, 359], 358: [356, 357, 359], 359: [356, 357, 358],
        # All bathtubs are tubs, but not all tubs are bathtubs:
        435: [876]
    }


if __name__ == '__main__':
    pass
