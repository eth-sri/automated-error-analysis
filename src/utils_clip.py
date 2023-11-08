from typing import Dict, List, Set, Tuple

import clip
from loguru import logger
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import datasets, utils

torch.multiprocessing.set_sharing_strategy('file_system')


def add_clip_visual_similarity_info_to_df(df: pd.DataFrame, dataset: str = 'imagenet') -> pd.DataFrame:
    SELECT_TOP_K = 10
    topk_train_indices, topk_train_targets = get_clip_knns(dataset)
    topk_train_indices = topk_train_indices[:, :SELECT_TOP_K]
    topk_train_targets = topk_train_targets[:, :SELECT_TOP_K]
    df['clip_top10_train_indices'] = topk_train_indices.tolist()
    df['clip_top10_train_targets'] = topk_train_targets.tolist()
    return df


def get_clip_knns(dataset: str = 'imagenet'):
    valid_clip_knn = utils.get_artefacts_path() / dataset / 'valid_clip_knn.npz'
    CACHE_SELECT_K = 50

    if valid_clip_knn.is_file():
        loaded = np.load(valid_clip_knn)
        topk_train_indices, topk_train_targets = loaded['topk_train_indices'], loaded['topk_train_targets']
        assert topk_train_indices.shape == topk_train_targets.shape == (utils.NUM_SAMPLES[dataset], CACHE_SELECT_K)
        return topk_train_indices, topk_train_targets

    logger.info('Compute CLIP kNNs (from the training images) for the validation set')
    # CLIP embeddings computed on the ImageNet training set
    train_files, train_targets, train_embeddings = get_train_clip_embeddings()
    device, model, preprocess = get_clip_model()
    train_embeddings = torch.from_numpy(train_embeddings).to(device)
    train_targets = torch.from_numpy(train_targets)

    valid_dset = datasets.get_dataset(dataset=dataset, split='val', transform=preprocess)
    data_loader = DataLoader(valid_dset, batch_size=128, shuffle=False, num_workers=4)

    with torch.no_grad():
        all_topk_train_indices = []
        all_topk_train_targets = []
        for _, inputs, _, _ in tqdm(data_loader, ncols=150, leave=False):
            val_features = model.encode_image(inputs.to(device))
            val_features /= val_features.norm(dim=-1, keepdim=True)
            similarity = val_features @ train_embeddings.T

            _, topk_train_indices = torch.topk(similarity, CACHE_SELECT_K, dim=-1, largest=True, sorted=True)
            topk_train_indices = topk_train_indices.cpu()
            topk_train_targets = train_targets[topk_train_indices]

            all_topk_train_indices.append(topk_train_indices)
            all_topk_train_targets.append(topk_train_targets)

        all_topk_train_indices = torch.cat(all_topk_train_indices).numpy()
        all_topk_train_targets = torch.cat(all_topk_train_targets).numpy()

        assert all_topk_train_indices.shape == (utils.NUM_SAMPLES[dataset], CACHE_SELECT_K)
        assert all_topk_train_targets.shape == (utils.NUM_SAMPLES[dataset], CACHE_SELECT_K)

        np.savez_compressed(
            valid_clip_knn,
            topk_train_indices=all_topk_train_indices,
            topk_train_targets=all_topk_train_targets
        )

        return all_topk_train_indices, all_topk_train_targets


def get_clip_model():
    device = utils.get_device()
    model, preprocess = clip.load('ViT-L/14@336px', device)
    return device, model, preprocess


def get_train_clip_embeddings():
    train_clip_embeddings_path = utils.get_artefacts_path() / 'train_clip_embeddings.npz'

    if train_clip_embeddings_path.is_file():
        logger.info('Load cached CLIP embeddings of the training images.')
        loaded = np.load(train_clip_embeddings_path)
        train_files, targets, embeddings = loaded['train_files'], loaded['targets'], loaded['embeddings']
        train_files = train_files.tolist()
        assert targets.ndim == 1 and embeddings.ndim == 2
        assert len(train_files) == targets.shape[0] == embeddings.shape[0]
        logger.info('Loading completed')
        return train_files, targets, embeddings

    logger.info('Compute the CLIP embeddings of the ImageNet training images.')
    device, model, preprocess = get_clip_model()

    imagenet_train_dset = datasets.get_dataset(dataset='imagenet', split='train', transform=preprocess)
    paths_prefix = imagenet_train_dset.root + '/'
    data_loader = DataLoader(imagenet_train_dset, batch_size=128, shuffle=False, num_workers=4)

    with torch.no_grad():
        train_files_list = []
        all_targets = []
        all_features = []
        for paths, inputs, targets, _ in tqdm(data_loader, ncols=150, leave=False):
            paths = list(paths)
            assert all(p.startswith(paths_prefix) for p in paths)
            paths = [p[len(paths_prefix):] for p in paths]
            train_files_list.extend(paths)

            all_targets.append(targets)

            features = model.encode_image(inputs.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            features = features.cpu()
            all_features.append(features)

        all_targets = torch.cat(all_targets).numpy()
        all_features = torch.cat(all_features).numpy()

        assert all_targets.ndim == 1 and all_features.ndim == 2
        assert len(train_files_list) == all_targets.shape[0] == all_features.shape[0]

        logger.info('Cache/save the CLIP embeddings of the training images.')
        np.savez_compressed(
            train_clip_embeddings_path,
            train_files=np.array(train_files_list),
            targets=all_targets,
            embeddings=all_features
        )

        return train_files_list, all_targets, all_features


def get_wnet_proposals(
        wnet_id_pred: str, wnet_id_target: str, wnet_id_to_superclass: Dict[str, Set[str]], debug_oov: bool = False
) -> List[Tuple[str, int]]:
    assert wnet_id_pred != wnet_id_target
    wn_synset_pred = wn.synset_from_pos_and_offset(wnet_id_pred[0], int(wnet_id_pred[1:]))
    wn_synset_target = wn.synset_from_pos_and_offset(wnet_id_target[0], int(wnet_id_target[1:]))
    blocking_ancestors: Set[int] = {node.offset() for path in wn_synset_target.hypernym_paths() for node in path}

    class Proposals:

        def __init__(self):
            self.proposals = []
            self.proposals_offset_set = set()

        def add_proposal_from_synset(self, synset):
            if synset.offset() in self.proposals_offset_set:
                return
            if debug_oov:
                for lemma_name in synset.lemma_names():
                    print(synset.offset(), lemma_name.lower().replace('_', ' '))
            self.proposals.append(synset)
            self.proposals_offset_set.add(synset.offset())

        def to_lemma_names_list(self) -> List[Tuple[str, int]]:
            return [
                (lemma_name.lower().replace('_', ' '), proposed_synset.offset())
                for proposed_synset in self.proposals for lemma_name in proposed_synset.lemma_names()
            ]

    if debug_oov: print('IV and OOV proposals:')

    proposals = Proposals()
    assert wn_synset_pred.offset() not in blocking_ancestors, \
        f'Maybe should be fine-grained in-voc: target={wnet_id_target}, pred={wnet_id_pred}'
    # proposals.add_proposal_from_synset(wn_synset_target)
    if debug_oov: print('* Predicted node')
    proposals.add_proposal_from_synset(wn_synset_pred)

    if debug_oov: print('* Same superclass with the predicted node')
    # Add from same superclass
    for wnet_id, superclass_set in wnet_id_to_superclass.items():
        if not wnet_id_to_superclass[wnet_id_pred].isdisjoint(superclass_set):
            synset = wn.synset_from_pos_and_offset(wnet_id[0], int(wnet_id[1:]))
            assert synset.offset() not in blocking_ancestors, \
                f'Maybe should be fine-grained in-voc: target={wnet_id_target}, pred={wnet_id_pred}, ' + \
                f'same superset clash={synset.offset()}'
            proposals.add_proposal_from_synset(synset)

    if debug_oov: print('* Direct siblings of the predicted node')
    # Propose the direct siblings of the predicted node
    pred_direct_parents = wn_synset_pred.hypernyms()
    for direct_parent in pred_direct_parents:
        for pred_sibling in direct_parent.hyponyms():
            if pred_sibling.offset() not in blocking_ancestors:
                proposals.add_proposal_from_synset(pred_sibling)

    # Ancestors of the predicted node (which are not also ancestors of the target)
    META_LABELS = [
        # From the ImageNet-X paper
        'device',
        'dog',
        'commodity',
        'bird',
        'structure',
        'covering',
        'wheeled_vehicle',
        'food',
        'equipment',
        'insect',
        'vehicle',
        'furniture',
        'primate',
        'vessel',
        'snake',
        'natural_object',

        # more general
        'organism',
        'geological_formation',
        'clothing',
        'container'
    ]

    if debug_oov: print('* (Valid) parents of the predicted node')
    for path in wn_synset_pred.hypernym_paths():
        covered_by_meta_label = False
        for node in path:
            if not covered_by_meta_label and any(meta_label in node.lemma_names() for meta_label in META_LABELS):
                covered_by_meta_label = True
            if node.offset() in blocking_ancestors:
                continue
            if covered_by_meta_label:
                proposals.add_proposal_from_synset(node)

    return proposals.to_lemma_names_list()


if __name__ == '__main__':
    pass
