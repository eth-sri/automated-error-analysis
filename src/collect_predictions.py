import argparse
import datetime
import json
from pprint import pprint

import torch.multiprocessing

from src.registry import ALL_MODULES, ModelRegistry
from src import utils

torch.multiprocessing.set_sharing_strategy('file_system')
utils.set_reproducibility()

MODEL_METADATA_FILE_NAME = 'metadata.json'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'imagenet-a'],
                    help="Dataset on which to run the models")
parser.add_argument('--modules', nargs='+', default=['all'],
                    help="Compute logits and predictions of models in these modules (space separated)")
parser.add_argument('--list_modules', action='store_true', default=False,
                    help="List the allowed modules")
parser.add_argument('--get_computation_time', action='store_true', default=False,
                    help="Calculate the total computation time for collecting the predictions of the models")
args = parser.parse_args()


if args.list_modules:
    print('Allowed modules:')
    pprint(ALL_MODULES)
elif args.get_computation_time:
    model_dir_ids = utils.list_model_dir_ids(args.dataset)
    total_time = 0
    for model_dir_id in model_dir_ids:
        with open(utils.get_artefacts_path() / args.dataset / model_dir_id / MODEL_METADATA_FILE_NAME, 'r') as meta_f:
            model_metadata = json.load(meta_f)
            total_time += model_metadata['preds_comp_time']
    print(f'Total time: {total_time:.2f} sec = {datetime.timedelta(seconds=total_time)}')
else:
    if 'all' in args.modules:
        assert len(args.modules) == 1
        modules = ALL_MODULES
    else:
        modules = args.modules

    assert all(module_name in ALL_MODULES for module_name in modules)

    for module_name in modules:
        registry = ModelRegistry()
        registry.add_models_from_module(module_name)

        for model in registry.models.values():
            model.save_preds(args.dataset)
