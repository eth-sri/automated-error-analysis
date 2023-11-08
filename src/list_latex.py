import argparse
import collections
import csv
import dataclasses
from dataclasses import dataclass
import json
from typing import List

from src import evaluation, utils

# Usage:
# python list_latex.py --models_summary_latex
# python list_latex.py --models_summary_csv
# python list_latex.py --superclasses
parser = argparse.ArgumentParser(description='Latex formatting')
parser.add_argument('--models_summary_latex', action='store_true', default=False,
                    help='Latex summary of all evaluated models')
parser.add_argument('--models_summary_csv', action='store_true', default=False,
                    help='CSV summary of all evaluated models')
parser.add_argument('--superclasses', action='store_true', default=False,
                    help='List (LaTeX formatted) superclass definitions')
args = parser.parse_args()

model_dir_ids = utils.list_model_dir_ids()

if args.models_summary_latex:
    MODEL_SUMMARY_TEMPLATE = '{}. '

    @dataclass
    class ModelRow:
        model_id: str
        source: str
        arch_family: str
        dataset: str

        def write_row(self) -> str:
            return '& {} & {} & {} & {} \\\\'.format(
                self.model_id,
                self.source,
                self.arch_family,
                self.dataset
            )

    rows = []
    for model_dir_id in model_dir_ids:
        metadata_path = utils.get_artefacts_path() / 'imagenet' / model_dir_id / evaluation.MODEL_METADATA_FILE_NAME
        with open(metadata_path, 'r') as metadata_file:
            model_metadata = json.load(metadata_file)
        source = model_metadata['source']
        if source.startswith('torch'):
            source = 'torch'
        arch_family = model_metadata['arch_family']
        if arch_family == 'transformer':
            arch_family = 'vit'
        training_data = model_metadata['training_data']
        rows.append(ModelRow(
            model_id=model_dir_id[model_dir_id.index('--') + 2:].replace('_', '\_'),
            source=source,
            arch_family=arch_family,
            dataset=training_data[:training_data.index('(') - 1]
        ))
    rows.sort(key=lambda r: r.arch_family + ' | ' + r.model_id)

    for i, r in enumerate(rows):
        print(f'{i + 1}.', r.write_row())

if args.models_summary_csv:
    @dataclass
    class ModelRow:
        model_id: str
        source: str
        arch: str
        arch_group: str
        arch_family: str
        dataset: str
        extra_annotations: List[str]

        def write_row(self) -> str:
            return '{}, {}, {}, {}, {}, {}, {}'.format(
                self.model_id,
                self.source,
                self.arch,
                self.arch_group,
                self.arch_family,
                self.dataset,
                self.extra_annotations
            )

    rows = []
    for model_dir_id in model_dir_ids:
        metadata_path = utils.get_artefacts_path() / 'imagenet' / model_dir_id / evaluation.MODEL_METADATA_FILE_NAME
        with open(metadata_path, 'r') as metadata_file:
            model_metadata = json.load(metadata_file)
        arch = model_metadata['arch']
        training_data = model_metadata['training_data']
        rows.append(ModelRow(
            model_id=model_dir_id[model_dir_id.index('--') + 2:],
            source=model_metadata['source'],
            arch=arch,
            arch_group=evaluation.ARCH_TO_ARCH_GROUP[arch],
            arch_family=model_metadata['arch_family'],
            dataset=training_data[:training_data.index('(') - 1],
            extra_annotations=model_metadata['extra_annotations']
        ))
    rows.sort(key=lambda r: r.arch_family + ' | ' + r.arch_group + ' | ' + r.arch + ' | ' + r.model_id)

    with open(utils.get_root_path() / 'models_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['idx', 'model_id', 'source', 'arch', 'arch_group', 'arch_family', 'dataset', 'extra_annotations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, r in enumerate(rows):
            row = dataclasses.asdict(r)
            row['idx'] = i + 1
            writer.writerow(row)

if args.superclasses:
    with open(utils.get_artefacts_path() / 'superclasses.txt', 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        lines = [l for l in lines if l]

    d = collections.deque(lines)
    cnt_superclasses = 0
    while d:
        l = d.popleft()
        idx = l.index(':')
        superclass_name = l[:idx].replace('_', '\_')
        l = l[idx + 1:-len(' wnet classes')].strip()
        superclass_size = int(l)
        SUPERCLASS_DEF = '\item '
        if superclass_name != 'no superclass':
            SUPERCLASS_DEF += '\\textbf{\\texttt{'
            SUPERCLASS_DEF += superclass_name
            SUPERCLASS_DEF += '}}'
        else:
            SUPERCLASS_DEF += '\\textbf{\\texttt{'
            SUPERCLASS_DEF += 'Without a superclass'
            SUPERCLASS_DEF += '}}'
        SUPERCLASS_DEF += f' -- {superclass_size} classes: '
        s = []
        for _ in range(superclass_size):
            l = d.popleft()
            s.append(f'\inc{{({l[:9]}, {l[10:].split(",")[0]})}}')
        print(SUPERCLASS_DEF)
        print()
        # print(', '.join(s))
        tbl = ''
        tbl += '\\begin{tabular}{p{0.5\\linewidth}p{0.5\\linewidth}}\n'
        if len(s) % 2:
            s.append('')
        for a, b in zip(s[::2], s[1::2]):
            tbl += a + ' & ' + b + '\\\\' + '\n'
        tbl += '\\end{tabular}'
        print(tbl)
