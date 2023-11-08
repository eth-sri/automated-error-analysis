import argparse
import collections
import os
from typing import Dict, List, Optional, Tuple

from loguru import logger
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import seaborn as sns

from src import evaluation, utils


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatio']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

ORGANISM_ARTIFACT_GROUP = ['organism', 'artifact']
ORGANISM_GROUP = ['organism']
ORGANISM_ARTIFACT_OTHER_GROUP = ORGANISM_ARTIFACT_GROUP + ['other']
ALL_GROUP = ['all']

DATA_TO_DATA_SIZE = {
    'ImageNet-1k (1.2M)': 1.2e6,
    'ImageNet-21k (14M)': 14e6,
    'Merged (30M)': 30e6,
    'Merged (38M)': 38e6,
    'Flickr YFCC (90M)': 90e6,
    'JFT (300M)': 300e6,
    'WIT (400M)': 400e6,
    'LAION-400M (400M)': 400e6,
    'Instagram (940M)': 940e6,
    'RandomInternetImages (2B)': 2e9,
    'LAION-2B (2.3B)': 2.3e9,
    'Instagram (3.6B)': 3.6e9
}

DATA_TO_DATA_GROUP = {
    'ImageNet-1k (1.2M)': 'small (< 5M)',

    'ImageNet-21k (14M)': 'medium (< 50M)',
    'Merged (38M)': 'medium (< 50M)',
    'Merged (30M)': 'medium (< 50M)',

    'JFT (300M)': 'large (< 500M)',
    'WIT (400M)': 'large (< 500M)',
    'Flickr YFCC (90M)': 'large (< 500M)',
    'LAION-400M (400M)': 'large (< 500M)',

    'RandomInternetImages (2B)': 'xlarge (> 500M)',
    'Instagram (3.6B)': 'xlarge (> 500M)',
    'LAION-2B (2.3B)': 'xlarge (> 500M)',
    'Instagram (940M)': 'xlarge (> 500M)',
}


def translate_dataset_to_size(results_df: pd.DataFrame) -> None:
    if 'training_data_size' not in results_df.columns:
        results_df.insert(
            3, 'training_data_size',
            results_df.apply(lambda x: DATA_TO_DATA_SIZE[x['training_data']], axis=1)
        )


def define_data_groups(results_df: pd.DataFrame) -> None:
    idx = results_df.columns.values.tolist().index('training_data') + 1
    if 'training_data_group' not in results_df.columns:
        results_df.insert(idx, 'training_data_group', results_df['training_data'].map(DATA_TO_DATA_GROUP))


def define_arch_groups(results_df: pd.DataFrame) -> None:
    idx = results_df.columns.values.tolist().index('arch') + 1
    if 'arch_group' not in results_df.columns:
        results_df.insert(idx, 'arch_group', results_df['arch'].map(evaluation.ARCH_TO_ARCH_GROUP))


def define_pretraining_method(results_df: pd.DataFrame) -> None:
    idx = results_df.columns.values.tolist().index('extra_annotations') + 1
    if 'pretrain_method' not in results_df.columns:
        pretrain_method_list = []
        for _, row in results_df.iterrows():
            extra_annot = eval(row.extra_annotations)
            assert isinstance(extra_annot, list)
            if len(extra_annot) >= 2:
                assert extra_annot == ['Self-Supervised Learning', 'CLIP Training']
                pretrain_method_list.append('CLIP Training')
            elif not extra_annot:
                pretrain_method_list.append('Standard')
            else:
                assert len(extra_annot) == 1
                pretrain_method_list.append(extra_annot[0])
        results_df.insert(idx, 'pretrain_method', pretrain_method_list)


def read_and_prepare_stats_dfs(dataset: str) -> Dict[str, pd.DataFrame]:
    results_dir = utils.get_root_path() / f'stats-{dataset}'

    results_dfs = {}
    for group in ORGANISM_ARTIFACT_OTHER_GROUP + ALL_GROUP:
        results_df = pd.read_csv(results_dir / f'non_problematic_stats_{group}.csv')

        translate_dataset_to_size(results_df)
        define_data_groups(results_df)
        define_arch_groups(results_df)
        define_pretraining_method(results_df)

        results_dfs[group] = results_df

    if dataset == 'imagenet-a':
        for results_df in results_dfs.values():
            assert results_df['num_ambiguous'].sum() == 0
            assert results_df['num_non_prototypical'].sum() == 0

    return results_dfs


def filter_dataframe(
        results_df: pd.DataFrame,
        list_of_columns: List[str],
        list_of_operatorts: List[str],
        list_of_references: list) -> pd.DataFrame:
    # Example for dataframe filtering
    # filter_dataframe(results_df, ["num_params"], ["<"], [5e7])
    current_df = results_df
    n_conditions = len(list_of_columns)
    valid_operators = [">", ">=", "<", "<=", "==", "in", "!=", "not in", "*"]
    idxs = np.ones(len(current_df), dtype=bool)

    assert len(list_of_columns) == n_conditions
    assert len(list_of_operatorts) == n_conditions
    assert len(list_of_references) == n_conditions

    for i, (index, operator, reference) in enumerate(zip(list_of_columns, list_of_operatorts, list_of_references)):
        assert operator in valid_operators
        if operator == "*":
            continue
        elif operator in ["in", "not in"]:
            idxs_tmp = current_df.apply(lambda x: x[index] in reference, axis=1)
            idxs = idxs.__and__(~idxs_tmp if "not" in operator else idxs_tmp)
        else:
            idxs = idxs.__and__(eval(f"current_df[index] {operator} reference"))

    return results_df[idxs]


def conf_lin_fit(x, y, idx=None, x_offset=0):
    if idx is None:
        res = scipy.stats.linregress(x, y)
        n = len(x)
    else:
        res = scipy.stats.linregress(x[idx], y[idx])
        n = len(x[idx])
    a, b, r, std_a, std_b = res.slope, res.intercept, res.rvalue, res.stderr, res.intercept_stderr
    mse = ((a * x + b - y) ** 2).mean()
    xxs = ((x - x.mean()) ** 2).sum()
    t = scipy.stats.t.ppf(1-0.05, n - 2)
    a_cf = t * np.sqrt(mse/xxs)
    b_cf = t * np.sqrt(mse/n)

    ref_x = np.array([80, 90, 95, 98.])
    lbs, ubs = eval_conf(a, b, a_cf, b_cf, ref_x-x_offset)

    ref_str = "range at " + "; ".join([
        f"{x_i:.0f}: [{lb_i:.2f}, {ub_i:.2f}]" for x_i, lb_i, ub_i in zip(ref_x, lbs, ubs)
    ])

    print(f'slope: {a:.4f}, intercept: {b:.4f}, rvalue: {r:.4f}, r^2: {r**2:.4f}, ' +
          f'a_conf: [{a - a_cf:.4f} , {a + a_cf:.4f}], std_b: [{b - b_cf:.4f} , {b + b_cf:.4f}], n: {n}')
    print(ref_str)
    return a, b, a_cf, b_cf


def eval_conf(a, b, a_cf, b_cf, x):
    lb = np.minimum(x * (a - a_cf), x * (a + a_cf)) + b - b_cf
    ub = np.maximum(x * (a - a_cf), x * (a + a_cf)) + b + b_cf
    return lb, ub


def plot_conf(x, y, idx=None, alpha=0.2, **kwargs):
    x_conf = np.arange(0, 100.1, 1) - x.mean()
    xp = x - x.mean()
    a, b, a_cf, b_cf = conf_lin_fit(xp, y, idx=idx, x_offset=x.mean())
    lb, ub = eval_conf(a, b, a_cf, b_cf, x_conf)
    plt.fill_between(
        x_conf + x.mean(),
        lb,
        ub,
        alpha=alpha, color='k',
        **kwargs
    )
    plt.plot(x_conf + x.mean(), x_conf * a + b, 'k', alpha=alpha)


COLORMAPS = {
    'organism': sns.color_palette("crest", as_cmap=True),
    'artifact': sns.color_palette("flare", as_cmap=True),
    'other': sns.dark_palette("#69d", reverse=True, as_cmap=True),
    'all': sns.color_palette("dark:salmon_r", as_cmap=True)
}

FIGURES_FOLDER = 'figures'

# Set marker paths:
marker_map_datasets = collections.OrderedDict([
    ('ImageNet-1k (1.2M)', 'o'),
    ('ImageNet-21k (14M)', 's'),
    ('Merged (30M)', 'p'),
    ('Merged (38M)', '8'),
    ('Flickr YFCC (90M)', '*'),
    ('JFT (300M)', 'D'),
    ('WIT (400M)', '<'),
    ('LAION-400M (400M)', '>'),
    ('Instagram (940M)', 'X'),
    ('RandomInternetImages (2B)', 'v'),
    ('LAION-2B (2.3B)', '^'),
    ('Instagram (3.6B)', 'P'),
])

marker_map_data_group = collections.OrderedDict([
    ('small (< 5M)', 'o'),
    ('medium (< 50M)', 'd'),
    ('large (< 500M)', '^'),
    ('xlarge (> 500M)', 's')
])

marker_map_arch_group = collections.OrderedDict([
    ('ResNet', 'o'),
    ('EfficientNet', 'P'),
    ('CNN', '*'),
    ('RegNet', 'p'),
    ('ConvNeXt', 'X'),

    ('Hybrid', '2'),
    ('MetaFormer', '1'),
    ('MLP', '4'),

    ('ViT', '^'),
    ('XCiT', '<'),
    ('MaxViT', '>'),
    ('Swin', 'd'),
    ('EVA', 'v'),
    ('BEIT', 'D'),
])

marker_map_arch_family = collections.OrderedDict([
    ('mlp', 'o'),
    ('cnn', 'd'),
    ('transformer', '^'),
    ('hybrid', 's')
])

marker_map_pretrain_method = collections.OrderedDict([
    ('Standard', 'o'),
    ('Semi-Supervised Learning', 's'),
    ('Self-Supervised Learning', '*'),
    ('Distillation', 'd'),
    ('CLIP Training', 'P'),
    ('Adversarial Training', 'X')
])


def plot_graphs(
        # The individual dataframes in the list are expected to be passed in a consistent order
        results_dfs: Dict[str, pd.DataFrame],
        groups: List[str],  # List of groups to plot, amongst 'organism', 'artifact', 'other', 'all'
        x_axis_column: str,
        y_axis_column: str,
        y_axis_computation: str,  # Options: abs, relative_to_group_[mle|top1_err], relative_to_all_[mle|top1_err], ...
        ylabel: str,

        marker_column: str = 'arch_family',  # Other options: training_data(_group), arch_group, pretrain_method
        color_column: str = 'training_data_size',

        filter: Optional[Dict[str, Tuple[str, List[str]]]] = None,

        results_df_all: Optional[pd.DataFrame] = None,
        convert_ratio_to_percentage: bool = True,

        linear_fit: bool = False,
        linear_fit_inflection_points: Optional[Dict[str, Optional[float]]] = None,
        linear_fit_by_column: Optional[str] = None,

        figsize: Tuple[float, float] = (10, 8),  # (width, height) in inches
        xy_label_fontsize=40,
        marker_size=35,
        marker_alpha=0.5,  # The alpha blending value, between 0 (transparent) and 1 (opaque).

        xlim=(60, 100), xticks=np.arange(60, 100.01, 20),
        ylim=(-2, 100), yticks=np.arange(0, 100.01, 20),

        add_legend: bool = True,
        legend_bbox_to_anchor=(0.11, 0.88),
        legend_loc='upper left',

        text_annotations: List[Tuple[float, float, str]] = (),

        dataset: Optional[str] = None,

        add_color_bars: bool = False,
        color_column_title: Optional[str] = None,

        show_plt: bool = True
):
    assert len(results_dfs) > 0
    assert all(k in results_dfs and k in COLORMAPS for k in groups)

    assert marker_column in ['training_data', 'training_data_group', 'arch_group', 'arch_family', 'pretrain_method']
    assert color_column in ['training_data_size', "num_params"]

    assert y_axis_computation in [
        'abs',
        'relative_to_group_mle',
        'relative_to_group_top1_err',
        'relative_to_all_mle',
        'relative_to_all_top1_err',
        'relative_to_num_group_samples'
    ]

    assert dataset is not None and dataset in ['imagenet', 'imagenet-a']

    # Filter rows:
    if filter is not None:
        results_dfs = results_dfs.copy()
        for group_id in results_dfs.keys():
            results_dfs[group_id] = filter_dataframe(
                results_dfs[group_id].copy(),
                list_of_columns=list(filter.keys()),
                list_of_operatorts=[x[0] for x in filter.values()],
                list_of_references=[x[1] for x in filter.values()]
            )

    # Defaults:
    x_scale = 'linear'
    c_scale = 'log'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Set marker paths:
    if marker_column == 'training_data':
        marker_map = marker_map_datasets
    elif marker_column == 'training_data_group':
        marker_map = marker_map_data_group
    elif marker_column == 'arch_group':
        marker_map = marker_map_arch_group
    elif marker_column == 'arch_family':
        marker_map = marker_map_arch_family
    elif marker_column == 'pretrain_method':
        marker_map = marker_map_pretrain_method
    else:
        raise ValueError(f'Marker column {marker_column} is not supported')
    markers_list = next(iter(results_dfs.values()))[marker_column].map(marker_map).tolist()
    paths = []
    for marker in markers_list:
        marker_obj = mmarkers.MarkerStyle(marker)
        path = marker_obj.get_path().transformed(marker_obj.get_transform())
        paths.append(path)

    # Compute y's:
    # model_idxs = (results_dfs["all"]["arch"] == "convnext").__or__(results_dfs["all"]["arch"] == "vit")
    # for k in results_dfs.keys():
    #     results_dfs[k] = results_dfs[k][model_idxs]

    # y_a = results_df_group_a[y_axis_column]
    # y_b = results_df_group_b[y_axis_column]
    # ...
    ys = {group_id: results_dfs[group_id][y_axis_column] for group_id in groups}

    if y_axis_computation == 'abs':
        pass
        # The rest of the cases assume that the y_axis_column column refers to an absolute number of errors:
    elif y_axis_computation == 'relative_to_group_mle':
        # Relative to the number of multi-label errors in the group:
        # y_a = y_a / (results_df_group_a[f'num_group_top1_errors']
        #              - results_df_group_a['num_collapsed_mappings']
        #              - results_df_group_a['num_ambiguous'])
        for group_id in groups:
            ys[group_id] = ys[group_id] / (results_dfs[group_id]['num_group_top1_errors']
                                           - results_dfs[group_id]['num_collapsed_mappings']
                                           - results_dfs[group_id]['num_ambiguous'])
    elif y_axis_computation == 'relative_to_group_top1_err':
        # Relative to the number of all (top-1) errors in the group:
        # y_a = y_a / results_df_group_a['num_group_top1_errors']
        for group_id in groups:
            ys[group_id] = ys[group_id] / results_dfs[group_id]['num_group_top1_errors']
    elif y_axis_computation == 'relative_to_all_mle':
        # Relative to the number of ALL multi-label errors:
        assert results_df_all is not None
        for group_id in groups:
            assert (results_dfs[group_id]['num_all_top1_errors'] == results_df_all['num_group_top1_errors']).all()
            ys[group_id] = ys[group_id] / (results_df_all['num_group_top1_errors']
                                           - results_df_all['num_collapsed_mappings']
                                           - results_df_all['num_ambiguous'])
    elif y_axis_computation == 'relative_to_all_top1_err':
        # Relative to the number of ALL top-1 errors:
        assert results_df_all is not None
        for group_id in groups:
            assert (results_dfs[group_id]['num_all_top1_errors'] == results_df_all['num_group_top1_errors']).all()
            ys[group_id] = ys[group_id] / results_df_all['num_group_top1_errors']
    elif y_axis_computation == 'relative_to_num_group_samples':
        for group_id in groups:
            ys[group_id] = ys[group_id] / results_dfs[group_id]['num_group_samples']
    else:
        # TODO: Plot MLE - Minor (we already have MLE - Explainable = 'num_not_classified' covered)
        raise ValueError(f'y_axis_computation={y_axis_computation} mode not supported')

    if y_axis_computation != 'abs' and convert_ratio_to_percentage:
        for group_id in groups:
            ys[group_id] = 100.0 * ys[group_id]

    print(f'Max ys: {[round(max(yi), 2) for yi in ys.values()]}')
    print(f'Min ys: {[round(min(yi), 2) for yi in ys.values()]}')

    vmin = min(results_dfs[group_id][color_column].min() for group_id in groups)
    vmax = max(results_dfs[group_id][color_column].max() for group_id in groups)

    if linear_fit:
        for group_id in groups:
            print(f'Plot linear fit for group: {group_id}')
            if linear_fit_inflection_points is not None:
                if group_id in linear_fit_inflection_points and linear_fit_inflection_points[group_id] is not None:
                    x_sep = linear_fit_inflection_points[group_id]
                    sep_idx = np.array(results_dfs[group_id][x_axis_column]) <= x_sep
                else:
                    sep_idx = None
            else:
                sep_idx = None
            if linear_fit_by_column is not None:
                unique_column_values = results_dfs[group_id][linear_fit_by_column].unique().tolist()
                alpha = 0.10
                for column_value in unique_column_values:
                    alpha = alpha * 1.4
                    print(f'Plot linear fit for column value: {column_value}')
                    column_idx = results_dfs[group_id][linear_fit_by_column] == column_value
                    idx = column_idx if sep_idx is None else sep_idx.__and__(column_idx)
                    if sum(idx) > 1:
                        plot_conf(results_dfs[group_id][x_axis_column], ys[group_id], idx=idx, alpha=alpha)
                    if sep_idx is not None:
                        idx = (~sep_idx).__and__(column_idx)
                        if sum(idx) > 1:
                            plot_conf(
                                results_dfs[group_id][x_axis_column], ys[group_id],
                                idx=(~sep_idx).__and__(column_idx),
                                alpha=alpha
                            )
            else:
                if sep_idx is None or sum(sep_idx) > 1:
                    plot_conf(results_dfs[group_id][x_axis_column], ys[group_id], idx=sep_idx)
                if sep_idx is not None and sum(~sep_idx) > 1:
                    plot_conf(results_dfs[group_id][x_axis_column], ys[group_id], idx=~sep_idx)

    # Create plots:
    plots = []
    for group_id in groups:
        plots.append(ax.scatter(
            results_dfs[group_id][x_axis_column], ys[group_id],
            c=results_dfs[group_id][color_column], s=marker_size, alpha=marker_alpha,
            vmin=vmin, vmax=vmax, norm=c_scale,
            cmap=COLORMAPS[group_id]
        ))
    for plot in plots:
        plot.set_paths(paths)

    ax.set_xscale(x_scale)  # 'linear'

    # Set up x/y labels/titles and x/y limits:
    if 'mla' in x_axis_column:
        xlabel = 'MLA [\%]'
    else:
        xlabel = 'Top-1 Accuracy [\%]'
    if x_axis_column.startswith('imagenet_a_'):
        xlabel = 'ImageNet-A ' + xlabel
    elif x_axis_column.startswith('imagenet_'):
        xlabel = 'ImageNet ' + xlabel
    ax.set_xlabel(xlabel, fontsize=xy_label_fontsize)

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)

    ax.set_ylabel(ylabel, fontsize=xy_label_fontsize, rotation=0, ha="left")
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    # Set up legend:
    if add_legend:
        legend_handles = []
        for k, v in marker_map.items():
            if v not in markers_list: continue
            k = "".join([f"${c}$" if c in ["<", ">"] else c for c in k])
            legend_handles.append(
                mlines.Line2D([], [], color='black', marker=v, linestyle='None', markersize=10, label=rf"{k}")
            )
        fig.legend(
            handles=legend_handles,
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
            frameon=False,
            fontsize=0.8*xy_label_fontsize,
            handletextpad=0.3
        )

    # Text annotations:
    for text_x, text_y, text_s in text_annotations:
        ax.text(text_x, text_y, text_s, fontsize=xy_label_fontsize)

    if add_color_bars:
        # TODO: add color bars
        # fig.subplots_adjust(right=0.88)
        # cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.50])
        # cbar_ax.set_title(color_column_label)
        # cbar = fig.colorbar(plot_a, cax=cbar_ax)
        # cbar.set_ticks([])
        # cbar_ax = fig.add_axes([0.96, 0.35, 0.02, 0.50])
        # cbar = fig.colorbar(plot_b, cax=cbar_ax)
        # cbar_ax.tick_params(labelsize=fontsize)
        pass

    rect = fig.patch
    rect.set_facecolor("white")

    ax.tick_params(labelsize=xy_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_facecolor((0.98, 0.98, 0.98))
    ax.yaxis.set_label_coords(0.0, 1.02)
    ax.grid(False)

    # plt.subplots_adjust(left=0.5, right=0.5)

    if not os.path.isdir(f'../{FIGURES_FOLDER}'):
        os.mkdir(f'../{FIGURES_FOLDER}')
    if not os.path.isdir(f'../{FIGURES_FOLDER}/{dataset}'):
        os.mkdir(f'../{FIGURES_FOLDER}/{dataset}')
    fig_filename = '../{}/{}/x_axis={}_y_axis={}_{}_marker={}_color={}_groups={}'.format(
        FIGURES_FOLDER, dataset,
        x_axis_column, y_axis_column, y_axis_computation,
        marker_column, color_column, '_'.join(groups)
    )
    print('Figure file name:', fig_filename)

    plt.savefig(f'{fig_filename}.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{fig_filename}.png', bbox_inches='tight', pad_inches=0.3)

    if show_plt:
        plt.show()


# ImageNet plots

def plot_class_overlap(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'top1_val_acc'
    y_axis_column = 'num_collapsed_mappings'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to top-1 accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[-0.2, 10], yticks=np.arange(0, 10 + 0.01, 5),
            add_legend=True,
            # legend_bbox_to_anchor=(0.11, 0.9),
            # legend_loc='upper left',
            dataset=dataset,
            **kwargs,
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Correct Class Overlap Top-1 Errors",
            ylabel="\# Top-1 Errors",

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[-2, 100], yticks=np.arange(0, 100 + 0.01, 50),
            add_legend=False,
            text_annotations=[
                (67, 39, "organisms"),
                (65, 83, "artifacts"),
                (65, 10, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to top-1 accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_top1_err',  # Equivalent to 'relative_to_all_top1_err' in this case
        ylabel="Portion of All Top-1 Errors [\%]",

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[0, 10], yticks=np.arange(0, 10 + 0.01, 5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Correct Class Overlap Top-1 Errors",
        ylabel="\# Top-1 Errors",

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[0, 150], yticks=np.arange(0, 150 + 0.01, 50),
        add_legend=False,
        dataset=dataset,
        **kwargs
    )


def plot_missing_multi_labels(results_dfs, one_only=True, **kwargs):
    x_axis_column = 'top1_val_acc'
    y_axis_column = 'num_ambiguous'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to top-1 accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[-1.5, 85], yticks=np.arange(0, 85 + 0.01, 20),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (80, 18, "organisms"),
                (61, 25, "artifacts"),
                (73, 54, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Missing Multi-Labels Top-1 Errors",
            ylabel="\# Top-1 Errors",

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[-20, 1000], yticks=np.arange(0, 1000 + 0.01, 500),
            add_legend=False,
            text_annotations=[
                (65, 270, "organisms"),
                (70, 850, "artifacts"),
                (69, 100, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to top-1 accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_top1_err',  # Equivalent to 'relative_to_all_top1_err' in this case
        ylabel="Portion of All Top-1 Errors [\%]",

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[0, 65], yticks=np.arange(0, 65 + 0.01, 20),
        add_legend=True,
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Missing Multi-Labels Top-1 Errors",
        ylabel="\# Top-1 Errors",

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[-20, 1150], yticks=np.arange(0, 1150 + 0.01, 250),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.59),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_fine_grained(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_same_superclass'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Portion of Group's Multi-Label Errors [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-2, 90], yticks=np.arange(0, 90 + 0.01, 30),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (80, 82, "organisms"),
                (90, 25, "artifacts"),
                (84, 49, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Fine-Grained Multi-Label Errors",
            ylabel="\# Multi-Label Errors",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-20, 950], yticks=np.arange(0, 950 + 0.01, 200),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.70),
            legend_loc='upper left',
            text_annotations=[
                (78, 300, "organisms"),
                (87, 550, "artifacts"),
                (86, 130, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-1.5, 70], yticks=np.arange(0, 80 + 0.01, 20),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Fine-Grained Multi-Label Errors",
        ylabel="\# Multi-Label Errors",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-30, 1800], yticks=np.arange(0, 1800 + 0.01, 500),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.58),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_fine_grained_OOV(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_OOV_clip'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Portion of Group's Multi-Label Errors [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-1, 30], yticks=np.arange(0, 30 + 0.01, 5),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (74, 0, "organisms"),
                (78, 17, "artifacts+others") if len(groups) == 3 else (80, 15, "artifacts"),
            ],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Fine-Grained OOV Multi-Label Errors",
            ylabel="\# Multi-Label Errors",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-4, 240], yticks=np.arange(0, 240 + 0.01, 60),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.65),
            legend_loc='upper left',
            text_annotations=[
                (82, 50, "organisms"),
                (90.5, 180, "artifacts"),
                (75, 5, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-0.5, 15], yticks=np.arange(0, 15 + 0.01, 5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Fine-Grained OOV Multi-Label Errors",
        ylabel="\# Multi-Label Errors",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-6, 300], yticks=np.arange(0, 300 + 0.01, 50),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_non_proto(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_non_prototypical'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Portion of Group's Multi-Label Errors [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-0.05, 2.5], yticks=np.arange(0, 2.5 + 0.01, 1),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Non-prototypical Multi-Label Errors",
            ylabel="\# Multi-Label Errors",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-0.5, 15], yticks=np.arange(0, 15 + 0.01, 5),
            add_legend=False,
            text_annotations=[
                (62, 1.5, "organisms"),
                (78, 12, "artifacts"),
                (67, 1.5, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-0.04, 2], yticks=np.arange(0, 2 + 0.01, 1),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc="upper left",
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Non-prototypical Multi-Label Errors",
        ylabel="\# Multi-Label Errors",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-0.4, 20], yticks=np.arange(0, 20 + 0.01, 5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_common_co_occ(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_common_co_occ'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Portion of Group's Multi-Label Errors [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-0.4, 22], yticks=np.arange(0, 20 + 0.01, 5),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (85, 4.1, "organisms"),
                (80, 10.5, "artifacts"),
                (85, 17.5, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            # ylabel="\# Spurious Correlation Multi-Label Errors",
            ylabel="\# Multi-Label Errors",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-4, 200], yticks=np.arange(0, 200 + 0.01, 50),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.74),
            legend_loc='upper left',
            text_annotations=[
                (67, 8, "organisms"),
                (88, 150, "artifacts"),
                (82, 42, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel='Portion of All Multi-Label Errors [\%]',

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-0.2, 10], yticks=np.arange(0, 10 + 0.01, 5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        # ylabel="\# Spurious Correlation Multi-Label Errors",
        ylabel="\# Top-1 Errors",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-5, 250], yticks=np.arange(0, 250 + 0.01, 50),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_model_failures(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_not_classified'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Portion of Group's Multi-Label Errors [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-1.5, 75], yticks=np.arange(0, 75 + 0.01, 25),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.6),
            legend_loc='upper left',
            text_annotations=[
                (83, 5, "organisms"),
                (90.5, 55, "artifacts"),
                (98, 15, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Model Failures",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-60, 3000], yticks=np.arange(0, 3000 + 0.01, 1000),
            add_legend=True,
            legend_bbox_to_anchor=(0.50, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (65, 350, "organisms"),
                (74, 950, "artifacts"),
                (65, 400, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-1.5, 70], yticks=np.arange(0, 70 + 0.01, 20),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Model Failures",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-90, 4500], yticks=np.arange(0, 4500 + 0.01, 2000),
        add_legend=True,
        legend_bbox_to_anchor=(0.65, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_model_failures_to_mle(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'mla_all'
    y_axis_column = 'num_not_classified'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to group's multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Multi-Label Failures (MLF) / Group's Multi-Label Errors (MLE)",

            convert_ratio_to_percentage=True,
            # linear_fit_inflection_points=None if len(groups) == 3 else {'organism': 90, 'artifact': 90},

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-0.01, 75], yticks=np.arange(0, 75 + 0.01, 25),
            add_legend=False,
            text_annotations=[
                (72, 25, "organisms"),
                (91, 54, "artifacts"),
                (98, 15, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, relative to all multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_all_mle',
            ylabel="Multi-Label Failures (MLF) / All Multi-Label Errors (MLE)",

            results_df_all=results_dfs['all'],
            convert_ratio_to_percentage=True,
            linear_fit_inflection_points={'organism': None, 'artifact': 90, 'other': None},

            xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
            ylim=[-1, 50], yticks=np.arange(0, 50 + 0.01, 25),
            add_legend=False,
            text_annotations=[
                (75, 16.5, "organisms"),
                (90, 37, "artifacts"),
                (68, 7.5, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to all multi-label errors:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_all_mle',
        ylabel="\# Multi-Label Failures (MLF) / All Multi-Label Errors (MLE)",

        results_df_all=results_dfs['all'],
        convert_ratio_to_percentage=True,
        linear_fit_inflection_points={'all': 90},

        xlim=[60, 100], xticks=np.arange(60, 100.01, 20),
        ylim=[-1, 75], yticks=np.arange(0, 75 + 0.01, 25),
        add_legend=False,
        dataset=dataset,
        **kwargs
    )


def plot_model_failures_to_top1_err(results_dfs, one_only=False, **kwargs):
    x_axis_column = 'top1_val_acc'
    y_axis_column = 'num_not_classified'
    dataset = 'imagenet'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to group's top-1 errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors",

            convert_ratio_to_percentage=True,
            # linear_fit_inflection_points=None if len(groups) == 3 else {'organism': 80, 'artifact': 80},

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[1.5, 75], yticks=np.arange(0, 75 + 0.01, 25),
            add_legend=False,
            text_annotations=[
                (58, 25, "organisms") if len(groups) == 2 else (50.75, 54, "organisms"),
                (80, 35, "artifacts"),
                (71, 15, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )
        if one_only:
            return

        # Per group, relative to all multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_all_top1_err',
            ylabel="Portion of Group's Top-1 Errors",

            results_df_all=results_dfs['all'],
            convert_ratio_to_percentage=True,
            linear_fit_inflection_points={'organism': None, 'artifact': 80, 'other': None},

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[0, 50], yticks=np.arange(0, 50 + 0.01, 25),
            add_legend=False,
            text_annotations=[
                (68, 14, "organisms"),
                (76, 28, "artifacts"),
                (60, 6, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to all multi-label errors:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_all_top1_err',
        ylabel="\# Multi-Label Failures (MLF) / All Top-1 Errors",

        results_df_all=results_dfs['all'],
        convert_ratio_to_percentage=False,
        linear_fit_inflection_points={'all': 80},

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[0, 0.75], yticks=np.arange(0, 0.75 + 0.01, 0.25),
        add_legend=False,
        dataset=dataset,
        **kwargs
    )


def plot_all(results_dfs, **kwargs):
    print("Class Overlap")
    plot_class_overlap(results_dfs, one_only=True, **kwargs)

    print("Missing Multi Label Annotations")
    plot_missing_multi_labels(results_dfs, one_only=True, **kwargs)

    print("Fine-Grained Errors")
    plot_fine_grained(results_dfs, one_only=True, **kwargs)

    print("Fine-Grained OOV Errors")
    plot_fine_grained_OOV(results_dfs, one_only=True, **kwargs)

    print("Non Prototypical Samples")
    plot_non_proto(results_dfs, one_only=True, **kwargs)

    print("Spurious Correlation")
    plot_common_co_occ(results_dfs, one_only=True, **kwargs)

    print("Model Failures")
    plot_model_failures(results_dfs, one_only=True, **kwargs)


# ImageNet-A plots


def plot_imagenet_a_vs_imagenet_top1_acc(results_dfs, **kwargs):
    x_axis_column = 'imagenet_top1_val_acc'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # ImageNet-A accuracy per group:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column='num_group_correct',
            y_axis_computation='relative_to_num_group_samples',
            ylabel="Group's ImageNet-A Top-1 Acc [\%]",

            xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
            ylim=[0, 80], yticks=np.arange(0, 80 + 0.01, 20),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            dataset=dataset,
            **kwargs,
        )

    print('=== ALL ===')

    # ImageNet-A accurcy on all samples:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column='num_group_correct',
        y_axis_computation='relative_to_num_group_samples',
        ylabel="ImageNet-A Top-1 Acc [\%]",

        xlim=[50, 95], xticks=np.arange(50, 95.01, 15),
        ylim=[0, 80], yticks=np.arange(0, 80 + 0.01, 20),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_imagenet_a_vs_imagenet_mla_acc(results_dfs, **kwargs):
    x_axis_column = 'imagenet_mla_val_acc'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # ImageNet-A accuracy per group:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            #x_axis_column='num_group_correct',
            y_axis_column='num_group_correct',
            y_axis_computation='relative_to_num_group_samples',
            ylabel="Group's ImageNet-A Top-1 Acc [\%]",

            xlim=[60, 100], xticks=np.arange(60, 100.01, 10),
            ylim=[0, 80], yticks=np.arange(0, 80 + 0.01, 20),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            dataset=dataset,
            **kwargs,
        )

    print('=== ALL ===')

    # ImageNet-A accurcy on all samples:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column='num_group_correct',
        y_axis_computation='relative_to_num_group_samples',
        ylabel="ImageNet-A Top-1 Acc [\%]",

        xlim=[60, 100], xticks=np.arange(60, 100.01, 10),
        ylim=[0, 80], yticks=np.arange(0, 80 + 0.01, 20),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_class_overlap_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100. * results_dfs['all']['num_group_correct'] / 7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_top1_val_acc'
        xlim = [50, 95]
        xticks = np.arange(50, 95.01, 15)
    else:
        assert False
    y_axis_column = 'num_collapsed_mappings'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')
    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP, ORGANISM_GROUP]:
        # Per group, relative to top-1 accuracy:

        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",
            xlim=xlim, xticks=xticks,
            ylim=[0, 2], yticks=np.arange(0, 2 + 0.01, 1),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            dataset=dataset,
            # text_annotations=[
            #     (72, 0.26, "organisms")
            # ],
            **kwargs,
        )

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Correct Class Overlap Top-1 Errors",
            xlim=xlim, xticks=xticks,
            ylim=[0, 25], yticks=np.arange(0, 25 + 0.01, 5),
            add_legend=False,  # True,
            legend_bbox_to_anchor=(0.932, 0.095),
            legend_loc='lower right',
            # text_annotations=[
            #     (71, 9.5, "organisms"),
            # ],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to top-1 accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_top1_err',  # Equivalent to 'relative_to_all_top1_err' in this case
        ylabel="Portion of All Top-1 Errors [\%]",
        xlim=xlim, xticks=xticks,
        ylim=[0, 1], yticks=np.arange(0, 1 + 0.01, 0.5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Correct Class Overlap Top-1 Errors",
        xlim=xlim, xticks=xticks,
        ylim=[0, 25], yticks=np.arange(0, 25 + 0.01, 5),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_fine_grained_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_same_superclass'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",
            xlim=xlim, xticks=xticks,
            ylim=[0, 50], yticks=np.arange(0, 50 + 0.01, 10),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            text_annotations=[
                (50, 30, "organisms"),
                (50, 6, "artifacts"),
                (55, 23.5, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Fine-Grained Top-1 Errors",
            xlim=xlim, xticks=xticks,
            ylim=[0, 900], yticks=np.arange(0, 900 + 0.01, 200),
            add_legend=False,
            # legend_bbox_to_anchor=(0.11, 0.9),
            # legend_loc='upper left',
            text_annotations=[
                (20, 600, "organisms"),
                (40, 224, "artifacts"),
                (20, 70, "others"),
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",
        xlim=xlim, xticks=xticks,
        ylim=[0, 35], yticks=np.arange(0, 35 + 0.01, 10),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Fine-Grained Multi-Label Errors",

        xlim=xlim, xticks=xticks,
        ylim=[0, 1100], yticks=np.arange(0, 1100 + 0.01, 250),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_fine_grained_OOV_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_OOV_clip'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",

            xlim=xlim, xticks=xticks,
            ylim=[0, 45], yticks=np.arange(0, 45 + 0.01, 10),
            add_legend=False,  # True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            # text_annotations=[
            #     (90, 5, "organisms"),
            #     (92, 15.5, "artifacts"),
            #     (83, 31, "others"),
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Fine-Grained OOV Top-1 Errors",

            xlim=xlim, xticks=xticks,
            ylim=[0, 400], yticks=np.arange(0, 400 + 0.01, 100),
            add_legend=False,
            # text_annotations=[
            #     (80, 215, "organisms"),
            #     (76, 367, "artifacts"),
            #     (78, 90, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Multi-Label Errors [\%]",

        xlim=xlim, xticks=xticks,
        ylim=[0, 25], yticks=np.arange(0, 25 + 0.01, 10),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Fine-Grained OOV Multi-Label Errors",

        xlim=xlim, xticks=xticks,
        ylim=[0, 1000], yticks=np.arange(0, 1000 + 0.01, 250),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.58),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_common_co_occ_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_common_co_occ'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",

            xlim=xlim, xticks=xticks,
            ylim=[0, 7], yticks=np.arange(0, 7 + 0.01, 2),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            # text_annotations=[
            #     (94, 0.05, "organisms") if len(groups) == 3 else (82, 0.5, "organisms"),
            #     (85, 3.5, "artifacts"),
            #     (96, 2.7, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Spurious Correlation Top-1 Errors",

            xlim=xlim, xticks=xticks,
            ylim=[0, 75], yticks=np.arange(0, 75 + 0.01, 25),
            add_legend=False,
            legend_bbox_to_anchor=(0.11, 0.9),
            legend_loc='upper left',
            # text_annotations=[
            #     (85, 17.2, "organisms"),
            #     (81, 51, "artifacts"),
            #     (97.2, 5, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_mle',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel='Portion of All Multi-Label Errors [\%]',

        xlim=xlim, xticks=xticks,
        ylim=[0, 3.5], yticks=np.arange(0, 3.5 + 0.01, 1),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Spurious Correlation Multi-Label Errors",

        xlim=xlim, xticks=xticks,
        ylim=[0, 100], yticks=np.arange(0, 100 + 0.01, 25),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.9),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_model_failures_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_not_classified'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to multi-label accuracy:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Portion of Group's Top-1 Errors [\%]",

            xlim=xlim, xticks=xticks,
            ylim=[0, 100], yticks=np.arange(0, 100 + 0.01, 25),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.6),
            legend_loc='upper left',
            # text_annotations=[
            #     (82, 90, "organisms"),
            #     (75, 70, "artifacts") if len(groups) == 2 else (70, 82, "artifacts"),
            #     (80, 60, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, absolute number:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='abs',
            ylabel="\# Model Failures",

            xlim=xlim, xticks=xticks,
            ylim=[0, 5000], yticks=np.arange(0, 5000 + 0.01, 1000),
            add_legend=False,
            # text_annotations=[
            #     (80, 4300, "organisms"),
            #     (78, 1900, "artifacts"),
            #     (75, 475, "others"),
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to multi-label accuracy:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_group_top1_err',  # Equivalent to 'relative_to_all_mle' in this case
        ylabel="Portion of All Top-1 Errors [\%]",

        xlim=xlim, xticks=xticks,
        ylim=[0, 90], yticks=np.arange(0, 90 + 0.01, 30),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )

    # All, absolute number:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='abs',
        ylabel="\# Model Failures",

        xlim=xlim, xticks=xticks,
        ylim=[0, 7000], yticks=np.arange(0, 7000 + 0.01, 2000),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_model_failures_to_mle_imagenet_a(results_dfs,  base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_not_classified'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to group's multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_mle',
            ylabel="Multi-Label Failures (MLF) / Group's Multi-Label Errors (MLE)",

            convert_ratio_to_percentage=False,
            linear_fit_inflection_points=None if len(groups) == 3 else {'organism': 90, 'artifact': 90},

            xlim=xlim, xticks=xticks,
            ylim=[0, 1], yticks=np.arange(0, 1 + 0.01, 0.25),
            add_legend=True,
            legend_bbox_to_anchor=(0.11, 0.6),
            legend_loc='upper left',
            text_annotations=[
                (82, 0.90, "organisms"),
                (75, 0.70, "artifacts") if len(groups) == 2 else (70, 0.82, "artifacts"),
                (80, 0.60, "others")
            ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, relative to all multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_all_mle',
            ylabel="Multi-Label Failures (MLF) / All Multi-Label Errors (MLE)",

            results_df_all=results_dfs['all'],
            convert_ratio_to_percentage=False,
            linear_fit_inflection_points={'organism': 90, 'artifact': None, 'other': None},

            xlim=xlim, xticks=xticks,
            ylim=[0, 0.6], yticks=np.arange(0, 0.6 + 0.01, 0.2),
            add_legend=False,
            # text_annotations=[
            #     (83, 0.45, "organisms"),
            #     (81, 0.19, "artifacts"),
            #     (75, 0.07, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to all multi-label errors:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_all_mle',
        ylabel="\# Multi-Label Failures (MLF) / All Multi-Label Errors (MLE)",

        results_df_all=results_dfs['all'],
        convert_ratio_to_percentage=False,
        linear_fit_inflection_points={'all': 90},

        xlim=xlim, xticks=xticks,
        ylim=[0, 1], yticks=np.arange(0, 1 + 0.01, 0.25),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


def plot_model_failures_to_top1_err_imagenet_a(results_dfs, base='imagenet-a', **kwargs):
    if base == 'imagenet-a':
        x_axis_column = 'imagenet_a_top1_val_acc'
        acc = 100.*results_dfs['all']['num_group_correct']/7500.
        for key in results_dfs.keys():
            results_dfs[key]['imagenet_a_top1_val_acc'] = acc
        xlim = [0, 70]
        xticks = np.arange(0, 70.01, 10)
    elif base == 'imagenet':
        x_axis_column = 'imagenet_mla_val_acc'
        xlim = [60, 100]
        xticks = np.arange(60, 100.01, 20)
    else:
        assert False

    y_axis_column = 'num_not_classified'
    dataset = 'imagenet-a'

    print('=== PER GROUP ===')

    for groups in [ORGANISM_ARTIFACT_GROUP, ORGANISM_ARTIFACT_OTHER_GROUP]:
        # Per group, relative to group's top-1 errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_group_top1_err',
            ylabel="Multi-Label Failures (MLF) / Group's Top-1 Errors",

            convert_ratio_to_percentage=False,
            linear_fit_inflection_points=None if len(groups) == 3 else {'organism': 80, 'artifact': 80},

            xlim=xlim, xticks=xticks,
            ylim=[0, 1], yticks=np.arange(0, 1 + 0.01, 0.25),
            add_legend=False,
            # text_annotations=[
            #     (73, 0.90, "organisms"),
            #     (65, 0.70, "artifacts") if len(groups) == 2 else (61, 0.82, "artifacts"),
            #     (70, 0.60, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

        # Per group, relative to all multi-label errors:
        plot_graphs(
            results_dfs=results_dfs,
            groups=groups,
            x_axis_column=x_axis_column,
            y_axis_column=y_axis_column,
            y_axis_computation='relative_to_all_top1_err',
            ylabel="Multi-Label Failures (MLF) / All Top-1 Errors",

            results_df_all=results_dfs['all'],
            convert_ratio_to_percentage=False,
            linear_fit_inflection_points={'organism': 80, 'artifact': None, 'other': None},


            xlim=xlim, xticks=xticks,
            ylim=[0, 0.6], yticks=np.arange(0, 0.6 + 0.01, 0.2),
            add_legend=False,
            # text_annotations=[
            #     (73, 0.45, "organisms"),
            #     (71, 0.19, "artifacts"),
            #     (65, 0.07, "others")
            # ][:len(groups)],
            dataset=dataset,
            **kwargs
        )

    print('=== ALL ===')

    # All, relative to all multi-label errors:
    plot_graphs(
        results_dfs=results_dfs,
        groups=ALL_GROUP,
        x_axis_column=x_axis_column,
        y_axis_column=y_axis_column,
        y_axis_computation='relative_to_all_top1_err',
        ylabel="\# Multi-Label Failures (MLF) / All Top-1 Errors",

        results_df_all=results_dfs['all'],
        convert_ratio_to_percentage=False,
        linear_fit_inflection_points={'all': 80},

        xlim=xlim, xticks=xticks,
        ylim=[0, 1], yticks=np.arange(0, 1 + 0.01, 0.25),
        add_legend=True,
        legend_bbox_to_anchor=(0.11, 0.6),
        legend_loc='upper left',
        dataset=dataset,
        **kwargs
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'imagenet-a'])
    args = parser.parse_args()

    # Read and prepare stats dataframes:
    results_dfs = read_and_prepare_stats_dfs(args.dataset)

    # Generate and save all plots
    if args.dataset == 'imagenet':
        # ImageNet
        logger.info('Class Overlap (Collapsed Mappings)')
        plot_class_overlap(results_dfs, show_plt=False)

        logger.info('Missing Multi-Label Annotations')
        plot_missing_multi_labels(results_dfs, show_plt=False)

        logger.info('Fine-Grained Errors')
        plot_fine_grained(results_dfs, show_plt=False)

        logger.info('Fine-Grained OOV Errors')
        plot_fine_grained_OOV(results_dfs, show_plt=False)

        logger.info('Non-prototypical')
        plot_non_proto(results_dfs, show_plt=False)

        logger.info('Common co-occurrences (Spurious Correlations)')
        plot_common_co_occ(results_dfs, show_plt=False)

        logger.info('Model Failures (Major Errors)')
        plot_model_failures(results_dfs, show_plt=False)

        logger.info('Multi-Label Model Failures (MLF) / Multi-Label Errors (MLE) (per group and all)')
        plot_model_failures_to_mle(results_dfs, show_plt=False)

        logger.info('Multi-Label Model Failures (MLF) / Top-1 Errors (per group and all)')
        plot_model_failures_to_top1_err(results_dfs, show_plt=False)
    elif args.dataset == 'imagenet-a':
        # ImageNet-A
        # Multi-label accuracy = Top-1 accuracy for ImageNet-A
        # The set of multi labels for each sample is simply the sample's target
        logger.info('ImageNet vs ImageNet-A accuracies')
        plot_imagenet_a_vs_imagenet_top1_acc(results_dfs, show_plt=False)
        plot_imagenet_a_vs_imagenet_mla_acc(results_dfs, show_plt=False)

        logger.info('Class Overlap (Collapsed Mappings)')
        plot_class_overlap_imagenet_a(results_dfs, show_plt=False)

        logger.info('Fine-Grained Errors')
        plot_fine_grained_imagenet_a(results_dfs, show_plt=False)

        logger.info('Fine-Grained OOV Errors')
        plot_fine_grained_OOV_imagenet_a(results_dfs, show_plt=False)

        logger.info('Common co-occurrences (Spurious Correlations)')
        plot_common_co_occ_imagenet_a(results_dfs, show_plt=False)

        logger.info('Model Failures (Major Errors)')
        plot_model_failures_imagenet_a(results_dfs, show_plt=False)

        logger.info('Multi-Label Model Failures (MLF) / Multi-Label Errors (MLE) (per group and all)')
        plot_model_failures_to_mle_imagenet_a(results_dfs, show_plt=False)

        logger.info('Multi-Label Model Failures (MLF) / Top-1 Errors (per group and all)')
        plot_model_failures_to_top1_err_imagenet_a(results_dfs, show_plt=False)
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')
