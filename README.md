# Automated Classification of Model Errors on ImageNet <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

We propose the first automated ImageNet error classification framework
and use it to comprehensively evaluate the error distributions of over 900 models.
We find that across model architectures, scales, and pre-training corpora, top-1
accuracy is a strong predictor for the _portion_ of all error types.

This repository accompanies our
[NeurIPS 2023](https://openreview.net/forum?id=zEoP4vzFKy) paper,
developed at the
[SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch)
as a part of the [Safe AI project](http://safeai.ethz.ch).

## Setup

Clone this repository and then download and extract the project artefacts:
```bash
git clone --recurse-submodules https://github.com/eth-sri/automated-error-analysis.git
cd automated-error-analysis
wget http://files.sri.inf.ethz.ch/imagenet-error-analysis/artefacts.tar.gz
(sha256sum artefacts.tar.gz == efd9b00879fc48c12494cde15477040ed04a1d50b815aec15d33985ffb10adf1)
tar -xvzf artefacts.tar.gz
```

### Dependencies

We use python 3.8 with PyTorch 1.13.1 and CUDA 11.6. Please, use the following commands to set up a correponding [conda](https://docs.conda.io/en/latest/miniconda.html) environment:
```bash
conda create --name error-analysis python=3.8
conda activate error-analysis
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Paths
Please, set the path to your ImageNet and ImageNet-A directories (adapt the command below) and PYTHONPATH as follows:
```bash
export IMAGENET_DIR="/home/path-to/imagenet"
export IMAGENET_A_DIR="/home/path-to/imagenet-a"
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

`IMAGENET_DIR` folder is preprocessed into PyTorch `ImageFolder` format, i.e., has the following structure:\
[`train`|`val`] / ImageNet labels / images with the corresponding label. The ImageNet-A dataset is already in the required structure if you download and extract it from its respective [repository](https://github.com/hendrycks/natural-adv-examples). See [datasets.py](src/datasets.py) for details.

### Multilabel dataset (from tensorflow-datasets)

The multilabel dataset we use requires manual download.
Please follow the following instructions (from [tensorflow-datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_multilabel)):
```
This dataset requires you to download the source data manually into
download_config.manual_dir (defaults to ~/tensorflow_datasets/downloads/manual/):
manual_dir should contain ILSVRC2012_img_val.tar file.

You need to register on http://www.image-net.org/download-images
in order to get the link to download the dataset.
```

## Collect Model Predictions

To collect all model predictions (and the models metadata), please execute:
```bash
python src/collect_predictions.py --dataset imagenet --modules all
python src/collect_predictions.py --dataset imagenet-a --modules all
```
This is expected to take up to 6 days for ImageNet and 1 day for ImageNet-A
(with a single NVIDIA RTX 2080Ti). You can specify which modules to execute if you
wish to analyse only particular model types.

We list a summary of all evaluated models together with their metadata at
[models_summary.csv](models_summary.csv).

## Run the Analysis

Afterwards, to run our analysis, please execute:
```bash
python src/evaluation.py --dataset imagenet --perform_error_analysis --collect_results
python src/evaluation.py --dataset imagenet-a --perform_error_analysis --collect_results
```

This should take around 12 to 24 hours for ImageNet and ImageNet-A respectively and
the summaries of the analyses are written in the [stats-imagenet](stats-imagenet) and
[stats-imagenet-a](stats-imagenet-a) folders.

## Error Analysis Archives

To speed up the process, allowing you to skip the two steps above,
we also provide archives with the error analysis for each model in the
`artefacts` folder. More concretely, we analyse the prediction of
each model on every sample from the validation sets.

```bash
cd artefacts
tar -zxf models-imagenet.tar.gz
tar -zxf models-imagenet-a.tar.gz
```

To produce the summaries in [stats-imagenet](stats-imagenet) and
[stats-imagenet-a](stats-imagenet-a), you still need to run

```bash
python src/evaluation.py --dataset imagenet --collect_results
python src/evaluation.py --dataset imagenet-a --collect_results
```

## Plots

Then, you can reproduce the plots from our paper and examine the results by running the
notebooks [analyse_results_imagenet.ipynb](src/analyse_results_imagenet.ipynb) and
[analyse_results_imagenet-a.ipynb](src/analyse_results_imagenet-a.ipynb) or
the [utils_plot.py](src/utils_plot.py) script.

The generated figures can be found in the [figures](figures) folder.

## Comparison to [Vasudevan et al. (2022)](https://openreview.net/forum?id=mowt1WNhTC7)

The evaluation was performed in the
[dough_bagel_eval.ipynb](src/dough_bagel_eval.ipynb) notebook.

You can also use this notebook as a skeleton to investigate the mistakes of
other models from our collection, provided that you have computed their
predictions as discussed above beforehand.

## Misc

You can show individual images from the datasets and visualize ImageNet classes
in the [show_images.ipynb](src/show_images.ipynb) notebook.

In the `artefacts` folder, the file `superclasses.txt` contains our (manual) label
superclass groupings (produced by the definitions in `superclasses.json`).

The most common erroneous samples from the ImageNet validation set according to
our pipeline can be found in the [common_error_samples](common_error_samples) folder.
They were collected by running
```bash
python src/evaluation.py --list_most_common_errors --error_type [ERROR_TYPE]
```

## Citing This Work

```
@inproceedings{peychev2023automated,
    title={Automated Classification of Model Errors on ImageNet},
    author={Momchil Peychev and Mark Niklas Müller and Marc Fischer and Martin Vechev},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=zEoP4vzFKy}
}
```

## Contributors

* [Momchil Peychev](https://www.sri.inf.ethz.ch/people/momchil)
* [Mark Niklas Müller](https://www.sri.inf.ethz.ch/people/mark)
* [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## License and Copyright

* Licensed under the [Apache-2.0 License](LICENSE)
* Copyright (c) 2023 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch)
