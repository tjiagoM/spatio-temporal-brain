[![GitHub license](https://img.shields.io/github/license/tjiagoM/spatio-temporal-brain)](https://github.com/tjiagoM/spatio-temporal-brain/blob/master/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.0000/...-blue.svg)](https://doi.org)

# A Deep Graph Neural Network Architecture for Spatio-temporal rs-fMRI data

This repository contains an implementation of a deep neural network architecture combining both graph neural networks (GNNs) and temporal convolutional networks (TCNs), which is able to learn from the spatial and temporal components of rs-fMRI data in an end-to-end fashion. Please check the [publications](#publications) at the end of this page for more details on how this architecture was used and evaluated.

If something is not clear or you have any question please [open an Issue](https://github.com/tjiagoM/spatio-temporal-brain/issues).

## Running the experiments

The code in this repository heavily relies on [Weights & Biases](https://www.wandb.com/) (W&B) to keep track and organise the results of experiments. W&B software was responsible to conduct the hyperparameter search, and all the sweeps (needed for hyperparameter search) used are defined in the `wandb_sweeps/` folder. All our runs, sweep definitions and reports are publicly available at our [project's W&B page](https://wandb.ai/st-team/spatio-temporal-brain). In particular, we provide [two reports](https://wandb.ai/st-team/spatio-temporal-brain/reportlist) to organise the main results of our experiments. 

We recommend that a user wanting to run and extend our code first gets familiar with the [online documentation](https://docs.wandb.com/). As an example, we would create a sweep by running the following command in a terminal:

```bash
$ wandb sweep --entity st-team wandb_sweeps/st_ukb_uni_gender_1_fmri_none_nodemeta_mean_128.yaml
``` 
Which yielded an identifier, thus allowing us to run 25 random sweeps of our code by executing:
```bash
$ wandb agent st-team/spatio-temporal-brain/qqqjagns --count=25
```

Note that we use a different sweep for each cross validation fold (as described in the last paper).



## Python dependencies

The file `meta_data/st_env.yml` contains the exact dependencies used to develop and run this repository. In order to install all the dependencies automatically with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://anaconda.org/), one can easily just run the following command in the terminal to create an Anaconda environment:

```bash
$ conda env create -f meta_data/st_env.yml
$ conda activate st_env
```


## Repository structure


## Publications

The architecture implemented in this repository is described in detail in [a preprint at BioRxiv](https://biorxiv.org). If you use this architecture in your research work please cite the paper, with the following bibtex:

```
@article{Azevedo2020,

}
``` 

Two preliminary versions of this work were also presented in two other venues, which can be accessible online:

* _A deep spatiotemporal graph learning architecture for brain connectivity analysis_. EMBC 2020. [DOI: 10.1109/EMBC44109.2020.9175360](https://doi.org/10.1109/EMBC44109.2020.9175360).
* _Towards a predictive spatio-temporal representation of brain data_. Ai4AH @ ICLR 2020. [ArXiv: 2003.03290](https://arxiv.org/abs/2003.03290).