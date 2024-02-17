# 🎙️ LAAL Audio Baseline

This repository contains implementation of [AST](https://github.com/YuanGongND/ast), [SSAST](https://github.com/YuanGongND/ssast), [SlowFast](https://github.com/ekazakos/auditory-slow-fast), and [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet).


### Architectures
|   Model   | Pretrained | Epic-Sounds |   ESC50    |
  | --------- |:---:|:--------:|:--------:|
  |AST |  IN | -  | 93.3 |
  |SSAST |  AS 2M | 53.4 | 85.7 |
  |UniRepLKNet | - | - | 38.3 🛠️ |
  |UniRepLKNet | IN | - | 78.1 🛠️ |
  

### LoRA 
  |   Model   | Pretrained | Fine-Tuning | ESC50    |
  | --------- |:---:|:--------:|:--------:|
  |AST |  IN |  Full | 93.3 | 
  |AST | IN | LoRA | 78.5 (85.1) |
 
_The results of 5-fold cross validation of AST + LoRA is [83.7, 83.5, 85.2, 87.8, 52.6]. The average accuracy is 85.1 without the last result (52.6)._

## Installation
### Dependencies
- [python 3.8](https://www.python.org/)
- [pytorch](https://pytorch.org/) 
- [torchaudio](https://pytorch.org/) 
- [timm](https://huggingface.co/docs/timm/index) 
- [librosa](https://librosa.org/) 
- [wandb](https://wandb.ai/site) 
- [h5py](https://www.h5py.org/) 
- [fvcore](https://github.com/facebookresearch/fvcore/) 
- [iopath](https://github.com/facebookresearch/iopath) 
- [simplejson](https://simplejson.readthedocs.io/en/latest/) 
- [psutil](https://psutil.readthedocs.io/en/latest/) 
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) 

### Installing on SNU GSDS Server
On GSDS Server, you can create conda environment and install dependencies by following commands. You should uninstall and reinstall some packages to avoid version conflicts.
- `conda create -n sound python=3.8 -y`
- `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia`
- `conda install -c conda-forge librosa wandb tensorboard simplejson psutil timm`
- `conda install -c anaconda h5py`
- `conda install -c iopath iopath`
- `conda install pandas`
- `pip install fvcore`
- `conda uninstall pytorch torchvision torchaudio`
- `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
- `pip install timm==0.4.5`
- `pip install numpy==1.23.0`


## Getting Started
Running the code is made simple with the included bash scripts. The training scripts are in `scripts/`
```
$ scripts/{model}_{dataset}_train.sh
```
If you are using [SLURM](https://slurm.schedmd.com/documentation.html) for a job submission, please check the scripts within the `scripts/slurm` folder.

## Configurations
Default parameters are configured with in the `config/defaults.py`. Further customization options are available by editing the following YAML files.
- `config/AST_epicsounds.yaml`
- `config/AST_esc50.yaml`
- `config/SSAST_epicsounds.yaml`
- `config/SSAST_esc50.yaml`
- `config/UNIREPLK_epicsounds.yaml`
- `config/UNIREPLK_esc50.yaml`


## Acknowledgments

This code is partly based on the open-source implementations from the following sources: [SlowFast](https://github.com/ekazakos/auditory-slow-fast), [Epic-Sounds](https://github.com/epic-kitchens/epic-sounds-annotations), [AST](https://github.com/YuanGongND/ast), [SSAST](https://github.com/YuanGongND/ssast), [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet), and [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT).