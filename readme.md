# Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture

This directory contains the code used to produce the results presented in *Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture*.
The subsections below provide an overview of the contents within this directory and how to run the provided code.


## Directory Contents

- ```/models```:  Helper functions that build all neuro-constrained models (both single-component models and composite models).
    - ```center_surround.py```: Code for building Center-Surround Antagonism ResNet50.
    - ```cortical_magnification.py```: Code for building Cortical Magnification ResNet50.
    - ```local_rf.py```: Code for building Local Receptive Field ResNet50.
    - ```composite.py```: Code for building divisive normalization models (tuned and standard) and composite models which integrate multiple neuro-constrained architectural components.
- ```/utils```: Contains pytorch implementations of neuro-constrained components evaluated in the paper.
    - ```DoG.py```: Center-surround convolution module.
    - ```LocallyConnected2D.py```: Local RF module.
    - ```DivisiveNormalization.py```: Divisive normalization modules.
    - ```PolarTransform.py```: Cortical magnification module.
- ```/eval/tiny_imagenet_c_eval.py```: Script for evaluating a trained model on Tiny-ImageNet-C
- ```main.py```: Script for ImageNet training, ImageNet validation set evaluation, and Tiny-ImageNet fine-tuning.


## Running The Code


### Requirements
Following installation of Python3.7+, all required Python packages can be installed with the command
```
pip install -r requirements.txt
```

Local copies of [ImageNet](https://image-net.org/challenges/LSVRC/2012/), [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet), and [Tiny-ImageNet-C](https://github.com/hendrycks/robustness) are additionally required if you wish to reproduce the results presented in *Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture* from scratch.


### Model training/fine-tuning
Models can be trained from scratch or fine-tuned by running ```main.py```, with the following arguments:
```
python main.py
    --dataset [dataset specifier (imagenet or tinyimagenet)]
    --data_root [Full path to dataset directory]
    --model_type [Model architecture specification]
    --model_path [Optional. Path to trained model weights, if available]
    --output_root [Path to directory where artifacts will be saved]
    --mode [train (for model training), validate (for validation set evaluation), or finetune (for tinyimagenet fine tuning)]
    --num_epochs [Maximum number of training epochs]
    --batch_size [Training/evaluation batch size]
    --lr [Initial learning rate]
    --lr_step_milestones [Epochs at which learning rate will be scaled by a factor of gamma]
    --gamma [Learning rate scaling factor]
    --momentum [Optimizer momentum factor]
    --weight_decay [L2 weight penalty]
    --save_freq [Epoch frequency at which model artifacts are saved]
    --iteration_verbosity [During training, loss and accuracy will be reported every iteration_verbosity batches]
    --num_workers [Number of dataloader worker processes]
    --device [Device to use for training/testing]
```
Default parameters specified in ```main.py``` were the ones used in the paper.


### Tiny-ImageNet-C Evaluation

After a model has been trained or fine-tuned on Tiny-ImageNet, corrupted image classification accuracy may be evaluated in Tiny-ImageNet-C using the ```/eval/tiny_imagenet_c_eval.py```.
After changing your working directory to ```/eval```, this code can be run using the following command:

```
python tiny_imagenet_c_eval.py
    --dataset tinyimagenet-c
    --data_root [Full path to tinyimagenet-c dataset directory]
    --model_type [Model architecture specification]
    --model_path [Path to trained model weights]
    --output_root [Path to directory where results will be saved]
    --batch_size [Evaluation batch size]
    --num_workers [Number of dataloader worker processes]
    --device [Device to use for testing]
```

