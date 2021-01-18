# Image super resolution
Project code for Data Representation, Reduction and Analysis course.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running instructions](#running-instructions)

## Prerequisites

As a prerequisite you will need to install [Python 3](https://www.python.org/). The code was tested with
versions `3.7.3` and `3.8.7`. Python 3 comes bundled with the package installer `pip` but if it is not available on your
machine after installing Python 3, install it by following
the [Pip Documentation](https://pip.pypa.io/en/stable/installing/).

Download the DIV2K 2018 dataset from [DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
and decompress the files each in their corresponding folder into a parent `data` folder.

## Installation

First, a generally good practice when working with Python is to create a virtual environment (venv) to run the code in.
In order to create a virtual environment please refer to
the [official documentation](https://docs.python.org/3.7/tutorial/venv.html).

To install project packages run the following command (make sure your venv is activated):
```
pip install -r requirements.txt
```

## Running instructions

The available command line arguments are the following:
* `-c` or `--category` followed by one of the following strings:
    * `x8` corresponds to using the low resolution images from folder _DIV2K_train_LR_x8_ and _DIV2K_valid_LR_x8_
    * `mild` corresponds to using the low resolution images from folder _DIV2K_train_LR_mild_ and _DIV2K_valid_LR_mild_
    * `wild` corresponds to using the low resolution images from folder _DIV2K_train_LR_wild_ and _DIV2K_valid_LR_wild_
    * `difficult` corresponds to using the low resolution images from folder _DIV2K_train_LR_difficult_ and _DIV2K_valid_LR_difficult_
* `-l` or `--load_model` load model weights from file _trained\_model.h5_, don't train it 

An example run command would look like:
To install project packages run the following command (make sure your venv is activated):
```
python main.py -c x8
```