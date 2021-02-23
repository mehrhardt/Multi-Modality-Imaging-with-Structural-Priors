# Multi-modality imaging with structure-promoting regularisers

This code reproduces all experiments from the book chapter "Multi-Modality Imaging with Structure-Promoting Regularizers" [1]. See below for two examples of multi-modality imaging.

<p align="center">
<img src="https://github.com/mehrhardt/Multi-Modality-Imaging-with-Structure-Promoting-Regularizers/blob/master/petmrct.png" width="80%" border="0"/>
</p>

<p align="center">
<img src="https://github.com/mehrhardt/Multi-Modality-Imaging-with-Structure-Promoting-Regularizers/blob/master/rgb.png" width="80%" border="0"/>
</p>

## Getting started

The code has been tested using Python 3.6 and a number of packages. The version numbers of the most important ones to generate our results are:
* [ODL](https://odlgroup.github.io/odl/) [1.0.0.dev0]
* [NumPy](https://numpy.org/) [1.13.3]
* [Astra](https://www.astra-toolbox.com/) ['1.8.3']

The code for the two example problems are
* [example_xray.py](example_xray.py)
* [example_superresolution.py](example_superresolution.py)

To execute all code, run
* [run_all.py](run_all.py)

For one of the motivation figures also the "spectral" package is needed.
* [spectral](https://pypi.org/project/spectral/) ['0.22.1']

## References
[1] Ehrhardt, M. J. (2021). Multi-Modality Imaging with Structure-Promoting Regularizers. Handbook of Mathematical Models and Algorithms in Computer Vision and Imaging [arXiv preprint](https://arxiv.org/abs/2007.11689)