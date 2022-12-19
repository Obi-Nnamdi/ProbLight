# ProbLight
A Bayesian Computer Vision model that performs inverse rendering to extract lighting conditions from images. This project relies heavily on [Open3D](https://github.com/isl-org/Open3D), a python library for working with and rendering 3D objects.

This work is a Final Project for MIT's "Cognitive Computational Science" course (9.66). If interested, the research paper associated with this code is [here](https://drive.google.com/file/d/1SVYO5AmH1YqGw9wHCIHP1DnT96zDUBRF/view?usp=share_link).

This code focuses on using Bayesian inference to find a light source from a given scene and its geometry.

## Running the Model

Helper python programs are in the "helpers.py" file.

Experiments with Open3D (and a brief high-level tutorial on how to use the libary) are in the "Getting_Started_with_Open3D.ipynb" file.

All of the model's current progress is saved to "3D_Scene_Inference.ipynb". Future directions for improving the model and its research are contained in "Edelson Illusion Inference".
