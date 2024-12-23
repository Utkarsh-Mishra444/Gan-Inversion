[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
# GAN Inversion for Latent Space Analysis

This project explores image manipulation through GAN (Generative Adversarial Network) inversion, specifically focusing on object rotation in car images. We leverage the `encoder4editing` framework, created by Omer Tov et al., to map images to a latent space, perform manipulations, and reconstruct the images. We thank the authors for providing the model and framework on which this project is built.

## Table of Contents

*   [Overview](#overview)
*   [Key Features](#key-features)
*   [Setup](#setup)
*   [How to Run](#how-to-run)
*   [Citation](#citation)


## Overview

This repository contains the code and experiments for exploring Generative Adversarial Network (GAN) inversion and latent space manipulation, conducted as part of the "Generative AI: Can we find meaning in the latent space?" project (ES431, IIT Gandhinagar). The primary goal is to explore image manipulation, particularly object rotation in car images, by leveraging the [`encoder4editing`](https://github.com/omertov/encoder4editing) framework developed by Tov et al.

This project explores the following key aspects of GAN inversion and latent space manipulation:

*   Mapping real-world car images to the latent space of a pre-trained StyleGAN model.
*   Manipulating latent codes to achieve rotations and other transformations.
*   Investigating the impact of backgrounds on image reconstruction quality.
*   Analyzing the fidelity of reconstructions after multiple passes through the encoder-decoder framework.
*   Demonstrating the non-bijective nature of GAN inversion.

The provided scripts offer a command-line interface for performing these explorations, enabling users to easily experiment with different images, interpolation techniques, and analysis parameters.


## Key Features

*   **GAN Inversion:** Implements a GAN inversion pipeline using the `encoder4editing` framework to map real-world images to latent space.
*   **Latent Space Exploration:** Performs latent space interpolation to achieve smooth rotations of objects in images.
*   **Background Testing:** Includes experiments to investigate how the image background affects reconstruction quality.
*   **Dataset:** Utilizes a set of car images from different viewpoints.
*   **Image Reconstructions:** Creates reconstructed images from latent space representations.
*   **Multi-pass Analysis:** Tests how the encoder-decoder model behaves with multiple passes.
*   **Command-line Interface:**  Easy-to-use command-line interface for all scripts.

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Utkarsh-Mishra444/Gan-Inversion.git
    cd Gan-Inversion
    ```

2.  **Install Dependencies:**

    We recommend using Anaconda or a similar tool to create a virtual environment. Please use the `environment/e4e_env.yaml` file provided in the original `encoder4editing` repository to create the environment. In addition, please install the following dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Pretrained Model:**

    *   Download the pretrained model (e.g., `e4e_cars_encode.pt`) and place it in the `pretrained_models/` directory. You can use the `gdown` command with the appropriate ID from the original repository or any other method. Here's an example:

        ```bash
        gdown --id  17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV -O pretrained_models/e4e_cars_encode.pt
        ```

    *   You may use the `main.py` file to download the required pretrained models. Please see the usage of `main.py` for how to do that.
    *   Alternately, you can visit the encoder4editing repository and download the models.

4.  **Dataset:** Place your dataset in the directory specified when running `main.py`.

## How to Run

*   **`main.py`**: This script performs GAN inversions on a set of images and saves the reconstructed outputs.

    ```bash
    python main.py --experiment_type cars_encode --image_directory /path/to/your/image/directory --output_directory /path/to/output/directory --download_models
    ```

    Where:

    *   `--download_models`: Downloads the pretrained models using IDs provided in the `main.py` script. If models are already downloaded, then they are not downloaded again.
    *   `--experiment_type`: Selects the experiment type. Select from `ffhq_encode`, `cars_encode`, `horse_encode`, `church_encode`.
    *   `--image_directory`: The input directory where the images are present.
    *   `--output_directory`: The output directory where you want to save the reconstructed images.

*   **`multi_pass_analysis.py`**: This script performs multiple passes of encoding and decoding on a single image and saves all the reconstruction outputs.

    ```bash
    python multi_pass_analysis.py --experiment_type cars_encode --image_path /path/to/your/image --num_passes 3
    ```

    Where:

    *   `--experiment_type`: Selects the experiment type. Select from `ffhq_encode`, `cars_encode`, `horse_encode`, `church_encode`.
    *   `--image_path`: Path to the image you want to process.
    *   `--num_passes`: The number of passes you want to perform through the encoder-decoder framework. Default value is 2.

*   **`interpolations.py`**: This script performs latent space interpolation on a set of images and generates transformed images and a GIF.

    ```bash
    python interpolations.py --experiment_type cars_encode --image_directory /path/to/your/image/directory --output_directory /path/to/output/directory --interpolation_type linear --interpolation_ratio 0.1
    ```

    Where:

    *   `--experiment_type`: Selects the experiment type. Select from `ffhq_encode`, `cars_encode`, `horse_encode`, `church_encode`.
    *   `--image_directory`: The input directory where the images are present.
    *   `--output_directory`: The output directory where you want to save the transformed images.
    *   `--interpolation_type`: Select the type of interpolation method you want to use. Currently, only `linear` is implemented.
    *   `--interpolation_ratio`: The ratio of latent codes to select from the input images. Default value is 0.1.


## Citation

If you use this code for your research, please cite the original authors' paper:
```
@article{tov2021designing,
  title={Designing an Encoder for StyleGAN Image Manipulation},
  author={Tov, Omer and Alaluf, Yuval and Nitzan, Yotam and Patashnik, Or and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2102.02766},
  year={2021}
}
```

