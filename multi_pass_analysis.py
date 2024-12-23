import os
import sys
import time
from argparse import Namespace, ArgumentParser
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils.common import tensor2im
from models.psp import pSp
import argparse
from configs import data_configs

"""
This script performs GAN inversion on a single input image and then performs
multiple passes through the encoder decoder framework to investigate
how the model behaves when given repeated passes through the same model.

The script loads a StyleGAN model and its corresponding encoder,
maps a single input image to the latent space, reconstructs the image,
then re-encodes the reconstructed image and generates a second
and subsequent reconstructions based on how many passes are mentioned
in the commandline argument, saving all three images.

Example Usage:

    python multi_pass_analysis.py --experiment_type cars_encode  --image_path "/path/to/image" --num_passes 3
"""

def display_alongside_source_image(result_image, source_image, resize_dims):
    """
    Displays the source image and the generated image side by side (optional).

    Args:
        result_image (PIL.Image.Image): The generated image.
        source_image (PIL.Image.Image): The original source image.
        resize_dims (tuple): The dimensions to resize the images to.

    Returns:
        PIL.Image.Image: The concatenated image for display.
    """
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def save_image(img, save_dir, idx):
    """
    Saves a single image to a given directory with a specific index.

    Args:
        img (torch.Tensor): The image tensor.
        save_dir (str): The directory to save the image in.
        idx (int): The index for the filename.
    """
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


def get_latents(net, x, is_cars=True):
    """
    Encodes input image to extract its latent code using the model.

    Args:
        net (torch.nn.Module): The GAN inversion network (encoder+decoder).
        x (torch.Tensor): The input image tensor.
        is_cars (bool): A flag to denote if the model is cars

    Returns:
        torch.Tensor: The extracted latent codes.
    """
    codes = net(x, randomize_noise=False, return_latents=True)[1]
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


@torch.no_grad()
def decode_image(net, latent):
  """
  Decodes a latent code into an image using the model.

  Args:
      net (torch.nn.Module): The GAN inversion network (encoder+decoder).
      latent (torch.Tensor): The latent code for the image.

  Returns:
      PIL.Image.Image: The reconstructed image
  """
  with torch.no_grad():
    img1 = net.decoder([latent.unsqueeze(0)],input_is_latent=True, randomize_noise=False, return_latents=False)
    img = img1[0][0]
  return tensor2im(img)

if __name__ == '__main__':
    parser = ArgumentParser(description="GAN Inversion and Latent Space Analysis")
    parser.add_argument('--experiment_type', type=str, default='cars_encode',
                        help='Experiment type: ffhq_encode, cars_encode, horse_encode, church_encode')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image.')
    parser.add_argument('--num_passes', type = int , default = 2,
                        help='Number of passes for the model')
    args = parser.parse_args()

    # Configuration
    experiment_type = args.experiment_type
    MODEL_PATHS = {
        "ffhq_encode": os.path.join("encoder4editing","pretrained_models/e4e_ffhq_encode.pt"),
        "cars_encode": os.path.join("encoder4editing","pretrained_models/e4e_cars_encode.pt"),
        "horse_encode": os.path.join("encoder4editing","pretrained_models/e4e_horse_encode.pt"),
        "church_encode": os.path.join("encoder4editing","pretrained_models/e4e_church_encode.pt"),
    }
    IMAGE_PATHS = {
        "ffhq_encode": os.path.join("encoder4editing", "notebooks/images/input_img.jpg"),
        "cars_encode": os.path.join("encoder4editing", "notebooks/images/car_img.png"),
        "horse_encode": os.path.join("encoder4editing", "notebooks/images/horse_img.jpg"),
        "church_encode": os.path.join("encoder4editing", "notebooks/images/church_img.jpg"),
    }
    # Define paths
    MODEL_PATHS = {
        "ffhq_encode": {"id": "1cUv_reLE6k3604or78EranS7XzuVMWeO", "name": "e4e_ffhq_encode.pt"},
        "cars_encode": {"id": "17faPqBce2m1AQeLCLHUVXaDfxMRU2QcV", "name": "e4e_cars_encode.pt"},
        "horse_encode": {"id": "1TkLLnuX86B_BMo2ocYD0kX9kWh53rUVX", "name": "e4e_horse_encode.pt"},
        "church_encode": {"id": "1-L0ZdnQLwtdy6-A_Ccgq5uNJGTqE7qBa", "name": "e4e_church_encode.pt"}
    }
    EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": os.path.join("encoder4editing","pretrained_models/e4e_ffhq_encode.pt"),
        "image_path": os.path.join("encoder4editing", "notebooks/images/input_img.jpg")
    },
    "cars_encode": {
        "model_path": os.path.join("encoder4editing","pretrained_models/e4e_cars_encode.pt"),
        "image_path": os.path.join("encoder4editing", "notebooks/images/car_img.png")
    },
    "horse_encode": {
        "model_path": os.path.join("encoder4editing","pretrained_models/e4e_horse_encode.pt"),
        "image_path": os.path.join("encoder4editing", "notebooks/images/horse_img.jpg")
    },
    "church_encode": {
        "model_path": os.path.join("encoder4editing","pretrained_models/e4e_church_encode.pt"),
        "image_path": os.path.join("encoder4editing", "notebooks/images/church_img.jpg")
    }
}
    # Setup required image transformations
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    if experiment_type == 'cars_encode':
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 192)
    else:
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)
    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    # Load the input image
    original_image = Image.open(args.image_path).convert("RGB")
    transformed_image = EXPERIMENT_ARGS['transform'](original_image)
    images_to_save = [original_image]
    current_image = transformed_image

    # Perform multiple passes
    for i in range(args.num_passes):
        latent = get_latents(net, current_image.unsqueeze(0).to("cuda").float(), is_cars=(experiment_type == 'cars_encode'))
        decoded_image = decode_image(net, latent)
        images_to_save.append(Image.fromarray(decoded_image))
        current_image =  EXPERIMENT_ARGS['transform'](Image.fromarray(decoded_image))

    #Save output
    output_directory = os.path.join(os.path.dirname(args.image_path), 'inference_output')
    os.makedirs(output_directory, exist_ok=True)
    for i, image in enumerate(images_to_save):
        save_image(image, output_directory, idx = i+1)
    print(f"Image Reconstructions saved to: {output_directory}")