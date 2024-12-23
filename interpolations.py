import os
import sys
import time
from argparse import Namespace, ArgumentParser
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import glob
import imageio
import argparse
from scipy.interpolate import interp1d
from utils.common import tensor2im
from models.psp import pSp
from configs import data_configs
from torch.utils.data import Dataset, DataLoader

"""
This script performs latent space interpolation for generating image transformations.
It loads a set of images, encodes them into the latent space, performs
interpolation between the latent codes, reconstructs the interpolated codes back
into images, and saves the generated images. It also creates a GIF of the
generated image sequence.

Example Usage:

    python interpolations.py --experiment_type cars_encode --image_directory "/path/to/your/images" --output_directory "/path/to/output_directory" --interpolation_type linear --interpolation_ratio 0.1
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

def generate_inversions(g, latent_codes, sno, is_cars, output_path):
    """
    Generates and saves inverted images from latent codes.

    Args:
        g (torch.nn.Module): The generator network (decoder).
        latent_codes (torch.Tensor): The latent codes for the images.
        sno (int): Starting number for saving the files.
        is_cars (bool): Flag to indicate if the images are cars
        output_path (str): The output directory to save the images to.
    """
    print('Saving inversion images')
    inversions_directory_path = os.path.join(output_path, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(len(latent_codes)):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + sno)

def get_latents(net, x, is_cars=True):
    """
    Encodes input images to extract their latent codes using the model.

    Args:
        net (torch.nn.Module): The GAN inversion network (encoder+decoder).
        x (torch.Tensor): The input image tensor.
        is_cars (bool): A flag to denote if the images are cars

    Returns:
        torch.Tensor: The extracted latent codes.
    """
    codes = net(x, randomize_noise=False, return_latents=True)[1]
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes

class InferenceDataset(Dataset):
    """
    Custom dataset to load the images for inference.

    Args:
        root (str): Path to the root folder containing images
        opts : Options for the experiment
        transform (torchvision.transforms): Transforms for the data
        preprocess (func): function to preprocess data
    """
    def __init__(self, root, opts, transform=None, preprocess=None):
        sys.path.append(os.path.join(os.getcwd(), "encoder4editing"))
        from utils import data_utils
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform
        self.preprocess = preprocess
        self.opts = opts
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        from_path = self.paths[index]
        if self.preprocess is not None:
            from_im = self.preprocess(from_path)
        else:
            from_im = Image.open(from_path).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)
        return from_im

def setup_data_loader(opts ,img_path):
    """
    Sets up the data loader for inference.

    Args:
        opts (argparse.Namespace): Command-line arguments and configuration.
        img_path (str): Path to the folder with input images.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = img_path if img_path is not None else print('error in path')
    print(f"images path: {images_path}")
    align_function = None

    test_dataset = InferenceDataset(root=images_path,
                                    transform= EXPERIMENT_ARGS['transform'], #transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)
    print(f'dataset length: {len(test_dataset)}')
    return data_loader

@torch.no_grad()
def get_all_latents(net, data_loader, n_images=None, is_cars=True):
    """
    Extracts the latent codes for all the images in a data loader.

    Args:
        net (torch.nn.Module): The GAN inversion network.
        data_loader (torch.utils.data.DataLoader): The data loader.
        n_images (int, optional): Max number of images to perform inversion. Defaults to None.
        is_cars (bool): Boolean to specify if the dataset is cars or not

    Returns:
         torch.Tensor: The concatenated latent codes for all images.
    """
    device='cuda'
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)

def create_gif(image_folder, gif_path, duration=0.5):
    """
    Creates a GIF from a folder of images.

    Args:
        image_folder (str): The path to the folder containing the images
        gif_path (str): The path where to save the generated GIF
        duration (float): The duration for each image in the GIF
    """
    images = []
    for filename in sorted(glob.glob(os.path.join(image_folder, '*.jpg'))):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_path, images, duration=duration)

def interpolate_list(original_list, ratio, kind='linear'):
  """
    Interpolates a list of latent codes.

    Args:
        original_list (List of torch.Tensor): List of latent codes to interpolate
        ratio (float): The interpolation ratio
        kind (str): The type of interpolation to perform. Default is linear.

    Returns:
         torch.Tensor: The interpolated latent codes.
    """
  n = len(original_list)
  num_elements = int(n * ratio)

  # Include first and last indices for interpolation
  indices = np.linspace(0, n-1, num_elements, dtype=int)
  selected_elements = [original_list[i] for i in indices]

  # Create interpolation function
  x = indices
  y = torch.stack(selected_elements)
  interp_func = interp1d(x, y, kind=kind, axis=0)

  # Interpolate for the full range
  new_x = np.arange(n)  # Indices for the full list
  interpolated_tensors = interp_func(new_x)

  # Create new list with interpolated tensors
  new_list = interpolated_tensors.tolist()
  return new_list

if __name__ == '__main__':
    parser = ArgumentParser(description="GAN Inversion and Latent Space Analysis")
    parser.add_argument('--experiment_type', type=str, default='cars_encode',
                        help='Experiment type: ffhq_encode, cars_encode, horse_encode, church_encode')
    parser.add_argument('--image_directory', type=str, required=True,
                        help='Path to the directory containing images.')
    parser.add_argument('--output_directory', type=str, required=True,
                         help='Path to save the interpolated images and gifs')
    parser.add_argument('--interpolation_type', type=str, default='linear',
                         help='Type of interpolation to use: linear, cubic')
    parser.add_argument('--interpolation_ratio', type = float, default = 0.1,
                        help= "Ratio to create interpolation points")
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

    # Setup data loader
    data_loader = setup_data_loader(opts ,img_path = args.image_directory)
    # Get latent codes for all images
    latent_codes = get_all_latents(net, data_loader, n_images=None, is_cars=(experiment_type == 'cars_encode'))
    print(f"length of latent codes {len(latent_codes)}")
    # Interpolate Latent Codes
    interpolated_latents = interpolate_list(latent_codes.cpu(), args.interpolation_ratio,
                                          kind=args.interpolation_type)
    interpolated_latents = torch.Tensor(interpolated_latents).to("cuda")

    # Generate Inverted images
    generate_inversions(net.decoder, interpolated_latents, 1, is_cars=(experiment_type == 'cars_encode'), output_path = args.output_directory)
    print("All interpolated image inversions have been generated")
    # Create and save GIF
    gif_path = os.path.join(args.output_directory, f"interpolation_{args.interpolation_type}_ratio_{args.interpolation_ratio}.gif")
    create_gif(os.path.join(args.output_directory, 'inversions'), gif_path)
    print(f"Interpolation GIF saved to: {gif_path}")