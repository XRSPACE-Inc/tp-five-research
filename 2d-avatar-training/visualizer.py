"""This script defines the visualizer for Deep3DFaceRecon_pytorch
"""

import warnings

import os
import numpy as np
import torch
import time
from PIL import Image
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter


def tensor2im(input_image: np.ndarray | torch.Tensor, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


class MyVisualizer:
    def __init__(
        self,
        name: str,
        checkpoints_dir: str = './checkpoints',
        test_mode: bool = False
    ):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.name = name
        self.img_dir = os.path.join(checkpoints_dir, name, 'results')

        self.test_mode = test_mode
        if not self.test_mode:
            self.writer = SummaryWriter(os.path.join(checkpoints_dir, name, 'logs'))
            # create a logging file to store training losses
            self.log_name = os.path.join(checkpoints_dir, name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            self.img_dir = './results/'

    def display_current_results(
        self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
        add_image=True
    ):
        """Display current results on tensorboad; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        # if (not add_image) and (not save_results): return
        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = tensor2im(image[i])
                if not self.test_mode and add_image:
                    self.writer.add_image(
                        label + '_%s' % (dataset),
                        image_numpy, total_iters, dataformats='HWC'
                    )

                if save_results:
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d' % (epoch, total_iters))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.jpg' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.jpg' % (label, i + count))
                    save_image(image_numpy, img_path)

    def plot_current_losses(self, total_iters, losses, dataset='train'):
        if self.test_mode:
            return
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s' % dataset, value, total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, message, pbar=None):
        """print current losses on console; also save the losses to the disk
        """
        now = time.strftime("%c")
        message = f'[{now}] {message}'

        if pbar is not None:
            pbar.write(message)
        else:
            print(message)  # print the message

        if not self.test_mode:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
