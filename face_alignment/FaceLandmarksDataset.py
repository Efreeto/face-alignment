import torch
from torch.utils.data import Dataset
from skimage import io, color, transform
import os, glob
import numpy as np


######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``RandomCrop``: to crop from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        img = img.astype('float32')
        landmarks = landmarks.astype('float32')

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks).float()}


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, type, transforms=None):
        """
        Args:
            path (string): Directory with all the images and landmarks.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.type = type
        self.transforms = transforms

        image_exts = ('*.jpg', '*.png')
        self.images_list = []
        for ext in image_exts:
            self.images_list.extend(sorted(glob.glob(os.path.join(path, ext))))
        assert self.images_list, "path does not contain images"

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = io.imread(self.images_list[idx])
        image = color.grey2rgb(image)   # For some gray scale images

        file_name = os.path.splitext(self.images_list[idx])[0]
        if self.type == 0:    # land110
            ground_truth = np.loadtxt(file_name + '.land', skiprows=1)
            ground_truth = np.vstack((ground_truth[0:32:2], ground_truth[32:64], ground_truth[88:108]))
        elif self.type == 1:  # 8W
            ground_truth = np.loadtxt(file_name + '.pts')
        elif self.type == 2:  # 300W, lfpw
            ground_truth = np.loadtxt(file_name + '.pts', skiprows=3, comments='}')

        bbox = np.loadtxt(file_name + '.rct_dlib_cpu').reshape(2,2)

        landmarks = np.vstack((bbox, ground_truth))

        sample = {'image': image, 'landmarks': landmarks}
        if self.transforms:
            sample = self.transforms(sample)

        return sample

