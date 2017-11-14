import torch
from torch.utils.data import Dataset
from skimage import io, color, transform
import torchvision
import os, glob
import numpy as np
import random
from scipy import ndimage
from PIL import Image
import torch.nn.functional as F

from . import utils

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

class RandomHorizFlip(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            landmarks = landmarks.transpose()
            landmarks[0] = image.shape[1] - landmarks[0]
            landmarks = landmarks.transpose()
            landmarks = utils.shuffle_lr(landmarks)

        return {'image': image, 'landmarks': landmarks}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return {'image': image.add(rgb.view(3, 1, 1).expand_as(image)), 'landmarks': landmarks}


class Grayscale(object):

    def __call__(self, img):

        gs = img.clone()
#        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        img = torch.from_numpy(img)
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return {'image': np.array(img.lerp(gs, alpha)), 'landmarks': landmarks}


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        img = torch.from_numpy(img).int()
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return {'image': img.lerp(gs, alpha).numpy(), 'landmarks': landmarks}


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        img, landmarks = sample['image'], sample['landmarks']
        img =torch.from_numpy(img).int()
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        img = img.lerp(gs, alpha)
        return {'image': np.array(img), 'landmarks': landmarks}


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

class FaceColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.color_jitter = ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'].copy()
        img = F.to_pil_image(image)
        return {'image':  image, 'landmarks': landmarks}


class RandomRotation(object):
    def __init__(self, maximum_angle=50., minimum_angle=5.):
        self.maximum_angle = maximum_angle - minimum_angle
        self.minimum_angle = minimum_angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        rotation_angle = (random.random() - 0.5) * 2 * self.maximum_angle
        if rotation_angle > 0:
            rotation_angle += self.minimum_angle
        else:
            rotation_angle -= self.minimum_angle
        manual_theta = utils.transformation_matrix(-rotation_angle)
        manual_theta_inv = utils.transformation_matrix(rotation_angle)

        image_rot = ndimage.rotate(image, rotation_angle, reshape=True)
        origin_org = ((image.shape[1] / 2.0, image.shape[0] / 2.0))
        origin_rot = ((image_rot.shape[1] / 2.0, image_rot.shape[0] / 2.0))

        # Rotate BBox here?
        # center, scale = bounding_box(landmarks)
        # center = utils.rotate(origin_org, center, rotation_angle)
        # # if self.enable_cuda:
        # #     center = center.cuda()
        # center = center + origin_rot - origin_org  # because reshape=True

        landmarks_rot = landmarks - origin_org
        landmarks_rot = np.dot(landmarks_rot, manual_theta_inv)[:, :2]
        landmarks_rot = landmarks_rot + origin_rot
        # display_landmarks(image, landmarks.cpu().numpy(), [], "Original")
        # display_landmarks(image_rot, landmarks_rot.cpu().numpy(), [], "Manually Rotated")

        sample['image_org'] = image
        sample['landmarks_org'] = landmarks
        sample['image'] = image_rot
        sample['landmarks'] = landmarks_rot
        sample['theta'] = manual_theta

        return sample

class LandmarkCrop(object):
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        bbox = utils.bounding_box(landmarks)
        center, scale = utils.center_scale_from_bbox(bbox)
        image = utils.crop(image, center, scale, self.resolution)
        landmarks = landmarks - [bbox[0], bbox[1]]
        sample['image'] = image
        sample['landmarks'] = landmarks

        if len(sample['image_org']):
            image, landmarks = sample['image_org'], sample['landmarks_org']
            bbox = utils.bounding_box(landmarks)
            center, scale = utils.center_scale_from_bbox(bbox)
            image = utils.crop(image, center, scale, self.resolution)
            landmarks = landmarks - [bbox[0], bbox[1]]
            sample['image_org'] = image
            sample['landmarks_org'] = landmarks

        return sample

class CreateHeatmaps(object):
    def __init__(self, output_size=64, n_features=68):
        self.output_size = output_size
        self.n_features = n_features

    def __call__(self, sample):
        landmarks = sample['landmarks']
        center, scale = center_scale_from_landmark(landmarks)
        heatmaps = torch.Tensor(1, self.n_features, self.output_size, self.output_size)
        heatmap = np.zeros((self.n_features, self.output_size, self.output_size))
        for i in range(self.n_features):
            heatmap[i] = utils.draw_gaussian(heatmap[i], utils.transform(landmarks[i], center, scale, self.output_size), 1)
        heatmaps = torch.from_numpy(heatmap).view(1, self.n_features, self.output_size, self.output_size).float()

        return {'image': sample['image'], 'landmarks': heatmaps}


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
        for key in sample:
            if key in ['image', 'image_org']:
                sample[key] = torchvision.transforms.ToTensor()(sample[key])
            elif key == 'filename':
                continue
            else:
                sample[key] = torch.from_numpy(sample[key]).float()
        return sample

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

        filename = self.images_list[idx]
        basename = os.path.splitext(filename)[0]
        if self.type == 0:    # land110
            landmarks = np.loadtxt(basename + '.land', skiprows=1)
            landmarks = np.vstack((landmarks[0:32:2], landmarks[32:64], landmarks[88:108]))
        elif self.type == 1:  # 8W
            landmarks = np.loadtxt(basename + '.pts')
        elif self.type == 2:  # 300W, lfpw
            landmarks = np.loadtxt(basename + '.pts', skiprows=3, comments='}')
        elif self.type == 3:  # FEI
            landmarks = np.ones((68,2))

        sample = {'image': image, 'landmarks': landmarks, 'filename': filename}
        if self.transforms:
            sample = self.transforms(sample)

        return sample

