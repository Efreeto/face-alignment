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


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']):
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


class FaceColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'].copy()

        to_pil = torchvision.transforms.ToPILImage()
        img = to_pil(image)
        img = self.color_jitter(img)
        to_tensor = torchvision.transforms.ToTensor()
        image = to_tensor(img).numpy().transpose(1,2,0)
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

        landmarks_rot = landmarks - origin_org
        landmarks_rot = np.asarray(np.dot(landmarks_rot, manual_theta_inv)[:, :2])
        landmarks_rot = landmarks_rot + origin_rot

        sample['image_rot'] = image_rot
        sample['landmarks_rot'] = landmarks_rot
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
        # landmarks = landmarks - (bbox[0], bbox[1])
        sample['image'] = image
        sample['landmarks'] = landmarks

        if 'image_rot' in sample:    # if RandomRotation, crop around the rotated image
            image, landmarks = sample['image_rot'], sample['landmarks_rot']
            bbox = utils.bounding_box(landmarks)
            center, scale = utils.center_scale_from_bbox(bbox)
            image = utils.crop(image, center, scale, self.resolution)
            # landmarks = landmarks - (bbox[0], bbox[1])
            sample['image_rot'] = image
            sample['landmarks_rot'] = landmarks

        return sample


class CreateHeatmaps(object):
    def __init__(self, output_size=64, n_features=68):
        self.output_size = output_size
        self.n_features = n_features

    def __call__(self, sample):
        landmarks = sample['landmarks']
        center, scale = utils.center_scale_from_bbox(utils.bounding_box(landmarks))
        heatmap = np.zeros((self.n_features, self.output_size, self.output_size))
        for i in range(self.n_features):
            new_pts = utils.transform(landmarks[i], center, scale, self.output_size)
            heatmap[i] = utils.draw_gaussian(heatmap[i], new_pts, 1)
        sample['heatmaps'] = torch.from_numpy(heatmap).view(self.n_features, self.output_size, self.output_size).float()

        return sample

class CreateHeatmaps2(object):
    def __init__(self, output_size=64, n_features=68):
        self.output_size = output_size
        self.n_features = n_features
        if self.n_features==68:
            self.neigbor_list = [[2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],
                                 [9,11],[10,12],[11,13],[12,14],[13,15],[14,16],[15,17],
                                 [16], [19], [18,20], [19,21], [20,22], [21],[24],[23,25],
                                 [24,26],[25,27],[26],[29],[28,30],[29,31],[30,34],[33],
                                 [32,34],[33,35],[34,36],[35],[],[37,39],[38,40],[],[40,42],
                                 [37,41],[],[43,45],[44,46],[],[46,48],[43,47],[],[49,51],
                                 [50,52],[51,53],[52,54],[53,55],[],[55,57],[56,58],[57,59],
                                 [58,60],[59,49],[49],[61,63],[62,64],[63,65],[55],[65,67],
                                 [66,68],[61,67]]
        elif self.n_features==108:
            self.neigbor_list = [[2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],
                                 [9,11],[10,12],[11,13],[12,14],[13,15],[14,16],[15,17],
                                 [16,18],[17,19],[18,20],[19,21],[20,22],[21,23],[22,24],
                                 [23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],
                                 [30,32],[31,33],[32],[],[34,36],[35,37],[36,38],[], [39,41],
                                 [40,42],[41,43], [],[45],[44,46], [45,47], [46], [49],[48,50],
                                 [],[50,52],[51],[],[53,55],[54,56],[],[56,58], [],[],[59,61],
                                 [60,62],[],[62,64],[],[],[65,67],[66,68],[],[],[69,71],[70,72],[]
                                 [54,55],[58,57],[],[60,61],[63,64],[],[81],[82],[79,83],[80,84],
                                 [81,85],[82,86],[83,87],[84,88],[48],[52],[],[89,91],[90,92],
                                 [91,93],[92,94],[93,95],[],[95,97],[96,98],[97,99],[98,100],[89,99],
                                 [],[101,103],[102,104],[103,105],[],[105,107],[106,108],[101,107]]

    def __call__(self, sample):
        landmarks = sample['landmarks']
        center, scale = center_scale_from_landmark(landmarks)
        heatmap = np.zeros((self.n_features, self.output_size, self.output_size))
        foo = np.zeros((self.output_size, self.output_size))

        for i in range(self.n_features):
            neighbors = self.get_neighbors(i)
            num_neighbors = len(neighbors)
            if num_neighbors == 0:
                heatmap[i] = utils.draw_gaussian(heatmap[i], utils.transform(landmarks[i], center, scale, self.output_size), 1)
                foo = utils.draw_gaussian(foo, utils.transform(landmarks[i], center, scale, self.output_size), 1)
            else:
                if num_neighbors == 2:
                    points = np.zeros((3,2))
                    points[0] = utils.transform(landmarks[neighbors[0]-1], center, scale, self.output_size).numpy()
                    points[1] = utils.transform(landmarks[i], center, scale, self.output_size).numpy()
                    points[2] = utils.transform(landmarks[neighbors[1]-1], center, scale, self.output_size).numpy()
                else:
                    points = np.zeros((2,2))
                    points[0] = utils.transform(landmarks[neighbors[0]-1], center, scale, self.output_size).numpy()
                    points[1] = utils.transform(landmarks[i], center, scale, self.output_size).numpy()

                heatmap[i] = utils.draw_gaussian2(heatmap[i], points, 1)
                # foo = utils.draw_gaussian(foo, utils.transform(landmarks[i], center, scale, self.output_size), 1)
                foo = utils.draw_gaussian2(foo, points, 1)
        """
        from PIL import Image
        im = Image.fromarray(foo*255)
        im.show()
        """

        heatmaps = torch.from_numpy(heatmap).view(1, self.n_features, self.output_size, self.output_size).float()

        return {'image': sample['image'], 'landmarks': heatmaps}

    def get_neighbors(self, landmark):
        return self.neigbor_list[landmark]


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
            if key in ['image', 'image_rot']:
                sample[key] = torchvision.transforms.ToTensor()(sample[key])
            elif key in  ['filename', 'heatmaps']:
                continue
            else:
                sample[key] = torch.from_numpy(sample[key]).float()
        return sample

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, type=1, transforms=None):
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
        if self.type == 1:  # 300W, lfpw
            landmarks = np.loadtxt(basename + '.pts', skiprows=3, comments='}')
        elif self.type == 2:  # land110
            landmarks = np.loadtxt(basename + '.land', skiprows=1)
            # landmarks = np.vstack((landmarks[0:32:2], landmarks[32:64], landmarks[88:108]))
        elif self.type == 3:  # FEI
            landmarks = np.ones((68,2))
        elif self.type == 4:  # 8W
            landmarks = np.loadtxt(basename + '.pts')

        sample = {'image': image, 'landmarks': landmarks, 'filename': filename}
        if self.transforms:
            sample = self.transforms(sample)

        return sample
