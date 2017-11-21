from __future__ import print_function
import os
import sys
import scipy
import torch
import math
import numpy as np
import cv2
import time

from torch.autograd import Variable
import matplotlib.pyplot as plt


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

def transform(point, center, scale, resolution, invert=False):
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def transform_Variable(point, center, scale, resolution, invert=False):
    _pt = Variable(torch.ones(3), requires_grad=True)
    if point.is_cuda:
        _pt = _pt.cuda()
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = Variable(torch.eye(3))
    if point.is_cuda:
        t = t.cuda()
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()



def center_scale_from_bbox(bbox):
    minx = bbox[0]
    miny = bbox[1]
    maxx = bbox[2]
    maxy = bbox[3]
    center = torch.FloatTensor([maxx - (maxx - minx) / 2.0, maxy - (maxy - miny) / 2.0])
    scale = (maxx - minx + maxy - miny) / 190.0
    return center, scale


def bounding_box(landmarks):
    minx = landmarks[:,0].min()
    miny = landmarks[:,1].min()
    maxx = landmarks[:,0].max()
    maxy = landmarks[:,1].max()
    return (minx, miny, maxx, maxy)


def crop(image, center, scale, resolution=256.0):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg


def crop2(image, bbox):
    # Crop and scale to prepare for TwoStage
    (left, top, right, bottom) = bbox
    sh_scale = 0.1
    rows = image.shape[0]
    cols = image.shape[1]
    w = right-left
    h = bottom-top
    rct_left = max(round(left - sh_scale*w), 0)
    rct_top = max(round(top - sh_scale*h), 0)
    rct_right = min(round(right + sh_scale*w), cols)
    rct_bottom = min(round(bottom+ sh_scale*h), cols)

    newImg = image[rct_top:rct_bottom, rct_left:rct_right]
    newImg = cv2.resize(newImg, (480, 480))
    return newImg


def get_preds_fromhm(hm, center=None, scale=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = preds[i, j, 0], preds[i, j, 1]
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[int(pY),
                         int(pX) + 1] - hm_[int(pY),
                                            int(pX) - 1],
                     hm_[int(pY) + 1, int(pX)] - hm_[int(pY) - 1, int(pX)]])
                preds[i, j].add(diff.sign().mul(.25))

    preds.add_(1)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


def get_preds_fromhm_Variable(hm, center=None, scale=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0] = (preds[..., 0] - 1) % hm.size(3) + 1
    preds[..., 1] = preds[..., 1].add(-1).div(hm.size(2)).floor().add(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :].data
            pX, pY = preds[i, j, 0].data.cpu().numpy(), preds[i, j, 1].data.cpu().numpy()
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = Variable(torch.FloatTensor(
                    [hm_[int(pY),
                         int(pX) + 1] - hm_[int(pY),
                                            int(pX) - 1],
                     hm_[int(pY) + 1, int(pX)] - hm_[int(pY) - 1, int(pX)]])).cuda()
                preds[i, j] = preds[i, j].add(diff.sign().mul(.25))

    preds = preds.add(1)

    preds_orig = preds.clone()
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform_Variable(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


# From pyzolib/paths.py (https://bitbucket.org/pyzo/pyzolib/src/tip/paths.py)


def appdata_dir(appname=None, roaming=False):
    """ appdata_dir(appname=None, roaming=False)

    Get the path to the application directory, where applications are allowed
    to write user specific files (e.g. configurations). For non-user specific
    data, consider using common_appdata_dir().
    If appname is given, a subdir is appended (and created if necessary).
    If roaming is True, will prefer a roaming directory (Windows Vista/7).
    """

    # Define default user directory
    userDir = os.getenv('FACEALIGNMENT_USERDIR', None)
    if userDir is None:
        userDir = os.path.expanduser('~')
        if not os.path.isdir(userDir):  # pragma: no cover
            userDir = '/var/tmp'  # issue #54

    # Get system app data dir
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = os.getenv('LOCALAPPDATA'), os.getenv('APPDATA')
        path = (path2 or path1) if roaming else (path1 or path2)
    elif sys.platform.startswith('darwin'):
        path = os.path.join(userDir, 'Library', 'Application Support')
    # On Linux and as fallback
    if not (path and os.path.isdir(path)):
        path = userDir

    # Maybe we should store things local to the executable (in case of a
    # portable distro or a frozen application that wants to be portable)
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):
        prefix = os.path.abspath(os.path.dirname(sys.executable))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):  # pragma: no cover
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass  # We cannot write in this directory
            else:
                path = localpath
                break

    # Get path specific for this app
    if appname:
        if path == userDir:
            appname = '.' + appname.lstrip('.')  # Make it a hidden directory
        path = os.path.join(path, appname)
        if not os.path.isdir(path):  # pragma: no cover
            os.mkdir(path)

    # Done
    return path


def shuffle_lr(parts, pairs=None):
    if pairs is None:
        if parts.shape[0] == 68:
            pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                 [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                 [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                 [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]
        elif parts.shape[0] == 108:
            pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26],
                     [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19],
                     [14, 18], [15, 17], [33, 42], [34, 41], [35, 40], [36, 39], [37, 38],
                     [67, 68], [66, 69], [65, 70], [64, 71], [52, 61], [53, 60], [72, 75], [54, 59],
                     [55, 58], [56, 63], [73, 76], [57, 62], [78, 79], [80, 81], [82, 83], [84, 85],
                     [86, 87], [47, 51], [48, 50], [88, 94], [89, 93], [90, 92], [100, 104], [101, 103],
                     [107, 105], [99, 95], [98, 96], [74, 77]]
        else:
            assert(False, "unexpected number of landmarks")
    for matched_p in pairs:
        idx1, idx2 = matched_p[0], matched_p[1]
        tmp = parts[idx1].copy()
        parts[idx1] = parts[idx2].copy()
        parts[idx2] = tmp
        # tmp = np.copy(parts[..., idx1])
        # np.copyto(parts[..., idx1], parts[..., idx2])
        # np.copyto(parts[..., idx2], tmp)

    return parts


def flip(tensor, is_label=False):
    was_cuda = False
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    elif isinstance(tensor, torch.cuda.FloatTensor):
        tensor = tensor.cpu().numpy()
        was_cuda = True

    was_squeezed = False
    if tensor.ndim == 4:
        tensor = np.squeeze(tensor)
        was_squeezed = True
    if is_label:
        tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
        tensor = cv2.flip(shuffle_lr(tensor), 1).reshape(tensor.shape)
        tensor = tensor.swapaxes(2, 1).swapaxes(1, 0)
    else:
        tensor = cv2.flip(tensor, 1).reshape(tensor.shape)
    if was_squeezed:
        tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor)
    if was_cuda:
       tensor = tensor.cuda()
    return tensor


def landmark_diff(lm1, lm2, write_diffs=False):
    norm = abs(lm1[0][1] - lm1[8][1])
    if norm < 0.1:
        print('unexpected landmark normalization')
        norm = 1

    distances = scipy.spatial.distance.cdist(lm1, lm2).diagonal()
    max_distance = distances.max()/norm
    avg_distance = distances.sum()/norm

    return max_distance, avg_distance


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    modified for origin at top left (increasing y proceeding downward)
    """
    angle = angle * math.pi / 180.
    ox, oy = origin
    px, py = point
    if px is -1 and py is -1:
        return px, py
    # flip y axis
    ox = float(ox)
    oy = float(oy)
    oy = -oy
    py = -py

    s = math.sin(angle)
    c = math.cos(angle)

    # translate point back to origin:
    px -= ox
    py -= oy

    # rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # translate point back:
    qx = xnew + ox
    qy = ynew + oy
    qy = -qy
    return qx, qy


# https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
# 3D version: https://stackoverflow.com/a/6802723/2680660
def transformation_matrix(rotation_angle):
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    mat = np.matrix('{} {} 0; {} {} 0'.format(c, -s, s, c), np.float32)
    return mat


def write2file(image, filename):
    cv2.imwrite(filename+".png", image)
    np.savetxt(filename+"_0.txt", image[...,0], fmt='%i')
    np.savetxt(filename+"_1.txt", image[...,1], fmt='%i')
    np.savetxt(filename+"_2.txt", image[...,2], fmt='%i')


def display_landmarks(fig, landmarks, gt_landmarks):
    if len(gt_landmarks):
        fig.plot(gt_landmarks[0:17,0] ,gt_landmarks[0:17,1], marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[17:22,0],gt_landmarks[17:22,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[22:27,0],gt_landmarks[22:27,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[27:31,0],gt_landmarks[27:31,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[31:36,0],gt_landmarks[31:36,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[36:42,0],gt_landmarks[36:42,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[42:48,0],gt_landmarks[42:48,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[48:60,0],gt_landmarks[48:60,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
        fig.plot(gt_landmarks[60:68,0],gt_landmarks[60:68,1],marker='o',markersize=4,linestyle='-',color='b',lw=1)
    if len(landmarks):
        fig.plot(landmarks[0:17,0],landmarks[0:17,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[17:22,0],landmarks[17:22,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[22:27,0],landmarks[22:27,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[27:31,0],landmarks[27:31,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[31:36,0],landmarks[31:36,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[36:42,0],landmarks[36:42,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[42:48,0],landmarks[42:48,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[48:60,0],landmarks[48:60,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)
        fig.plot(landmarks[60:68,0],landmarks[60:68,1],marker='o',markersize=6,linestyle='-',color='r',lw=2)

times = {}
def tic(timename):
    times[timename] = time.time()
def toc(timename, fps=False):
    elapsed = time.time() - times[timename]
    if not fps:
        print(timename, ": %.3f" % (elapsed))
    else:
        print(timename, ": %.3f (fps: %.1f)" % (elapsed, 1/elapsed))


def plot_landmarks_on_image(calculated, expected, frame, num_landmarks=68):
    for i in range(num_landmarks):
        expected_point_color = (0,0,255)
        calculated_point_color = (255,255,255)
        # for landmark in landmarks:
        cv2.circle(frame, (int(expected[i][0]), int(expected[i][1])),
                   1, expected_point_color, thickness=2)
        cv2.circle(frame, (int(calculated[i][0]), int(calculated[i][1])),
                   1, calculated_point_color, thickness=2)
        cv2.line(frame,
                 (int(expected[i][0]), int(expected[i][1])),
                 (int(calculated[i][0]), int(calculated[i][1])),
                 (255,255,255),
                 thickness=1,
                 lineType=1)

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    print(images_batch.shape, landmarks_batch.shape)
    im_size = images_batch.size(2)
    print(im_size)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')
