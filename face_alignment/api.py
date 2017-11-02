from __future__ import print_function

import glob
from enum import Enum

import dlib
import torch.nn as nn
from skimage import io
from torch.autograd import Variable

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth, STN
from .utils import *
import matplotlib.pyplot as plt

from .FaceLandmarksDataset import FaceLandmarksDataset, ToTensor
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.use_face_normalization = False
        self.use_face_normalization_from_caffe = False
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        # if self.enable_cuda or self.use_cnn_face_detector:
        if 0:   # TODO: always use the generic dlib ad-hoc (cpu) face detector for now
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)   # Warning: freezes when processing large images

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignment networks
        self.face_alignment_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)
        fan_dict = {k.replace('module.', ''): v for k,
                    v in fan_weights['state_dict'].items()}

        self.face_alignment_net.load_state_dict(fan_dict)

        if self.enable_cuda:
            self.face_alignment_net.cuda()
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda()
            self.depth_prediciton_net.eval()

        self.face_normalization_net = STN()
        if self.enable_cuda:
            self.face_normalization_net.cuda()

    def detect_faces(self, image):
        """Run the dlib face detector over an image

        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.

        Returns:
            Returns a list of detected faces
        """
        return self.face_detector(image, 1)

    def make_rct_files(self, path):

        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(sorted(glob.glob(os.path.join(path, files))))

        for image_name in images_list:
            image = io.imread(image_name)
            tic("detect_faces")
            detected_faces = self.detect_faces(image)
            toc("detect_faces")
            if len(detected_faces) == 0:
                print(image_name, " - Warning: No faces were detected")
                continue

            # TODO: always one face per image (most centric one)
            if len(detected_faces) > 1:
                centric_face_index = 0
                centric_face_center_diff = 9999.9
                for i, d in enumerate(detected_faces):
                    # if self.enable_cuda or self.use_cnn_face_detector:
                    if 0:   # TODO: always use the generic dlib ad-hoc (cpu) face detector for now
                        d = d.rect
                    center_diff = abs(
                        (d.right() + d.left() - image.shape[1]) + (d.top() + d.bottom() - image.shape[0]))
                    if center_diff < centric_face_center_diff or centric_face_center_diff == 9999.9:
                        centric_face_center_diff = center_diff
                        centric_face_index = i
                detected_faces = [detected_faces[centric_face_index]]

            # if self.enable_cuda or self.use_cnn_face_detector:
            if 0:   # TODO: always use the generic dlib ad-hoc (cpu) face detector for now
                d = detected_faces[0].rect
                np.savetxt(os.path.splitext(image_name)[0] + '.rct_dlib_cuda',
                           (d.left(), d.top(), d.right(), d.bottom()), fmt='%d', newline=' ')
            else:
                d = detected_faces[0]
                np.savetxt(os.path.splitext(image_name)[0] + '.rct_dlib_cpu',
                           (d.left(), d.top(), d.right(), d.bottom()), fmt='%d', newline=' ')

    def get_landmarks(self, input_image, type, all_faces=False):
        print(input_image, " ---")
        # tic("total")
        try:
            image = io.imread(input_image)
        except IOError:
            print("Error opening file")
            return None, None, None

        # tic("detect_faces")
        detected_faces = self.detect_faces(image)
        # toc("detect_faces")
        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            # toc("total", fps=True)
            return None, None, None

        # TODO: always one face per image (most centric one)
        if len(detected_faces) > 1:
            centric_face_index = 0
            centric_face_center_diff = -9.9
            for i, d in enumerate(detected_faces):
                # if self.enable_cuda or self.use_cnn_face_detector:
                if 0:   # TODO: always use the generic dlib ad-hoc (cpu) face detector for now
                    d = d.rect
                center_diff = abs((d.right()+d.left()-image.shape[1]) + (d.top()+d.bottom()-image.shape[0]))
                if center_diff < centric_face_center_diff or centric_face_center_diff == -9.9:
                    centric_face_center_diff = center_diff
                    centric_face_index = i
            detected_faces = [detected_faces[centric_face_index]]

        landmarks = []
        gt_landmarks = []
        proposal_images = []
        frontal_images = []
        images_so_far = 0
        running_error = 0.0
        for i, d in enumerate(detected_faces):
            if i > 1 and not all_faces:
                break
            # if self.enable_cuda or self.use_cnn_face_detector:
            if 0:   # TODO: always use the generic dlib ad-hoc (cpu) face detector for now
                d = d.rect

            center = torch.FloatTensor(
                [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                 (d.bottom() - d.top()) / 2.0])
            center[1] = center[1] - (d.bottom() - d.top()) * 0.1
            scale = (d.right() - d.left() + d.bottom() - d.top()) / 200.0

            if self.use_face_normalization or self.use_face_normalization_from_caffe:
                inp = crop(image, center, scale, resolution=480.0)
            else:
                inp = crop(image, center, scale)
                front_img = inp
            prop_img = inp
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float().div(255.0).unsqueeze_(0)
            if self.enable_cuda:
                inp = inp.cuda()
            inp = Variable(inp, volatile=True)

            if self.use_face_normalization:
                tic(input_image + " - face_normalization")
                inp, _, theta = self.face_normalization_net(inp)
                toc(input_image + " - face_normalization")
                front_img = inp.data.cpu().numpy()
                front_img = front_img[-1].transpose(1, 2, 0)
            elif self.use_face_normalization_from_caffe:
                theta = torch.from_numpy(np.loadtxt(os.path.splitext(input_image)[0] + '.theta_cpu', skiprows=1).astype('float32')).view(1, 2, 3).cuda()
                # theta[0,1,0] = theta[0,1,0] * -1.0
                # theta[0,0,1] = theta[0,0,1] * -1.0
                # theta[0,0,2] = theta[0,0,2] * -1.0
                # theta[0,1,2] = theta[0,1,2] * -1.0
                grid = nn.functional.affine_grid(theta, torch.Size([1, 3, 256, 256]))
                inp = nn.functional.grid_sample(inp, grid)
                front_img = inp.data.cpu().numpy()
                front_img = front_img[-1].transpose(1, 2, 0)

            # tic("face_alignment")
            out = self.face_alignment_net(inp)[-1].data
            # toc("face_alignment")
            if self.flip_input:
                out += flip(self.face_alignment_net(Variable(flip(inp.data),
                                                             volatile=True))[-1].data.cpu(), is_label=True)

            if self.use_face_normalization or self.use_face_normalization_from_caffe:
                theta_inv = torch.eye(3)
                theta_inv[0:2] = theta.data[0]
                theta_inv = torch.inverse(theta_inv)[0:2].unsqueeze(0).cuda()
                grid = nn.functional.affine_grid(theta_inv, torch.Size([1, 68, 64, 64]))
                out = nn.functional.grid_sample(out, grid).data.cpu()

            pts, pts_img = get_preds_fromhm(out.cpu(), center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            images_so_far += 1
            if type == 0:    #land110
                ground_truth = np.loadtxt(os.path.splitext(input_image)[0] + '.land', skiprows=1)
                ground_truth = np.vstack((ground_truth[0:32:2], ground_truth[32:64], ground_truth[88:108]))
            elif type == 1:    #8W
                ground_truth = np.loadtxt(os.path.splitext(input_image)[0] + '.pts')
            elif type == 2:    #300W
                ground_truth = np.loadtxt(os.path.splitext(input_image)[0] + '.pts', skiprows=3, comments='}')
            mse = ((pts_img.numpy() - ground_truth) ** 2).mean(axis=None)
            running_error += mse

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256))
                for i in range(68):
                    if pts[i, 0] > 0:
                        heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).view(1, 68, 256, 256).float()
                if self.enable_cuda:
                    heatmaps = heatmaps.cuda()
                # tic(input_image + " - depth_prediction")
                depth_pred = self.depth_prediciton_net(
                    Variable(
                        torch.cat(
                            (inp.data, heatmaps), 1), volatile=True)).data.cpu().view(
                    68, 1)
                # toc(input_image + " - depth_prediction")
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())
            gt_landmarks.append(ground_truth)
            proposal_images.append(prop_img)
            frontal_images.append(front_img)

        avg_error = running_error / images_so_far
        print("MSE: {:4f}".format(avg_error))
        # toc("total", fps=True)
        return landmarks, gt_landmarks, proposal_images, frontal_images, avg_error

    def process_folder(self, path, type, all_faces=False):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(sorted(glob.glob(os.path.join(path, files))))

        predictions = []
        running_error = 0.0
        for image_name in images_list:
            predictions.append(
                [image_name, self.get_landmarks(image_name, type, all_faces)])

        for preds in predictions:
            running_error += preds[1][4]
        avg_error = running_error / len(predictions)
        print("Average MSE: {:4f}".format(avg_error))
        return predictions

    def train_STN_standalone(self, path, type, save_state_file):
        image_datasets = {x: FaceLandmarksDataset(path, type,
                                                  transforms=transforms.Compose([
                                                      ToTensor()
                                                  ]))
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.face_normalization_net.parameters(), lr=0.000001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

        num_epochs = 20

        since = time.time()

        best_model_wts = self.face_normalization_net.state_dict()
        best_loss = 999.99

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.face_normalization_net.train(True)  # Set model to training mode
                else:
                    self.face_normalization_net.train(False)  # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    # get the inputs
                    data_image, data_landmarks = data['image'][-1], data['landmarks'][-1] # Assume one face per image

                    image = data_image.cpu().numpy().transpose((1, 2, 0))
                    d = data_landmarks[0:2].view(4)
                    landmarks = data_landmarks[2::]

                    center = torch.FloatTensor(
                        [d[2] - (d[2] - d[0]) / 2.0, d[3] -
                         (d[3] - d[1]) / 2.0])
                    center[1] = center[1] - (d[3] - d[1]) * 0.1
                    scale = (d[2] - d[0] + d[3] - d[1]) / 200.0
                    input = crop(image, center, scale, resolution=480.0)
                    input = torch.from_numpy(input.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)

                    ### Rotate the image and landmarks
                    rigged_points3 = Variable(torch.ones(landmarks.shape[0], landmarks.shape[1] + 1)).float().cuda()
                    rigged_points3[:,0:2] = landmarks

                    # wrap them in Variable
                    if self.enable_cuda:
                        input, landmarks = input.cuda(), landmarks.cuda()
                    input = Variable(input, requires_grad = True)
                    landmarks = Variable(landmarks)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    _, _, theta = self.face_normalization_net(input)

                    theta3 = Variable(torch.eye(3)).cuda()
                    theta3[0:2] = theta
                    output3 = torch.matmul(rigged_points3, theta3)
                    output = output3[:,0:2]

                    loss = criterion(output, landmarks)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0]

                epoch_loss = running_loss / dataset_sizes[phase]

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and (epoch_loss < best_loss or best_loss == 999.99):
                    best_loss = epoch_loss
                    best_model_wts = self.face_normalization_net.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        self.face_normalization_net.load_state_dict(best_model_wts)

        # save the weights to a file
        torch.save(self.face_normalization_net.state_dict(), save_state_file)


    def train_STN_with_FAN(self, path, type, save_state_file):
        image_datasets = {x: FaceLandmarksDataset(path, type,
                                                  transforms=transforms.Compose([
                                                      ToTensor()
                                                  ]))
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.face_normalization_net.parameters(), lr=0.000001, momentum=0.9)
        # optimizer = optim.SGD(self.face_normalization_net.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.RMSprop(self.face_normalization_net.parameters(), lr=0.00025, eps=1.e-8)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

        num_epochs = 6
        ##########
        since = time.time()

        best_model_wts = self.face_normalization_net.state_dict()
        best_loss = 999.99

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.face_normalization_net.train(True)  # Set model to training mode
                else:
                    self.face_normalization_net.train(False)  # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    # get the inputs
                    data_image, data_landmarks = data['image'][-1], data['landmarks'][-1] # Assume one face per image

                    image = data_image.cpu().numpy().transpose((1, 2, 0))
                    d = data_landmarks[0:2].view(4)
                    landmarks = data_landmarks[2::]

                    center = torch.FloatTensor(
                        [d[2] - (d[2] - d[0]) / 2.0, d[3] -
                         (d[3] - d[1]) / 2.0])
                    center[1] = center[1] - (d[3] - d[1]) * 0.1
                    scale = (d[2] - d[0] + d[3] - d[1]) / 200.0
                    input = crop(image, center, scale, resolution=480.0)
                    input = torch.from_numpy(input.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)

                    # create heat maps from ground truth landmarks
                    nFeatures = len(landmarks)
                    # reference_heatmaps = torch.Tensor(4, 1, nFeatures, 64, 64)
                    # for stack in range(4):
                    #     heatmap = np.zeros((nFeatures, 64, 64))
                    #     for i in range(nFeatures):
                    #         heatmap[i] = draw_gaussian(heatmap[i], transform(landmarks[i], center, scale, 64), 1)
                    #         reference_heatmaps[stack] = torch.from_numpy(heatmap).view(1, nFeatures, 64, 64).float()
                    reference_heatmaps = torch.Tensor(1, 1, nFeatures, 64, 64)
                    heatmap = np.zeros((nFeatures, 64, 64))
                    for i in range(nFeatures):
                        heatmap[i] = draw_gaussian(heatmap[i], transform(landmarks[i], center, scale, 64), 1)
                    reference_heatmaps[0] = torch.from_numpy(heatmap).view(1, nFeatures, 64, 64).float()

                    # wrap them in Variable
                    if self.enable_cuda:
                        input, reference_heatmaps = input.cuda(), reference_heatmaps.cuda()
                    input = Variable(input, requires_grad = True)
                    reference_heatmaps = Variable(reference_heatmaps)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    frontal_img, _, theta = self.face_normalization_net(input)

                    hm = self.face_alignment_net(frontal_img)
                    # pts, pts_img = get_preds_fromhm(hm[-1].data.cpu(), center, scale)
                    # pts, pts_img = pts.view(nFeatures, 2) * 4, pts_img.view(nFeatures, 2)
                    #
                    # loss = criterion(pts_img, Variable(landmarks))

                    # loss = criterion(hm[-1], reference_heatmaps[-1])

                    # loss = None
                    # for i in range(4):
                    #     stackLoss = criterion(hm[i], reference_heatmaps[i])
                    #     if loss is None:
                    #         loss = stackLoss
                    #     else:
                    #         loss += stackLoss


                    loss = None
                    for i in range(self.face_alignment_net.num_modules):
                        moduleLoss = criterion(hm[i], reference_heatmaps[-1])
                        if loss is None:
                            loss = moduleLoss
                        else:
                            loss += moduleLoss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0]

                epoch_loss = running_loss / dataset_sizes[phase]

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and (epoch_loss < best_loss or best_loss == 999.99):
                    best_loss = epoch_loss
                    best_model_wts = self.face_normalization_net.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        self.face_normalization_net.load_state_dict(best_model_wts)

        # save the weights to a file
        torch.save(self.face_normalization_net.state_dict(), save_state_file)

    def use_STN(self, load_state_file):
        self.use_face_normalization = True

        self.face_normalization_net.eval()

        # self.face_normalization_net.load_state_dict(torch.load(load_state_file))

    def use_STN_from_caffe(self):
        self.use_face_normalization_from_caffe = True
        self.use_face_normalization = False
