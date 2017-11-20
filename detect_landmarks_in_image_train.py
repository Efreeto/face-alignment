from __future__ import print_function, division

import argparse

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import matplotlib.pyplot as plt
import time
import face_alignment.models
from face_alignment.models import FAN, STEFAN
from face_alignment.FaceLandmarksDataset import *
from shutil import copyfile

import cv2

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in face_alignment.models.__dict__
                     if not name.startswith("__")
                     and callable(face_alignment.models.__dict__[name]))

# Training settings
parser = argparse.ArgumentParser(description='Train FAN/STN model')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='FAN',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: FAN)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.00025, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help= 'resume training from checkpoint file in logging directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--loss-type', default='MSELoss', type=str, metavar='PATH',
                    help='loss function (default: MSELoss)')
parser.add_argument('--num-FAN-modules', default=4, type=int, metavar='PATH',
                    help='number of FAN modules (default: 4)')
parser.add_argument('-nl', '--num-landmarks', default=68, type=int, metavar='PATH',
                    help='number of landmarks (default: 68)')
parser.add_argument('-l', '--log-dir', default='train_log', type=str, metavar='PATH',
                    help='logging directory (default: train_log)')
parser.add_argument('-lp', '--log-progress', default=True, type=bool,
                    help='log intermediate loss values to csv (default: True)')


use_gpu = torch.cuda.is_available()

def weights_init(m):
    """
    taken from https://github.com/pytorch/examples/blob/62d5ca57af2c33c96c40010c115e5ff34136abb5/dcgan/main.py#L96
    :param m: model
    :return: model with randomized weights
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_model(model,
                criterion,
                optimizer,
                dataloaders,
                scheduler=None,
                num_epochs=25,
                results_dir="train_log",
                resume=True,
                checkpoint_file=None,
                log_progress=True):
    """
        Use the criterion, optimizer and Train/Validate data loaders to train the model
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = -9.9
    start_epoch = 1

    display_mode = True
    use_manual_rotation = False
    loss_image = False
    loss_manual_theta = False    # requires use_manual_rotation
    loss_manual_landmarks = False    # requires use_manual_rotation
    loss_hm = True    # FAN model default
    loss_hm_landmarks = False  # Get landmarks from hm and use gradients from them

    if use_gpu:
        model = model.cuda()

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if log_progress:
        progress_file = open(os.path.join(results_dir, "progress.csv"), "a")
        progress_file.write('epoch, learning_rate, batch_size, trainset_loss, testset_loss, testset_max_loss\n')

    if resume:
        if checkpoint_file is None:
            checkpoint_file = "checkpoint.pth.tar"
        resume_file = os.path.join(results_dir, checkpoint_file)
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    for epoch in range(start_epoch, num_epochs+1):
        print('Epoch {}/{} Learning rate:{}'.format(epoch, num_epochs, optimizer.param_groups[0]['lr']))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['trainset', 'testset']:
            if phase == 'trainset':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for j, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs = data['image']
                if use_manual_rotation:
                    inputs = torch.cat((inputs, data['image_rot']))
                if use_gpu:
                    inputs = inputs.cuda()
                # wrap them in Variable
                inputs = Variable(inputs, volatile=False)
                # # get the inputs
                # inputs, heatmaps = data['image'], data['heatmaps']
                # if use_gpu:
                #     inputs, heatmaps = inputs.cuda(), heatmaps.cuda()
                # # wrap them in Variable
                # inputs, heatmaps = Variable(inputs, volatile=False), Variable(heatmaps, requires_grad=False)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                if loss_image:
                    input_org_Var = Variable(torch.from_numpy(input_org.transpose((2, 0, 1))).float().cuda(), requires_grad=False)

                    # loss = criterion(frontal_img.mul(255.0), input_org_Var)
                    # loss = criterion(frontal_img[:, :, 64:192,64:192].mul(255.0), input_org_Var[64:192,64:192])
                    loss = criterion(frontal_img[:, :, 48:208,48:208].mul(255.0), input_org_Var[:,48:208,48:208])

                if loss_hm_landmarks:
                    hm = self.face_alignment_net(frontal_img)[-1]
                    if self.flip_input:
                        hm += flip(self.face_alignment_net(Variable(flip(frontal_img)))[-1], is_label=True)

                    theta_inv = Variable(torch.eye(3))
                    theta_inv[0:2] = theta[0]
                    theta_inv = torch.inverse(theta_inv)[0:2].unsqueeze(0).cuda()
                    grid = nn.functional.affine_grid(theta_inv, torch.Size([1, 68, 64, 64]))
                    hm = nn.functional.grid_sample(hm, grid)

                    pts, pts_img = get_preds_fromhm_Variable(hm, center, scale)
                    pts, pts_img = pts.view(-1, 2) * 4, pts_img.view(-1, 2)

                    loss = criterion(pts_img, landmarks)

                    # pts, pts_img = get_preds_fromhm_Variable(hm, center, scale)
                    # pts, pts_img = pts.view(-1, 2) * 4, pts_img.view(-1, 2)
                    #
                    # # grid = nn.functional.affine_grid(theta.data, torch.Size([1, 1, 68, 2]))
                    # # landmarks_rot = nn.functional.grid_sample(landmarks.unsqueeze(0).unsqueeze(0), grid)
                    # # loss = criterion(pts_img, landmarks_rot)
                    #
                    # theta_ = theta[0].data.t()
                    # landmarks3 = torch.cat((landmarks.data, torch.ones(68,1)), dim=1)
                    # landmarks_rot = torch.matmul(landmarks3, theta_)[:,:2]
                    # loss = criterion(pts_img, landmarks_rot)

                    landmarks_disp = pts_img.data.cpu().numpy()
                    gt_landmarks_disp = landmarks.numpy()

                if loss_hm:
                    heatmaps = data['heatmaps']
                    if use_gpu:
                        heatmaps = heatmaps.cuda()
                    heatmaps = Variable(heatmaps, requires_grad=False)

                    loss = criterion(outputs[-1], heatmaps)
                    # loss = None
                    # for i in range(model.num_modules):
                    #     module_loss = criterion(outputs[i], heatmaps)
                    #     if loss is None:
                    #         loss = module_loss
                    #     else:
                    #         loss += module_loss

                    # if phase == 'trainset':
                    #     heatmaps = data['heatmaps']
                    #     if use_gpu:
                    #         heatmaps = heatmaps.cuda()
                    #     heatmaps = Variable(heatmaps, requires_grad=False)
                    #
                    #     # loss = criterion(outputs[-1], heatmaps)
                    #     loss = None
                    #     for i in range(model.num_modules):
                    #         module_loss = criterion(outputs[i], heatmaps)
                    #         if loss is None:
                    #             loss = module_loss
                    #         else:
                    #             loss += module_loss
                    # else:
                    #     landmarks = data['landmarks']
                    #     if use_gpu:
                    #         landmarks = landmarks.cuda()
                    #     landmarks = Variable(landmarks, requires_grad=False)
                    #
                    #     center, scale = utils.center_scale_from_bbox(utils.bounding_box(landmarks))
                    #     pts, pts_img = utils.get_preds_fromhm(outputs[-1].cpu().data, center, scale)
                    #     loss = utils.landmark_diff(landmarks.cpu().numpy(), pts_img[0].numpy())

                if loss_manual_landmarks:
                    landmarks_rot_Var = Variable(landmarks_rot - origin_rot, requires_grad=False)
                    landmarks_rot_Var = torch.matmul(landmarks_rot_Var, theta[0])[:,:2]
                    landmarks_rot_Var = landmarks_rot_Var + Variable(origin_org, requires_grad=False)

                    loss = criterion(landmarks_rot_Var, landmarks)

                    if j == 2:
                        landmarks_disp = landmarks_rot_Var.data.cpu().numpy()
                        gt_landmarks_disp = landmarks.data.cpu().numpy()

                        # this is correct. Just another way to get gt_landmark_disp
                        # gt_landmarks_disp = landmarks_rot - origin_rot
                        # gt_landmarks_disp = torch.matmul(gt_landmarks_disp, manual_theta.data)[:,:2]
                        # gt_landmarks_disp = gt_landmarks_disp + origin_org
                        # gt_landmarks_disp = gt_landmarks_disp.cpu().numpy()

                        display_landmarks(image, landmarks_disp, gt_landmarks_disp, "Results")

                if loss_manual_theta:
                    manual_theta = torch.FloatTensor(utils.transformation_matrix(0)).repeat(data['theta'].shape[0],1,1)
                    manual_theta = torch.cat((manual_theta, data['theta']))
                    if self.enable_cuda:
                        manual_theta = manual_theta.cuda()
                    manual_theta = Variable(manual_theta, requires_grad=False)
                    w_theta = 100.0
                    loss = criterion(theta*w_theta, manual_theta*w_theta)

                if display_mode and phase == 'train' and j == 10:
                    def TensorToImg(tensor):
                        return tensor.cpu().numpy().transpose(1, 2, 0)
                    image = io.imread(data['filename'][0])
                    input_org = TensorToImg(data['image'][0])
                    input_rot = TensorToImg(input[0].data)
                    output_img = TensorToImg(frontal_img[0].data)

                    fig = plt.figure(figsize=(9, 6), tight_layout=True)
                    ax = fig.add_subplot(1, 4, 1)
                    ax.axis('off')
                    ax.imshow(image)
                    ax = fig.add_subplot(2, 4, 2)
                    ax.axis('off')
                    ax.imshow(input_org)
                    if use_manual_rotation:
                        ax = fig.add_subplot(2, 4, 3)
                        ax.axis('off')
                        manual_theta_text = manual_theta[0].data
                        ax.title.set_text("{:.3f} {:.3f} {:.1f}\n{:.3f} {:.3f} {:.1f}".format(manual_theta_text[0,0], manual_theta_text[0,1], manual_theta_text[0,2],
                                                                                              manual_theta_text[1,0], manual_theta_text[1,1], manual_theta_text[1,2]))
                        ax.imshow(input_rot)
                    ax = fig.add_subplot(2, 4, 4)
                    ax.axis('off')
                    theta_text = theta[0].data
                    ax.title.set_text(
                        "{:.3f} {:.3f} {:.1f}\n{:.3f} {:.3f} {:.1f}".format(theta_text[0, 0], theta_text[0, 1], theta_text[0, 2],
                                                                            theta_text[1, 0], theta_text[1, 1], theta_text[1, 2]))
                    ax.imshow(output_img)

                    idx_rot = int(data['theta'].shape[0])
                    input_rot = TensorToImg(input[idx_rot].data)
                    output_img = TensorToImg(frontal_img[idx_rot].data)

                    if use_manual_rotation:
                        ax = fig.add_subplot(2, 4, 7)
                        ax.axis('off')
                        manual_theta_text = manual_theta[idx_rot].data
                        ax.title.set_text("{:.3f} {:.3f} {:.1f}\n{:.3f} {:.3f} {:.1f}".format(manual_theta_text[0, 0],
                                                                                              manual_theta_text[0, 1],
                                                                                              manual_theta_text[0, 2],
                                                                                              manual_theta_text[1, 0],
                                                                                              manual_theta_text[1, 1],
                                                                                              manual_theta_text[1, 2]))
                        ax.imshow(input_rot)
                    ax = fig.add_subplot(2, 4, 8)
                    ax.axis('off')
                    theta_text = theta[idx_rot].data
                    ax.title.set_text(
                        "{:.3f} {:.3f} {:.1f}\n{:.3f} {:.3f} {:.1f}".format(theta_text[0, 0], theta_text[0, 1],
                                                                            theta_text[0, 2],
                                                                            theta_text[1, 0], theta_text[1, 1],
                                                                            theta_text[1, 2]))
                    ax.imshow(output_img)
                    plt.show(block=False)

                # backward + optimize only if in training phase
                if phase == 'trainset':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'testset':
                testset_loss = epoch_loss
                if scheduler:
                    scheduler.step(testset_loss)
            else:
                trainset_loss = epoch_loss

            print('{} Loss: {:.6f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'testset':
                if best_loss > epoch_loss or best_loss == -9.9:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, epoch_loss <= best_loss, dir=results_dir)

        if log_progress:
            progress_file.write('{}, {}, {}, {}, {}\n'.format(epoch, optimizer.param_groups[0]['lr'], dataloaders[phase].batch_size, trainset_loss, testset_loss))
            progress_file.flush()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(state, is_best, dir=".", filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        copyfile(checkpoint_path, os.path.join(dir, 'model_best.pth.tar'))


def evaluate_model(model, dataloader, num_images=999, results_dir='test_out'):
    """
    apply the model to a selection of test data, and save images with overlayed predicted landmarks
    :param model: model for prediction
    :param dataloader: dataloader containing evaluation images and reference landmarks
    :param num_images: number of images to render calculated landmarks
    :param results_dir: directory for output
    :return:
    """
    images_so_far = 0
    running_loss_max = 0.0
    running_loss_sum = 0.0

    errors_file = open(os.path.join(results_dir, "errors.csv"), "w")
    errors_file.write("max_error,sum_error\n")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for i, data in enumerate(dataloader):
        inputs, filename, landmarks = data['image'], data['filename'][0], data['landmarks'][0].cpu().numpy()
        original_input = cv2.imread(filename)
        if use_gpu:
            model = model.cuda()
            inputs = inputs.cuda()
        inputs = Variable(inputs)

        outputs = model(inputs)
        center, scale = utils.center_scale_from_bbox(utils.bounding_box(landmarks))
        _, out_landmarks = utils.get_preds_fromhm(outputs[-1].cpu().data, center, scale)
        out_landmarks = out_landmarks[0].numpy()

        images_so_far += 1
        max_error, sum_error = utils.landmark_diff(landmarks, out_landmarks)
        errors_file.write("{},{}\n".format(max_error, sum_error))
        running_loss_max += max_error
        running_loss_sum += sum_error
        errorText = "MaxErr:{:4.3f} ".format(max_error)
        cv2.putText(original_input, errorText, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2)
        utils.plot_landmarks_on_image(out_landmarks, landmarks, original_input, model.num_landmarks)
        cv2.imwrite("{}/lmk{}.png".format(results_dir,i), original_input)

    avg_loss_max = running_loss_max / images_so_far
    print('Loss: {:4f}'.format(avg_loss_max))
    # plt.show(block=True)    # when database has less images than num_images

def newFAN(num_modules=4, num_landmarks=68):
    """ Create a new FAN model with randomized weights """
    model = FAN(num_modules, num_landmarks)
    model.apply(weights_init)
    return model


def main():
    """Parse command line input and train or evaluate model."""
    global args
    args = parser.parse_args()

    args.data = os.path.expanduser(args.data)
    args.log_dir = os.path.expanduser(args.log_dir)

    data_transforms = {
        'trainset': transforms.Compose([
            RandomRotation(40, 5),
            LandmarkCrop(480),
            CreateHeatmaps(),
            ToTensor()
        ]),

        'testset': transforms.Compose([
            RandomRotation(40, 5),
            LandmarkCrop(480),
            CreateHeatmaps(),
            ToTensor()
        ]),
    }
    # data_transforms = {
    #     'trainset': transforms.Compose([
    #         FaceColorJitter(),
    #         RandomHorizFlip(),
    #         RandomRotation(),
    #         LandmarkCrop(256),
    #         CreateHeatmaps(n_features=args.num_landmarks),
    #         ToTensor()
    #     ]),
    #
    #     'testset': transforms.Compose([
    #         LandmarkCrop(256),
    #         CreateHeatmaps(n_features=args.num_landmarks),
    #         ToTensor()
    #     ]),
    # }

    datatype = 1 if args.num_landmarks == 68 else 2
    dataset = {x: FaceLandmarksDataset(os.path.join(args.data, x),
                                      transforms=data_transforms[x], type=datatype)
              for x in ['trainset', 'testset']}

    if args.arch == 'FAN':
        model_ft = newFAN(args.num_FAN_modules, args.num_landmarks)
    elif args.arch == 'STEFAN':
        model_ft = STEFAN(args.num_FAN_modules, args.num_landmarks)
    else:
        model_ft = None

    if args.evaluate:
        model_ft.load_state_dict(torch.load(os.path.join(args.log_dir,'checkpoint.pth.tar'))['state_dict'])
        dataloader = DataLoader(dataset['testset'], shuffle=False, num_workers=1)
        evaluate_model(model_ft, dataloader, results_dir=args.log_dir)
    else:
        optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=args.lr)
        interval_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,milestones=[15,30])
        dataloaders = {'trainset': DataLoader(dataset['trainset'], shuffle=True, batch_size=args.batch_size, num_workers=args.workers),
                       'testset': DataLoader(dataset['testset'], shuffle=False, batch_size=1, num_workers=1)}

        if args.loss_type is not 'MSELoss':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        if use_gpu:
            criterion = criterion.cuda()

        model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders, interval_scheduler,
                            num_epochs=args.epochs, results_dir=args.log_dir, resume=args.resume, log_progress=args.log_progress)


if __name__ == '__main__':
    main()