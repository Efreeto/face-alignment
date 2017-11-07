# Modified from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import matplotlib.pyplot as plt
import time
import os, glob
from face_alignment.models import FAN, STN
from face_alignment.FaceLandmarksDataset import *
from shutil import copyfile

import cv2

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

USE_FAN = 1
USE_STN = 0


# TODO: move this to utils.py and connect similar points with lines like in detect_landmarks_in_image.py
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


def plotLandmarks(calculated, expected, frame):
    for i in range(68):
        expectedPointColor = (0,0,255)
        calculatedPointColor = (255,255,255)
        # for landmark in landmarks:
        cv2.circle(frame, (int(expected[0][i][0]), int(expected[0][i][1])),
                   1, expectedPointColor, thickness=2)
        cv2.circle(frame, (int(calculated[0][i][0]), int(calculated[0][i][1])),
                   1, calculatedPointColor, thickness=2)
        cv2.line(frame, (int(expected[0][i][0]), int(expected[0][i][1])), (int(calculated[0][i][0]), int(calculated[0][i][1])), (255,255,255),thickness=1,lineType=1)


data_transforms = {
     'trainset': transforms.Compose([
         RandomHorizFlip(),
         RandomRotation(),
         LandmarkCrop(256),
         CreateHeatmaps()
     ]),

    'testset': transforms.Compose([
        LandmarkCrop(256)
    ]),
}

image_datasets = {x: FaceLandmarksDataset(os.path.join('/home/cpaulse/lfpw', x),
                                        transforms=data_transforms[x])
                    for x in ['trainset', 'testset']}

dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=10, num_workers=4)
                  for x in ['trainset', 'testset']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['trainset', 'testset']}

use_gpu = torch.cuda.is_available()

"""
for i_batch, sample_batched in enumerate(dataloaders['trainset']):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show(block=True)
        break
"""

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, resume=None):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 999.99
    start_epoch = 1

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{} Learning rate:{}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['trainset', 'testset']:
            if phase == 'trainset':
                if scheduler:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, landmarks = data['image'], data['landmarks']
                minibatchsize = inputs.size()[0]
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    landmarks = landmarks.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                for batch in range(minibatchsize):
                    input = Variable(inputs[batch], volatile=False)

                    #if phase == 'train':

                    # forward
                    if USE_STN:
                        _, outputs, _ = model(input)
                    if USE_FAN:
                        outputs = model(input)

                        if phase == 'trainset':
                            loss = None
                            for i in range(model.num_modules):
                                moduleLoss = criterion(outputs[i], Variable(landmarks[batch], requires_grad=False))
                                if loss is None:
                                    loss = moduleLoss
                                else:
                                    loss += moduleLoss
                        else:
                            center, scale = center_scale_from_landmark(landmarks[batch].cpu().numpy())
                            pts, pts_img = utils.get_preds_fromhm(outputs[-1].cpu().data, center, scale)
                            loss = utils.landmark_diff(pts_img[0].numpy(), landmarks[batch].cpu().numpy())
                    else:
                        loss = criterion(outputs, landmarks[batch])
                    if phase == 'trainset':
                        loss.backward()

                # backward + optimize only if in training phase
                if phase == 'trainset':
                    optimizer.step()
                    # statistics
                    running_loss += loss.data[0]
                else:
                    running_loss += loss

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.6f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'testset' and (epoch_loss < best_loss or best_loss == 999.99):
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, True)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')


def visualize_model(model, dataloader, num_images=120):
    images_so_far = 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, original_input, landmarks = data['image'], data['original'], data['landmarks']
        original_input = cv2.cvtColor( original_input[0].numpy(), cv2.COLOR_RGB2BGR)
        if use_gpu:
            inputs, landmarks = inputs[0].cuda(), landmarks.cuda()

        inputs = Variable(inputs)
        #landmarks = Variable(landmarks)

        if USE_STN:
            _, outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
            center, scale = center_scale_from_landmark(landmarks.cpu().numpy())
            _, out_landmarks = utils.get_preds_fromhm(outputs[-1].cpu().data, center, scale)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            mse = utils.landmark_diff(out_landmarks[0].numpy(), landmarks.cpu().numpy()[0])
            running_loss += mse
            print('MSE: {}'.format(mse))
            errorText = "MSE:{:4.2f} ".format(mse)
            cv2.putText(original_input, errorText, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2, cv2.LINE_AA)
            plotLandmarks(out_landmarks, landmarks.cpu().numpy(), original_input)
            cv2.imwrite("/home/cpaulse/testOut/lmk{}.png".format(i), original_input)

            #show_landmarks(image, out_landmarks[0].numpy())

            if images_so_far == num_images:
                break
        else:
            continue    # Continue if the inner loop wasn't broken.
        break   # Inner loop was broken, break the outer.

    avg_loss = running_loss / images_so_far
    print('Loss: {:4f}'.format(avg_loss))
    # plt.show(block=True)    # when database has less images than num_images

if USE_STN:
    model_ft = STN()
if USE_FAN:
    """
    fan_weights = torch.load(
        "/home/w80053412/.face_alignment/data/2DFAN-4.pth.tar",
        map_location=lambda storage,
        loc: storage)
    fan_dict = {k.replace('module.', ''): v for k,
                v in fan_weights['state_dict'].items()}
    """
    model_ft = FAN(3)
    #model_ft.load_state_dict(fan_dict)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.MSELoss()
if 0:
    # Observe that all parameters are being optimized
    model_ft.load_state_dict(torch.load('mytraining.pth'))
    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=2.5e-4)
    # exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.1)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders=dataloaders,
                        num_epochs=10)
#    model_ft = train_model(model_ft, criterion, optimizer_ft, None, dataloaders=dataloaders,
#                        num_epochs=20)

    torch.save(model_ft.state_dict(), 'mytraining.pth')
else:
    model_ft.load_state_dict(torch.load('mytraining.pth'))


#validationdataset = FaceLandmarksDataset('/home/cpaulse/lfpw/testset', transforms=transforms.Compose([
#        LandmarkCropWithOriginal(256)
#    ]))

validationdataset = FaceLandmarksDataset('/home/cpaulse/lfpw/testset', transforms=transforms.Compose([
        ColorJitter(),
        LandmarkCropWithOriginal(256)
    ]))

dataloader = DataLoader(validationdataset, shuffle=False, num_workers=1)

visualize_model(model_ft, dataloader)
