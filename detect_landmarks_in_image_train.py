# Modified from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import time
import os, glob
from face_alignment.models import FAN, STN
from face_alignment.FaceLandmarksDataset import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

USE_FAN = 0
USE_STN = 1

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


# # Data augmentation and normalization for training
# # Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Scale((224,224)),
#         transforms.ToTensor()
#     ]),
#     'val': transforms.Compose([
#         transforms.Scale((224,224)),
#         transforms.ToTensor()
#     ]),
# }


image_datasets = {x: FaceLandmarksDataset('300W/val',
                                          transforms=transforms.Compose([
                            Rescale((480,480)),
                            ToTensor()
                        ]))
                    for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

for i_batch, sample_batched in enumerate(dataloaders['train']):
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 999.99

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, landmarks = data['image'], data['landmarks']

                # wrap them in Variable
                if use_gpu:
                    inputs, landmarks = Variable(inputs.cuda()), Variable(landmarks.cuda())
                else:
                    inputs, landmarks = Variable(inputs), Variable(landmarks)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if USE_STN:
                    _, outputs = model(inputs)
                if USE_FAN:
                    outputs = model(inputs)
                    # pts, pts_img = get_preds_fromhm(outputs, center, scale)
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
                landmarks = landmarks.view(1,-1)
                loss = criterion(outputs, landmarks)

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
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, criterion, num_images=6):
    images_so_far = 0
    running_loss = 0.0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, landmarks = data['image'], data['landmarks']
        if use_gpu:
            inputs, landmarks = Variable(inputs.cuda()), Variable(landmarks.cuda())
        else:
            inputs, landmarks = Variable(inputs), Variable(landmarks)

        _, outputs, _ = model(inputs)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            image = inputs.data[j].cpu().numpy().transpose((1, 2, 0))
            out_landmarks = outputs.data[j].cpu().numpy().reshape(-1, 2)
            landmarks = landmarks.view(1, -1)
            loss = criterion(outputs, landmarks)
            running_loss += loss.data[0]
            show_landmarks(image, out_landmarks)

            if images_so_far == num_images:
                break
        else:
            continue    # Continue if the inner loop wasn't broken.
        break   # Inner loop was broken, break the outer.

    avg_loss = running_loss / images_so_far
    print('Loss: {:4f}'.format(avg_loss))
    plt.show(block=True)    # when database has less images than num_images

if USE_STN:
    model_ft = STN()
if USE_FAN:
    fan_weights = torch.load(
        "/home/w80053412/.face_alignment/data/2DFAN-4.pth.tar",
        map_location=lambda storage,
        loc: storage)
    fan_dict = {k.replace('module.', ''): v for k,
                v in fan_weights['state_dict'].items()}
    model_ft = FAN(4)
    model_ft.load_state_dict(fan_dict)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.MSELoss()
if 0:
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=2)

    torch.save(model_ft.state_dict(), 'mytraining.pth')
else:
    model_ft.load_state_dict(torch.load('mytraining.pth'))

visualize_model(model_ft, criterion)