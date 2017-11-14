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
from face_alignment.models import FAN, STN
from face_alignment.FaceLandmarksDataset import *
from shutil import copyfile

import cv2

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

USE_FAN = 1
USE_STN = 0

model_names = sorted(name for name in face_alignment.models.__dict__
                     if name.islower() and not name.startswith("__")
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
    best_loss = 999.99
    start_epoch = 1

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

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{} Learning rate:{}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['trainset', 'testset']:
            if phase == 'trainset':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, landmarks = data['image'], data['landmarks']
                num_in_batch = inputs.shape[0]
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    landmarks = landmarks.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                for batch in range(num_in_batch):
                    input = Variable(inputs[batch], volatile=False)

                    # forward
                    if USE_STN:
                        _, outputs, _ = model(input)
                    if USE_FAN:
                        outputs = model(input)

                        if phase == 'trainset':
                            loss = None
                            for i in range(model.num_modules):
                                module_loss = criterion(outputs[i], Variable(landmarks[batch], requires_grad=False))
                                if loss is None:
                                    loss = module_loss
                                else:
                                    loss += module_loss
                        else:
                            center, scale = center_scale_from_landmark(landmarks[batch].cpu().numpy())
                            pts, pts_img = utils.get_preds_fromhm(outputs[-1].cpu().data, center, scale)
                            loss = utils.landmark_diff(landmarks[batch].cpu().numpy(), pts_img[0].numpy())
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
                    running_loss += loss[0]

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
                if best_loss > epoch_loss:
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
    print('Best val Loss: {:4f}'.format(best_loss))

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
        inputs, original_input, landmarks = data['image'], data['original'], data['landmarks']
        original_input = cv2.cvtColor( original_input[0].numpy(), cv2.COLOR_RGB2BGR)
        if use_gpu:
            model = model.cuda()
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
            max_error, sum_error = utils.landmark_diff(landmarks.cpu().numpy()[0], out_landmarks[0].numpy())
            errors_file.write("{},{}\n".format(max_error, sum_error))
            running_loss_max += max_error
            running_loss_sum += sum_error
            errorText = "MaxErr:{:4.3f} ".format(max_error)
            cv2.putText(original_input, errorText, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2)
            utils.plot_landmarks_on_image(out_landmarks, landmarks.cpu().numpy(), original_input, model.num_landmarks)
            cv2.imwrite("{}/lmk{}.png".format(results_dir,i), original_input)

            #show_landmarks(image, out_landmarks[0].numpy())

            if images_so_far == num_images:
                break
        else:
            continue    # Continue if the inner loop wasn't broken.
        break   # Inner loop was broken, break the outer.

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
            FaceColorJitter(),
            RandomHorizFlip(),
            RandomRotation(),
            LandmarkCrop(256),
            CreateHeatmaps(n_features=args.num_landmarks)
        ]),

        'testset': transforms.Compose([
            LandmarkCropWithOriginal(256)
        ]),
    }

    datatype = 1 if args.num_landmarks == 68 else 2
    dataset = {x: FaceLandmarksDataset(os.path.join(args.data, x),
                                      transforms=data_transforms[x], type=datatype)
              for x in ['trainset', 'testset']}

    if args.arch == 'FAN':
        model_ft = newFAN(args.num_FAN_modules, args.num_landmarks)
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