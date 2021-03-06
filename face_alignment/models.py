import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, num_modules=1, num_landmarks=68):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.num_landmarks = num_landmarks

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            self.num_landmarks, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.num_landmarks,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs


class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class STN(nn.Module):

    def __init__(self):
        super(STN, self).__init__()
        self.downsample = nn.AvgPool2d(8) # or stride=448/60
        self.net1_conv1 = nn.Conv2d(3,20,5)
        self.net1_PReLU1 = nn.PReLU(20)
        self.net1_pool = nn.MaxPool2d(2)
        self.net1_conv2 = nn.Conv2d(20,48,5)
        self.net1_PReLU2 = nn.PReLU(48)
        self.net1_conv3 = nn.Conv2d(48,64,3)
        self.net1_PReLU3 = nn.PReLU(64)
        self.net1_conv4 = nn.Conv2d(64,80,3)
        self.net1_PReLU4 = nn.PReLU(80)
        self.net1_fc5_1 = nn.Linear(80*3*3, 512)
        self.net1_PReLU5 = nn.PReLU(512)
        self.net1_drop6 = nn.Dropout2d(0.2)
        self.net1_68point = nn.Linear(512, 136)
        self.net1_PReLU6 = nn.PReLU(136)
        self.loc_reg_ = nn.Linear(136, 6)
        # self.loc_reg_ = nn.Linear(136, 4)

        self.base_theta = Variable(torch.eye(2, 3), requires_grad=False).unsqueeze(0)
        # self.base_theta = Variable(torch.eye(2,2)).unsqueeze(0)
        # self.base_transl = Variable(torch.zeros(2,1)).unsqueeze(0)

    def forward(self, inp):
        batch_size = inp.size(0)

        x = self.downsample(inp)    # inp: 480x480
        x = self.net1_conv1(x)
        x = self.net1_PReLU1(x)
        x = self.net1_pool(x)
        x = self.net1_conv2(x)
        x = self.net1_PReLU2(x)
        x = self.net1_pool(x)
        x = self.net1_conv3(x)
        x = self.net1_PReLU3(x)
        x = self.net1_pool(x)
        x = self.net1_conv4(x)
        x = self.net1_PReLU4(x)
        x = x.view(batch_size, -1)
        x = self.net1_fc5_1(x)
        x = self.net1_PReLU5(x)
        x = self.net1_drop6(x)
        x = self.net1_68point(x)
        landmarks = self.net1_PReLU6(x)    # landmarks: 136 (68x2 very inaccurate landmarks)
        theta = self.loc_reg_(landmarks)
        theta = theta.view(batch_size, 2, 3)
        # theta = theta.view(batch_size, 2, 2)

        # if theta.is_cuda:
        #     self.base_theta = self.base_theta.cuda()
        #     # self.base_transl = self.base_transl.cuda()
        # # base_transl = self.base_transl.repeat(batch_size,1,1)
        # theta += self.base_theta
        # # theta = torch.cat((theta, base_transl), dim=2)

        grid = F.affine_grid(theta, torch.Size([batch_size, 3, 256, 256]))   # Prepare the transfomer grid with (256, 256) size that FAN expects, w.r.t theta
        outp = F.grid_sample(inp, grid)                             # "Rotate" the image by applying the grid
        return outp, theta    # outp: 256x256, theta: 2x3


class STN_padded(nn.Module):

    def __init__(self):
        super(STN_padded, self).__init__()
        self.downsample = nn.AvgPool2d(8) # or stride=448/60
        self.net1_conv1 = nn.Conv2d(3,20,5)
        self.net1_conv2 = nn.Conv2d(20,48,5)
        self.net1_conv3 = nn.Conv2d(48,64,3)
        self.net1_conv4 = nn.Conv2d(64,80,3)
        self.net1_PReLU = nn.PReLU()
        self.net1_pool = nn.MaxPool2d(2)
        self.net1_fc5_1 = nn.Linear(80*3*3, 512)
        self.net1_drop6 = nn.Dropout2d(0.2)
        self.net1_68point = nn.Linear(512, 136)
        self.loc_reg_ = nn.Linear(136, 6)

        self.base_theta = Variable(torch.eye(2, 3)).unsqueeze(0)

        self.pad = 120

    def forward(self, inp):
        batch_size = inp.size(0)

        x = inp[:, :, self.pad:-self.pad, self.pad:-self.pad]    # inp: 720x720, x: 480x480. Only use the center portion for training (not the black triangles)
        x = self.downsample(x) # 3x60x60
        x = self.net1_conv1(x) # 20x56x56
        x = self.net1_PReLU(x) # 20x56x56
        x = self.net1_pool(x) # 20x28x28
        x = self.net1_conv2(x) # 48x24x24
        x = self.net1_PReLU(x) # 48x24x24
        x = self.net1_pool(x) # 48x12x12
        x = self.net1_conv3(x) # 64x10x10
        x = self.net1_PReLU(x)# 64x10x10
        x = self.net1_pool(x)# 64x5x5
        x = self.net1_conv4(x) # 80x3x3
        x = self.net1_PReLU(x) # 80x3x3
        x = x.view(x.size(0), -1) # 720
        x = self.net1_fc5_1(x) # 512
        x = self.net1_PReLU(x) # 512
        x = self.net1_drop6(x) # 512
        x = self.net1_68point(x) # 136
        landmarks = self.net1_PReLU(x)    # landmarks: 136 (68x2 very inaccurate landmarks)
        theta = self.loc_reg_(landmarks) # 6
        theta = theta.view(batch_size, 2, 3)

        if theta.is_cuda:
            self.base_theta = self.base_theta.cuda()
        theta += self.base_theta

        # For testing
        # theta = Variable(torch.Tensor([[1, 0, 0],[0, 1, 0]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # identity transform matrix
        # theta = Variable(torch.Tensor([[1.1, 0.5, 0.3], [0, 0.8, -0.1]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # stretching transform matrix

        grid = F.affine_grid(theta, torch.Size([batch_size, 3, 256, 256]))   # Prepare the transfomer grid with (256, 256) size that FAN expects, w.r.t theta
        outp = F.grid_sample(inp, grid)    # "Rotate" the image by applying the grid
        return outp, theta    # outp: 256x256, theta: 2x3


class STN_small(nn.Module):

    def __init__(self):
        super(STN_small, self).__init__()
        self.downsample = nn.AvgPool2d(8)
        self.net1_conv1 = nn.Conv2d(3,15,7)
        self.net1_conv2 = nn.Conv2d(15,30,5)
        self.net1_PReLU = nn.PReLU()
        self.net1_pool = nn.MaxPool2d(2)
        self.net1_drop = nn.Dropout2d(0.2)
        self.net1_fc5_1 = nn.Linear(480, 120)
        self.loc_reg_ = nn.Linear(120, 6)

        self.base_theta = Variable(torch.eye(2, 3)).unsqueeze(0)

        self.pad = 60

    def forward(self, inp):
        batch_size = inp.size(0)

        x = inp[:, :, self.pad:-self.pad, self.pad:-self.pad]    # inp: 360x360, x: 240x240. Only use the center portion for training (not the black triangles)
        x = self.downsample(x) # 3x30x30
        x = self.net1_conv1(x) # 15x24x24
        x = self.net1_PReLU(x) # 15x24x24
        x = self.net1_pool(x) # 15x12x12
        x = self.net1_conv2(x) # 30x8x8
        x = self.net1_PReLU(x) # 30x8x8
        x = self.net1_pool(x) # 30x4x4
        x = x.view(x.size(0), -1) # 480
        x = self.net1_fc5_1(x) # 120
        x = self.net1_PReLU(x) # 120
        x = self.net1_drop(x) # 120
        theta = self.loc_reg_(x) # 6
        theta = theta.view(batch_size, 2, 3)

        if theta.is_cuda:
            self.base_theta = self.base_theta.cuda()
        theta += self.base_theta

        # For testing
        # theta = Variable(torch.Tensor([[1, 0, 0],[0, 1, 0]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # identity transform matrix
        # theta = Variable(torch.Tensor([[1.1, 0.5, 0.3], [0, 0.8, -0.1]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # stretching transform matrix

        grid = F.affine_grid(theta, torch.Size([batch_size, 3, 256, 256]))   # Prepare the transfomer grid with (256, 256) size that FAN expects, w.r.t theta
        outp = F.grid_sample(inp, grid)    # "Rotate" the image by applying the grid
        return outp, theta    # outp: 256x256, theta: 2x3


class STN_one(nn.Module):

    def __init__(self):
        super(STN_one, self).__init__()
        self.downsample = nn.AvgPool2d(8)
        self.net1_conv1 = nn.Conv2d(3,15,7)
        self.net1_conv2 = nn.Conv2d(15,30,5)
        self.net1_conv3 = nn.Conv2d(30,60,3)
        self.net1_PReLU = nn.PReLU()
        self.net1_pool = nn.MaxPool2d(2)
        self.net1_drop6 = nn.Dropout2d(0.2)
        self.net1_fc5_1 = nn.Linear(240, 12)
        self.loc_reg_ = nn.Linear(12, 1)

        self.pad = 60

    def forward(self, inp):
        batch_size = inp.size(0)

        x = inp[:, :, self.pad:-self.pad, self.pad:-self.pad]    # inp: 360x360, x: 240x240. Only use the center portion for training (not the black triangles)
        x = self.downsample(x) # 3x30x30
        x = self.net1_conv1(x) # 15x24x24
        x = self.net1_PReLU(x) # 15x24x24
        x = self.net1_pool(x) # 15x12x12
        x = self.net1_conv2(x) # 30x8x8
        x = self.net1_PReLU(x) # 30x8x8
        x = self.net1_pool(x) # 30x4x4
        x = self.net1_conv3(x) # 60x2x2
        x = self.net1_PReLU(x) # 60x2x2
        x = x.view(x.size(0), -1) # 240
        x = self.net1_fc5_1(x) # 12
        x = self.net1_PReLU(x) # 12
        x = self.net1_drop6(x) # 12
        angle = self.loc_reg_(x) # 1

        c, s = torch.cos(angle), torch.sin(angle)
        theta = torch.cat((torch.cat((c, -s), 1), torch.cat((s, c), 1)), 1).view(batch_size, 2, 2)
        theta = torch.cat((theta, Variable(torch.zeros(2, 1).cuda(), requires_grad=False).unsqueeze(0).repeat(batch_size, 1, 1)), 2)
        # For testing
        # theta = Variable(torch.Tensor([[1, 0, 0],[0, 1, 0]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # identity transform matrix
        # theta = Variable(torch.Tensor([[1.1, 0.5, 0.3], [0, 0.8, -0.1]]).cuda().view(1, 2, 3).repeat(batch_size, 1, 1), requires_grad=True)    # stretching transform matrix

        grid = F.affine_grid(theta, torch.Size([batch_size, 3, 256, 256]))   # Prepare the transfomer grid with (256, 256) size that FAN expects, w.r.t theta
        outp = F.grid_sample(inp, grid)    # "Rotate" the image by applying the grid
        return outp, theta    # outp: 256x256, theta: 2x3


class STN_HG(nn.Module):

    def __init__(self):
        super(STN_HG, self).__init__()
        self.base_theta = Variable(torch.eye(2, 3)).unsqueeze(0)

        hg_module = 1
        self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
        self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        self.add_module('conv_last' + str(hg_module),
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

        self.loc_reg_ = nn.Linear(136, 6)

    def forward(self, inp):
        batch_size = inp.size(0)

        i = 1
        hg = self._modules['m' + str(i)](inp)
        ll = hg
        ll = self._modules['top_m_' + str(i)](ll)
        ll = F.relu(self._modules['bn_end' + str(i)]
                    (self._modules['conv_last' + str(i)](ll)), True)
        landmarks = ll
        theta = self.loc_reg_(ll)

        grid = F.affine_grid(theta, torch.Size([batch_size, 3, 256, 256]))   # Prepare the transfomer grid with (256, 256) size that FAN expects, w.r.t theta
        outp = F.grid_sample(inp, grid)                             # "Rotate" the image by applying the grid
        return outp, landmarks, theta


class STEFAN(nn.Module):

    def __init__(self, num_modules=1, num_landmarks=68):
        super(STEFAN, self).__init__()
        self.num_modules = num_modules
        self.num_landmarks = num_landmarks
        self.fan = FAN(num_modules, num_landmarks)
        self.stn = STN()

    def forward(self, inp):
        frontal_img, theta = self.stn(inp)
        hm_rot = self.fan(frontal_img)

        theta_inv = Variable(torch.eye(3))
        if theta.is_cuda:
            theta_inv = theta_inv.cuda()
        theta_inv[0:2] = theta[0]
        theta_inv = torch.inverse(theta_inv)[0:2].repeat(theta.data.shape[0], 1, 1)

        grid = nn.functional.affine_grid(theta_inv, torch.Size([theta.data.shape[0], 68, 64, 64]))
        hm = []
        for i in range(self.num_modules):
            hm.append(nn.functional.grid_sample(hm_rot[i], grid))

        return hm, frontal_img, hm_rot
