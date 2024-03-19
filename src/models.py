# code adapted and corrected from : https://github.com/wkentaro/pytorch-fcn/tree/main

import numpy as np
import torch 
import torch.nn as nn

def get_bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    As suggested by http://arxiv.org/abs/1411.4038, the upsampling layer weights should be a bilinear interpolation
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    coord = np.ogrid[:kernel_size, :kernel_size]

    weight= (1 - np.abs(coord[0] - center)/factor) * (1 - np.abs(coord[1] - center)/factor)
    weight = torch.from_numpy(weight)
    
    return weight.expand(out_channels, in_channels, kernel_size, kernel_size).contiguous()

class LayerFCN8(nn.Module):
    def __init__(self, n_class=7, initialize_from_scratch: bool = False,
                 path_to_pretrained_folder: str = None):
        super().__init__()

        self.pretrained_weight_file = 'fcn8s-heavy-pascal.pth'

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        if initialize_from_scratch:
            for m in self.modules():
                self.initialize_weights(m)
        
        if path_to_pretrained_folder:
            weight_dict = torch.load(f'{path_to_pretrained_folder}/{self.pretrained_weight_file}')
            param_dict = self.state_dict()

            for k in weight_dict.keys():
                if 'score' in k:
                    layer = k.split('.')[0]
                    self.initialize_weights(getattr(self, layer))
                else:
                    param_dict[k].copy_(weight_dict[k])
        
    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.ConvTranspose2d):
            assert module.kernel_size[0] == module.kernel_size[1]
            bilinear_kernel = get_bilinear_kernel(module.in_channels, module.out_channels, module.kernel_size[0])
            module.weight.data.copy_(bilinear_kernel)

    def forward(self, x):
        _, _, H, W = x.size()

        h = self.relu1_1(self.conv1_1(x))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h 

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  

        h = upscore2 + score_pool4c  
        h = self.upscore_pool4(h)
        upscore_pool4 = h  

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  

        h = upscore_pool4 + score_pool3c  

        h = self.upscore8(h)
        h = h[:, :, 31:31 + H, 31:31 + W].contiguous()

        return h   

class FCN8s(nn.Module):
    def __init__(self, n_class=21, initialize: bool = False):

        super().__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        
        if initialize:
            self.initialize_weights()

        self.pretrained_weight_file = 'fcn8s-heavy-pascal.pth'

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                bilinear_kernel = get_bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(bilinear_kernel)

    def forward(self, x):
        _, _, H, W = x.size()

        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h 

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  

        h = upscore2 + score_pool4c  
        h = self.upscore_pool4(h)
        upscore_pool4 = h  

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  

        h = upscore_pool4 + score_pool3c  

        h = self.upscore8(h)
        h = h[:, :, 31:31 + H, 31:31 + W].contiguous()

        return h
            
class OwnFCN8s(nn.Module):
    def __init__(self, n_class=21, initialize: bool = False):

        super().__init__()

        # conv1
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=100),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, padding=1), 
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2, ceil_mode=True))
        
        # conv2
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2, ceil_mode=True))
        # conv3
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2, ceil_mode=True))
        # conv4
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2, ceil_mode=True))
        # conv5
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2, ceil_mode=True))
        # fc6
        self.fc6 = nn.Sequential(nn.Conv2d(512, 4096, 7),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d())
        # fc7
        self.fc7 = nn.Sequential(nn.Conv2d(4096, 4096, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d())
        
        # coarse_score_fc7 (for fc7)
        self.coarse_score_fc7 = nn.Conv2d(4096, n_class, 1)
        # coarse_score3 (for conv3)
        self.coarse_score3 = nn.Conv2d(256, n_class, 1)
        # coarse_score4 (for conv4)
        self.coarse_score4 = nn.Conv2d(512, n_class, 1)

        # upsample_fc7
        self.upsample_fc7 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        # upsample_last (stride 8)
        self.upsample_last = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        # upsample4 (after first skip connection with coarse_score4)
        self.upsample4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)

        if initialize:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                bilinear_kernel = get_bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(bilinear_kernel)

    def forward(self, x):
        _, _, H, W = x.size()
        h = self.conv1(x)
        h = self.conv2(h)

        h = self.conv3(h)
        score_coarse3 = self.coarse_score3(h)
        
        h = self.conv4(h)
        score_coarse4 = self.coarse_score4(h)

        h = self.conv5(h)

        h = self.fc6(h)
        h = self.fc7(h)
        h = self.coarse_score_fc7(h)
        h = self.upsample_fc7(h)
        upscore1 = h

        h = score_coarse4[:, :, 5:5+upscore1.size()[2], 5:5+upscore1.size()[3]] + upscore1

        upscore2 = self.upsample4(h)

        h = score_coarse3[:, :, 9:9+upscore2.size()[2], 9:9+upscore2.size()[3]] + upscore2

        upscore_last = self.upsample_last(h)

        return upscore_last[:, :, 31:31+H, 31:31+W].contiguous()  


# mod = FCN8s()

    
# pretrained_weight_path = 'fcn8s-heavy-pascal.pth'

# # dict_pw = torch.load(pretrained_weight_path)
# # print(dict_pw.keys())
# # print(dict_pw["conv3_2.weight"].shape)

# mod.load_state_dict(torch.load(pretrained_weight_path))
# mod.eval()
# x = torch.rand(2, 3, 512, 512)
# y = mod(x)
# print(y.size())


# # # print(mod)
# # for m in mod.modules():
# #     print(m)

# # for i, p in enumerate(mod.parameters()):
# #     print(p)

# # print('total', i)