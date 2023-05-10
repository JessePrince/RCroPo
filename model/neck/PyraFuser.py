import torch 
from torch.nn import functional as F
from torch import nn

def interpolate(tensor, scale_factor=2, mode='nearest'):
    return F.interpolate(tensor, scale_factor=scale_factor, mode=mode)

class PyramidFuser(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, depth):
        super(PyramidFuser, self).__init__()
        self.n_stages = n_stages
        self.pwconv = nn.ModuleList()
        self.catChannel = 0
        for i in in_channels:
            self.catChannel += i
            
        self.dwconv = nn.Conv3d(in_channels = self.catChannel,
                                out_channels = self.catChannel * depth,
                                kernel_size = (depth, 1, 1))
        self.pwconv = nn.Conv2d(in_channels = self.catChannel * depth,
                                out_channels = out_channels,
                                kernel_size = 1)
        
    def forward(self, pyramid):
        catFeature = pyramid[0]
        for i in range(self.n_stages - 1): # skip the first one
            if not pyramid[i + 1].shape[-1] == catFeature.shape[-1]:
                feature = interpolate(pyramid[i + 1], scale_factor=(catFeature.shape[-1]/pyramid[i + 1].shape[-1]))
            else:
                feature = pyramid[i + 1]
            
        catFeature = torch.cat((feature, catFeature), dim=1)
        # depth reduction & channel reduction -------------
        catFeature = self.dwconv(catFeature)
        catFeature = torch.squeeze(catFeature, dim=2)
        catFeature = self.pwconv(catFeature)
        #------------------------------------------------
        return catFeature
    
pyramid = [torch.randn(1, 192, 2, 64, 64), 
           torch.randn(1, 384, 1, 32, 32)]

fuser = PyramidFuser(n_stages = 2, in_channels=[192, 384], out_channels=128, depth=2)
out = fuser(pyramid)

print(out.shape)