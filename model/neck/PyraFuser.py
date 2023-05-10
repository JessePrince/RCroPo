import torch 
from torch.nn import functional as F
from torch import nn

def interpolate(tensor, scale_factor=2, mode='nearest'):
    return F.interpolate(tensor, scale_factor=scale_factor, mode=mode)

class PyramidFuser(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels):
        super(PyramidFuser, self).__init__()
        self.n_stages = n_stages
        self.channels = in_channels
        self.out_channels = out_channels
        self.pwconv = nn.ModuleList()
        for i in range(self.n_stages - 1): 
            m = nn.Conv3d(in_channels = self.channels[i],
                          out_channels = self.out_channels,
                          kernel_size = 1)
            self.pwconv.append(m)
        
    def forward(self, pyramid):
        catFeature = pyramid[0]
        # # channel reduction------------------------------
        # for i in range(self.n_stages - 1):
        #     pyramid[i] = self.pwconv[i](pyramid[i])
                                
        # #------------------------------------------------
        for i in range(self.n_stages - 1): # skip the first one
            if not pyramid[i + 1].shape[-1] == catFeature.shape[-1]:
                feature = interpolate(pyramid[i + 1], scale_factor=(catFeature.shape[-1]/pyramid[i + 1].shape[-1]))
            else:
                feature = pyramid[i + 1]
            
        catFeature = torch.cat((feature, catFeature), dim=1)
            
        return catFeature
    
pyramid = [torch.randn(1, 192, 2, 64, 64), 
           torch.randn(1, 384, 1, 32, 32)]

fuser = PyramidFuser(n_stages = 2, in_channels=[192, 384, 768], out_channels=128)
out = fuser(pyramid)

print(out.shape)