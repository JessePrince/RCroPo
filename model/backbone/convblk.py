from torch import nn
from timm.models.layers import DropPath

class ConvNextBLK3D(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm3d(num_features=dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1) # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.dropPath = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def ResConnect(self, x, y):
        return x + y 
    
    def forward(self, x):
        _input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)

        x = self.ResConnect(_input, self.dropPath(x))
        return x