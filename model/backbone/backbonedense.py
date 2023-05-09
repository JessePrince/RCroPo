from torch import nn
from model.backbone.convblk import ConvNextBLK3D

class Backbone3D(nn.Module):
    def __init__(self, in_channels, depths=[3, 3, 9, 3], width=[96, 192, 384, 768]):
        super().__init__()
        self.n_stages = len(depths)
        self.depths = depths
        # Branch 1--------------------------------------------------------------------------
        self.stem1 = nn.Sequential(nn.Conv3d(in_channels = in_channels,
                                           out_channels = width[0],
                                           kernel_size = 4,
                                           stride = 4),
                                  nn.BatchNorm3d(num_features=width[0], eps=1e-6)) # 创建patch层，分patch
        self.downSampler1 = nn.ModuleList()
        self.downSampler1.append(self.stem1) # 装入下采样器
        for i in range(self.n_stages - 1):
            sampler = nn.Sequential(nn.BatchNorm3d(num_features=width[i], eps=1e-6),
                                    nn.Conv3d(width[i], width[i+1], kernel_size=2, stride=2)) # 2 times down sample
            self.downSampler1.append(sampler) # 再装入其余下采样器
            
        self.extractor1 = nn.ModuleList() # shape invariant
        
        for n_stage in range(self.n_stages):
            stage = nn.ModuleList()
            for i in range(depths[n_stage]):
                stage.append(ConvNextBLK3D(dim = width[n_stage]))
            
            self.extractor1.append(stage) # append深度设定的不变形卷积块
            
        self.norm1 = nn.BatchNorm3d(num_features=width[-1], eps=1e-6)
        #-----------------------------------------------------------------------------------------
        
        # Branch 2---------------------------------------------------------------------------------
        self.stem2 = nn.Sequential(nn.Conv3d(in_channels = in_channels,
                                           out_channels = width[0],
                                           kernel_size = 4,
                                           stride = 4),
                                  nn.BatchNorm3d(num_features=width[0], eps=1e-6)) # 创建patch层，分patch
        self.downSampler2 = nn.ModuleList()
        self.downSampler2.append(self.stem1) # 装入下采样器
        for i in range(self.n_stages - 1):
            sampler = nn.Sequential(nn.BatchNorm3d(num_features=width[i], eps=1e-6),
                                    nn.Conv3d(width[i], width[i+1], kernel_size=2, stride=2)) # 三维2x2卷积下采样
            self.downSampler2.append(sampler) # 再装入其余下采样器
            
        self.extractor2 = nn.ModuleList() # 不变形卷积层
        for n_stage in range(self.n_stages):
            stage = nn.ModuleList()
            for i in range(depths[n_stage]):
                stage.append(ConvNextBLK3D(dim = width[n_stage]))
            
            self.extractor2.append(stage) # append深度设定的不变形卷积块
            
        self.norm2 = nn.BatchNorm3d(num_features=width[-1], eps=1e-6)
        # --------------------------------------------------------------------------------------------
        
        
    def forward(self, hori, vert):
        in1 = hori
        in2 = vert
        pyramid = []
        for i in range(self.n_stages):
            in1 = self.downSampler1[i](in1) # update in1 with down sampler
            in2 = self.downSampler2[i](in2)
            if self.n_stages > 0: # store the feature map after down sample
                pyramid.append([in1, in2])
                
            f11 = in1 # hold in1, use f11 as forward feature
            f22 = in2
            for j in range(self.depths[i]):
                f1 = self.extractor1[i][j](f11) # f1 is the output of convnext block
                f2 = self.extractor2[i][j](f22)
                fuse = f1 + f2 # fuse upper and lower branch feature
                f11 = fuse # update f11 with fuse
                f22 = fuse
                
            in1 = f1 # update in1 to f1
            in2 = f2
                
        out1 = self.norm1(in1)
        out2 = self.norm2(in2)
        pyramid.append([out1, out2])
        
        return pyramid