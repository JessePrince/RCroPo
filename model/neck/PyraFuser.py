import torch 
from torch.nn import functional as F

def interpolate(tensor, scale_factor=2, mode='nearest'):
    return F.interpolate(tensor, scale_factor=scale_factor, mode=mode)

def PyramidFuser(pyramid):
    n_stages = len(pyramid)
    n_features = len(pyramid[0])
    catFeature = pyramid[0][0]
    catFeature = torch.cat((catFeature, pyramid[0][1]), dim=1)
    for i in range(n_stages - 1): # skip the first one
        for j in range(n_features):
            if not pyramid[i + 1][j].shape[-1] == catFeature.shape[-1]:
                feature = interpolate(pyramid[i + 1][j], scale_factor=(catFeature.shape[-1]/pyramid[i + 1][j].shape[-1]))
            else:
                feature = pyramid[i + 1][j]
            
            catFeature = torch.cat((feature, catFeature), dim=1)
            
    return catFeature
