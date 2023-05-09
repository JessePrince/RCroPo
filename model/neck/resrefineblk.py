from torch import nn

class ResRefineBLK(nn.Module):
    def __init__(self, channels):
        super(ResRefineBLK, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels = channels,
                               out_channels = (channels / 4),
                               kernel_size = 1,
                               stride = 1)
        self.Conv2 = nn.Conv2d(in_channels = (channels / 4),
                               out_channels = (channels / 4),
                               kernel_size = 3,
                               padding = 1,
                               stride = 1)
        self.Conv3 = nn.Conv2d(in_channels = (channels / 4),
                               out_channels = channels,
                               kernel_size = 1,
                               stride = 1)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        y = self.Conv1(x)
        y = self.Conv2(y)
        y = self.Conv3(y)
        y = self.ReLU(y)
        
        return x + y