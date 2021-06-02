import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()

        self.ConvLayer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8))

        self.FlattenLayer = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 5))

    def forward_once(self, x):
        x = self.ConvLayer(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.FlattenLayer(x)
        return x

    def forward(self, inp1, inp2):
        out1 = self.forward_once(inp1)
        out2 = self.forward_once(inp2)
        return out1, out2
