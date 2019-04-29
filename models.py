import torch.nn as nn


class ConvolutionalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class CNN(nn.Module):
    '''
    Expects square images with even number of pixels for H and W (eg 7x7 does not work)
    H and W must each satisfy: (H / 2^{num_layers}) % 2 == 0
    '''

    def __init__(self, image_size, cnn_layers, cnn_kernel_size, fc_hidden_feats, fc_out_feats):
        '''
        cnn_layers: (list of ints) list of number of out channels of each cnn layer
        '''
        super().__init__()
        if cnn_kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd number. We found: {cnn_kernel_size}")
        cnn_num_layers = len(cnn_layers)
        feature_maps = [3] + cnn_layers
        layers = [ConvolutionalLayer(feature_maps[i], feature_maps[i + 1], cnn_kernel_size)
                  for i in range(cnn_num_layers)]

        self.cnn = nn.Sequential(*layers)
        fc_in_feats = int(image_size / 2 ** cnn_num_layers) ** 2 * cnn_layers[-1]
        self.fc1 = nn.Linear(fc_in_feats, fc_hidden_feats)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_feats, fc_out_feats)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        print(x.size())
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == '__main__':
    import torch
    batch_size = 32
    in_channels = 3
    input_size = 64
    x = torch.randn(batch_size, in_channels, input_size, input_size)

    layers = [200, 400, 200]
    kernel_size = 5
    fc_hidden_feats = 3000
    fc_out_feats = 964
    model = CNN(input_size, layers, kernel_size, fc_hidden_feats, fc_out_feats)
    out = model(x)
    assert(out.size() == (batch_size, 964))
