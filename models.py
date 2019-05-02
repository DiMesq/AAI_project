import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class ConvolutionalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, dropout=0.):
        super().__init__()
        padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(2, stride=2) if max_pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=True, dropout=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, self.hidden_size, num_layers=1, batch_first=True,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_lens):
        '''
        x: tensor (B, L, H)
        x_lens: tensor (B, 1) - real lengths of each x in the batch
        '''
        B, L, H = x.size()
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True)
        _, h_rnn = self.gru(x_packed)
        h_rnn = h_rnn.transpose(0, 1).contiguous().view(B, -1)
        return h_rnn


class CNN(nn.Module):
    '''
    Expects square images with even number of pixels for H and W (eg 7x7 does not work)
    H and W must each satisfy: (H / 2^{num_layers}) % 2 == 0
    '''

    def __init__(self, image_size, layers, kernel_size, num_classes=None,
                 classifier=True, global_pool_hidden_size=500, cnn_dropout=0.,
                 fc_dropout=0.):
        '''
        layers: (list of ints) list of number of out channels of each cnn layer
        '''
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd number. We found: {kernel_size}")
        fc_hidden_size = 3000
        num_layers = len(layers)
        feature_maps = [1] + layers
        layers = [ConvolutionalLayer(feature_maps[i], feature_maps[i + 1], kernel_size, dropout=cnn_dropout)
                  for i in range(num_layers)]


        self.classifier = classifier
        last_feature_map_size = int(image_size / 2 ** num_layers)
        if self.classifier: # todo
            fc_in_feats = last_feature_map_size ** 2 * feature_maps[-1]
            self.dropout = nn.Dropout(fc_dropout)
            self.fc1 = nn.Linear(fc_in_feats, fc_hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        else:
            layers.append(ConvolutionalLayer(feature_maps[-1],
                                             global_pool_hidden_size,
                                             kernel_size, max_pool=False))
            self.global_pool = nn.AvgPool2d(kernel_size=last_feature_map_size)

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        # cnn
        h_cnn = self.cnn(x)

        if self.classifier:
            # fc
            h_fc1 = h_cnn.view(batch_size, -1)
            h_fc1 = self.dropout(self.relu(self.fc1(h_fc1)))
            h_fc2 = self.fc2(h_fc1)
            return h_fc2, h_fc1, h_cnn
        else:
            h_cnn2 = self.global_pool(h_cnn).squeeze()
            return None, h_cnn2, h_cnn


class CNN_RNN(nn.Module):

    def __init__(self, num_classes, cnn_input_size, cnn_layers, cnn_kernel_size,
                 cnn_hidden_size, cnn_dropout, rnn_input_size, rnn_hidden_size,
                 rnn_dropout, fc_dropout):
        super().__init__()
        self.cnn = CNN(cnn_input_size, cnn_layers, cnn_kernel_size,
                       classifier=False, global_pool_hidden_size=cnn_hidden_size,
                       cnn_dropout=cnn_dropout)
        self.rnn = RNN(rnn_input_size, rnn_hidden_size, dropout=rnn_dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        # RNN is bidirectional
        fc_hidden_size = 3000
        self.fc1 = nn.Linear(cnn_hidden_size + 2 * rnn_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, images, strokes, num_strokes, strokes_lens):
        '''
        images: (B, 1, H, W)
        strokes: (B, S, L, 2)
        num_strokes: (B, 1)
        strokes_lens: (B, S)

        where B is size of batch, S is number of strokes, L is length of a stroke
        and 2 is the (x,y) coordinate of point from a stroke
        '''
        _, h_image, _ = self.cnn(images)
        B, S, L, _ = strokes.size()
        h_strokes = []
        for stroke_ix in range(S):
            stroke = strokes[:, stroke_ix, :, :]
            h_stroke = self.rnn(stroke, strokes_lens[:, stroke_ix])
            # if element in batch doesn't have this stroke then set h to 0
            h_stroke[stroke_ix >= num_strokes, :] = 0
            h_strokes.append(h_stroke)
        # get (B, S, H) where H is self.hidden_size
        h_strokes = torch.stack(h_strokes, dim=1)
        # avg: get (B, H)
        h_strokes = h_strokes.sum(dim=1) / num_strokes.unsqueeze(1)
        h_final = torch.cat((h_image, h_strokes), dim=1)

        h_fc1 = self.dropout(self.relu(self.fc1(h_final)))
        h_fc2 = self.dropout(self.relu(self.fc2(h_fc1)))
        h_fc3 = self.fc3(h_fc2)
        return h_fc3, h_fc2, h_final


if __name__ == '__main__':
    # test CNN
    batch_size = 32
    in_channels = 1
    input_size = 64
    x = torch.randn(batch_size, in_channels, input_size, input_size)

    layers = [200, 400, 200]
    kernel_size = 5
    fc_hidden_size = 3000
    num_classes = 964
    model = CNN(input_size, layers, kernel_size, num_classes=964)
    out, h_fc, h_cnn = model(x)
    print(h_cnn.size())
    assert(h_cnn.size() == (batch_size, layers[-1], 8, 8))
    assert(h_fc.size() == (batch_size, 3000))
    assert(out.size() == (batch_size, 964))

    # test RNN
    B, L, I, H = 3, 70, 2, 5
    x = torch.randn(B, L, I)
    x[1, :, :] = 0
    x_lens = torch.LongTensor([70, 65, 63])
    model = RNN(I, H)
    out = model(x, x_lens)
    assert(out.size() == (B, 2*H))

    # test CNN+RNN
    num_classes = 964
    B, S, L, I, H_RNN, H_CNN = 2, 3, 70, 4, 5, 10
    images = torch.randn(B, 1, 28, 28)
    strokes = torch.randn(B, S, L, I)
    strokes[1, 2, :] = 0
    num_strokes = torch.FloatTensor([3, 2])
    stroke_lens = torch.FloatTensor([[70, 67, 64], [65, 61, 58]])
    model = CNN_RNN(num_classes, cnn_input_size=28, cnn_layers=[120, 300], cnn_kernel_size=5,
                    cnn_hidden_size=10, rnn_input_size=I, rnn_hidden_size=H_RNN,
                    cnn_dropout=0, rnn_dropout=0, fc_dropout=0.5)
    print("CNN+RNN model:")
    print(model)
    h_fc3, h_fc2, h_final = model(images, strokes, num_strokes, stroke_lens)
    assert(h_final.size() == (B, H_CNN + 2*H_RNN))
    assert(h_fc2.size() == (B, 3000))
    assert(h_fc3.size() == (B, num_classes))







































