import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.normal_(param.data)

def modify_CNN_RNN_for_finetune(model):
    model.rnn = Identity()
    return model

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvolutionalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, dropout=0.):
        super().__init__()
        padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv.apply(weight_init)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
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

    def __init__(self, input_size, hidden_size, num_layers,
                 num_directions, dropout=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        bidirectional = True if self.num_directions > 1 else False
        self.gru = nn.GRU(input_size, self.hidden_size,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=dropout, bidirectional=bidirectional)
        self.gru.apply(weight_init)

    def forward(self, x, x_lens):
        '''
        x: tensor (B, T, H)
        x_lens: tensor (B, 1) - real lengths of each x in the batch
        '''
        B, T, H = x.size()
        assert(x_lens.size() == (B, 1))
        need_sort = not self.__is_sorted(x_lens)
        if need_sort:
            sort_ixs, reverse_sort_ixs = self.__get_sort_ixs(x_lens.squeeze(dim=1))
            x, x_lens = x[sort_ixs], x_lens[sort_ixs]
            assert(x.size() == (B, T, H))
        x_packed = pack_padded_sequence(x, x_lens.squeeze(dim=1), batch_first=True)
        _, h_rnn = self.gru(x_packed)
        h_rnn = h_rnn.view(self.num_layers, self.num_directions, B, self.hidden_size)
        h_rnn = h_rnn.sum(dim=0) # (D, B, H)
        h_rnn = h_rnn.transpose(0, 1).contiguous().view(B, -1) # (B, D * H)
        if need_sort:
            h_rnn = h_rnn[reverse_sort_ixs]
        return h_rnn

    def __is_sorted(self, x):
        '''
        x: (B,)
        '''
        v_prev = x[0]
        for i in range(1, x.size(0)):
            v = x[i]
            if v > v_prev:
                return False
            v_prev = v
        return True

    def __get_sort_ixs(self, x):
        sort_ixs = torch.argsort(x, dim=0, descending=True)
        reverse_sort_ixs = torch.argsort(sort_ixs, dim=0)
        return sort_ixs, reverse_sort_ixs


class CNN(nn.Module):
    '''
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
        if self.classifier:
            self.cnn = nn.Sequential(*layers)
            fc_in_feats = last_feature_map_size ** 2 * feature_maps[-1]
            self.fc1 = nn.Linear(fc_in_feats, fc_hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(fc_dropout)
            self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        else:
            layers.append(ConvolutionalLayer(feature_maps[-1],
                                             global_pool_hidden_size,
                                             kernel_size, max_pool=False))
            self.cnn = nn.Sequential(*layers)
            self.global_pool = nn.AvgPool2d(kernel_size=last_feature_map_size)

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
                 cnn_hidden_size, cnn_dropout, rnn_input_size, rnn_num_layers,
                 rnn_hidden_size, rnn_num_directions, rnn_dropout, pool_kind,
                 fc_num_layers, fc_dropout):
        super().__init__()
        self.pool_kind = pool_kind
        self.rnn_num_directions = rnn_num_directions
        if self.pool_kind == 'add':
            hidden_size = min([cnn_hidden_size, rnn_hidden_size])
            # have to multiply hidden_size by 2 because RNN is bidirectional
            self.cnn_hidden_size, self.rnn_hidden_size = self.rnn_num_directions * hidden_size, hidden_size
        else:
            self.cnn_hidden_size, self.rnn_hidden_size = cnn_hidden_size, rnn_hidden_size

        self.cnn = CNN(cnn_input_size, cnn_layers, cnn_kernel_size,
                       classifier=False,
                       global_pool_hidden_size=self.cnn_hidden_size,
                       cnn_dropout=cnn_dropout)
        self.rnn = RNN(rnn_input_size, self.rnn_hidden_size,
                       rnn_num_layers, rnn_num_directions, dropout=rnn_dropout)
        # RNN is bidirectional
        fc_in_size = self.cnn_hidden_size if pool_kind == 'add'\
            else self.cnn_hidden_size + self.rnn_num_directions * self.rnn_hidden_size
        fc_hidden_size = 1500 if fc_num_layers > 2 else 3000

        self.fc1 = nn.Sequential(
            nn.Linear(fc_in_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout))

        middle_layers = []
        for layer in range(fc_num_layers - 1):
            middle_layers.extend([
                nn.Linear(fc_hidden_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
            ])
        self.fc2 = nn.Sequential(*middle_layers)
        self.fc3 = nn.Linear(fc_hidden_size, num_classes)

        # self.dropout = nn.Dropout(fc_dropout)
        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(fc_in_size, fc_hidden_size)
        # self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        # self.fc3 = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, images, *strokes_data):
        '''
        images: (B, 1, H, W)
        strokes: (B, S, T, 2)
        num_strokes: (B, 1)
        strokes_lens: (B, S, 1)

        where B is size of batch, S is number of strokes, L is length of a stroke
        and 2 is the (x,y) coordinate of point from a stroke
        '''
        B = images.size(0)
        if strokes_data and len(strokes_data) == 3:
            strokes, num_strokes, strokes_lens = strokes_data
            _, S, T, _ = strokes.size()
            assert(num_strokes.size() == (B, 1))
            assert(strokes_lens.size() == (B, S, 1))

        _, h_image, _ = self.cnn(images)

        if strokes_data:
            h_strokes = []
            for stroke_ix in range(S):
                # if element in batch doesn't have this stroke then set its
                # stroke_len to 1. so RNN doesn't throw error. However, must set
                # the output of RNN to a tensor of zeros later
                strokes_lens[stroke_ix >= num_strokes.squeeze(dim=1), stroke_ix] = 1.
                stroke = strokes[:, stroke_ix, :, :]
                h_stroke = self.rnn(stroke, strokes_lens[:, stroke_ix])
                # if element in batch doesn't have this stroke then set h to 0's (see above)
                h_stroke[stroke_ix >= num_strokes.squeeze(dim=1), :] = 0
                h_strokes.append(h_stroke)
            # get (B, S, H) where H is self.hidden_size
            h_strokes = torch.stack(h_strokes, dim=1)
            # avg: get (B, H)
            h_strokes = h_strokes.sum(dim=1) / num_strokes
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            h_strokes = torch.zeros(B, self.rnn_num_directions * self.rnn_hidden_size).to(device)

        if self.pool_kind == 'add':
            h_final = h_image + h_strokes
        else:
            h_final = torch.cat((h_image, h_strokes), dim=1)

        h_fc1 = self.fc1(h_final)
        h_fc2 = self.fc2(h_fc1)
        h_fc3 = self.fc3(h_fc2)

        # h_fc1 = self.dropout(self.relu(self.fc1(h_final)))
        # h_fc2 = self.dropout(self.relu(self.fc2(h_fc1)))
        # h_fc3 = self.fc3(h_fc2)

        return h_fc3, h_fc2, h_final


if __name__ == '__main__':
    print('test CNN (even size: 64x64)')
    start = time.time()
    batch_size = 32
    in_channels = 1
    input_size = 64
    x = torch.randn(batch_size, in_channels, input_size, input_size)

    layers = [20, 40, 50]
    kernel_size = 5
    fc_hidden_size = 3000
    num_classes = 964
    model = CNN(input_size, layers, kernel_size, num_classes=num_classes)
    out, h_fc, h_cnn = model(x)
    assert(h_cnn.size() == (batch_size, layers[-1], input_size / 2 ** (len(layers)), input_size / 2 ** (len(layers))))
    assert(h_fc.size() == (batch_size, fc_hidden_size))
    assert(out.size() == (batch_size, num_classes))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')

    # test CNN (odd size: 105x105)
    print('test CNN (odd size: 105x105)')
    start = time.time()
    batch_size = 32
    in_channels = 1
    input_size = 105
    x = torch.randn(batch_size, in_channels, input_size, input_size)

    layers = [20, 50]
    kernel_size = 5
    fc_hidden_size = 3000
    num_classes = 20
    model = CNN(input_size, layers, kernel_size, num_classes=num_classes)
    out, h_fc, h_cnn = model(x)
    assert(h_cnn.size() == (batch_size, layers[-1], 26, 26))
    assert(h_fc.size() == (batch_size, fc_hidden_size))
    assert(out.size() == (batch_size, num_classes))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')

    print('test RNN')
    start = time.time()
    B, T, I, H, L, D = 3, 70, 2, 5, 3, 2
    x = torch.randn(B, T, I)
    x[1, :, :] = 0
    x_lens = torch.LongTensor([63, 1, 70]).unsqueeze(dim=1)
    model = RNN(I, H, L, D)
    out = model(x, x_lens)
    assert(out.size() == (B, D*H))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')

    print('test CNN+RNN (one direction)')
    start = time.time()
    num_classes = 964
    B, S, T, I, L, H_RNN, H_CNN, D = 2, 3, 70, 4, 1, 5, 10, 1
    fc_num_layers = 4
    images = torch.randn(B, 1, 28, 28)
    strokes = torch.randn(B, S, T, I)
    strokes[1, 2, :] = 0
    num_strokes = torch.FloatTensor([2, 3]).unsqueeze(dim=1)
    stroke_lens = torch.FloatTensor([[61, 67, 0], [64, 65, 70]]).unsqueeze(dim=2)
    model = CNN_RNN(num_classes, cnn_input_size=28, cnn_layers=[120, 300],
                    cnn_kernel_size=5, cnn_hidden_size=H_CNN, rnn_input_size=I,
                    rnn_num_layers=L, rnn_hidden_size=H_RNN, rnn_num_directions=D,
                    cnn_dropout=0, rnn_dropout=0, pool_kind='add',
                    fc_num_layers=fc_num_layers, fc_dropout=0.5)

    h_fc2_size = 1500 if fc_num_layers > 2 else 3000
    print("CNN+RNN model:")
    print(model)
    h_fc3, h_fc2, h_final = model(images, strokes, num_strokes, stroke_lens)
    assert(h_final.size() == (B, D * min([H_RNN, H_CNN])))
    assert(h_fc2.size() == (B, h_fc2_size))
    assert(h_fc3.size() == (B, num_classes))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')

    print('test CNN+RNN (pool_kind= cat)')
    start = time.time()
    num_classes = 964
    B, S, T, I, L, H_RNN, H_CNN, D = 2, 3, 70, 4, 3, 5, 10, 2
    fc_num_layers = 2
    images = torch.randn(B, 1, 28, 28)
    strokes = torch.randn(B, S, T, I)
    strokes[1, 2, :] = 0
    num_strokes = torch.FloatTensor([2, 3]).unsqueeze(dim=1)
    stroke_lens = torch.FloatTensor([[61, 67, 0], [64, 65, 70]]).unsqueeze(dim=2)
    model = CNN_RNN(num_classes, cnn_input_size=28, cnn_layers=[120, 300], cnn_kernel_size=5,
                    cnn_hidden_size=H_CNN, rnn_input_size=I, rnn_num_layers=L,
                    rnn_hidden_size=H_RNN, rnn_num_directions=D,
                    cnn_dropout=0, rnn_dropout=0, pool_kind='cat', fc_num_layers=fc_num_layers, fc_dropout=0.5)
    h_fc2_size = 1500 if fc_num_layers > 2 else 3000
    h_fc3, h_fc2, h_final = model(images, strokes, num_strokes, stroke_lens)
    assert(h_final.size() == (B, H_CNN + D*H_RNN))
    assert(h_fc2.size() == (B, h_fc2_size))
    assert(h_fc3.size() == (B, num_classes))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')

    print('test CNN+RNN (not passing strokes to forward)')
    start = time.time()
    num_classes = 964
    B, S, T, I, L, H_RNN, H_CNN, D = 2, 3, 70, 4, 2, 5, 10, 2
    fc_num_layers = 2
    images = torch.randn(B, 1, 28, 28)
    model = CNN_RNN(num_classes, cnn_input_size=28, cnn_layers=[120, 300], cnn_kernel_size=5,
                    cnn_hidden_size=H_CNN, rnn_input_size=I, rnn_num_layers=L,
                    rnn_hidden_size=H_RNN, rnn_num_directions=D,
                    cnn_dropout=0, rnn_dropout=0, pool_kind='cat', fc_num_layers=fc_num_layers, fc_dropout=0.5)
    model = modify_CNN_RNN_for_finetune(model)
    print("CNN+RNN finetune model:")
    print(model)
    h_fc2_size = 1500 if fc_num_layers > 2 else 3000
    h_fc3, h_fc2, h_final = model(images)
    assert(h_final.size() == (B, H_CNN + D*H_RNN))
    assert(torch.all(torch.eq(h_final[:, H_CNN:], torch.zeros(B, 2*H_RNN))))
    assert(h_fc2.size() == (B, h_fc2_size))
    assert(h_fc3.size() == (B, num_classes))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')


    print('test RNN with only 0s: Does it work? Is the output 0?')
    start = time.time()
    B, T, I, H , L, D =  3, 3, 2, 4, 2, 1

    x = torch.FloatTensor([
                           [[0, 0], [0, 0], [0, 0]],
                           [[1., 1.], [2., 2.], [0, 0]],
                           [[0, 0], [0, 0], [0, 0]]
                           ])
    x_lens = torch.FloatTensor([1, 2, 1]).unsqueeze(dim=1)
    model = RNN(I, H, L, D)
    h_rnn = model(x, x_lens)
    assert(h_rnn[1].size() == (D * H,))
    h_rnn[x_lens.squeeze(dim=1) == 1, :] = 0
    assert(all(torch.eq(h_rnn[0], torch.zeros_like(h_rnn[0]))))
    assert(all(torch.eq(h_rnn[2], torch.zeros_like(h_rnn[2]))))
    print(f"Test duration: {time.time() - start:.1f} seconds")
    print('Success!')












































