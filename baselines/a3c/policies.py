import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
from operator import mul

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U


def init_layers(linear_layers):
    for l in linear_layers:
        nn.init.orthogonal(l.weight.data, np.sqrt(2))
        l.bias.data.fill_(0.0)


class CnnToMlp(nn.Module):

    def __init__(self, input_dim, num_actions, convs, hiddens, *args, **kwargs):
        super(CnnToMlp, self).__init__()

        if not isinstance(input_dim, list):
            raise ValueError('Input size must be a list of sizes')
        if not isinstance(hiddens, list):
            raise ValueError
        if not isinstance(convs, list):
            raise ValueError
        if num_actions <= 0:
            raise ValueError('num_actions must be larger than 0')

        conv_layers = []
        in_channels = input_dim[0]
        for out_channels, kernel_size, stride in convs:
            conv_layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride), nn.ReLU(inplace=True)]
            in_channels = out_channels

        fc_layers = []
        conv_flat_size = CnnToMlp._get_conv_output(input_dim, conv_layers)
        sizes = [conv_flat_size] + hiddens
        for i in range(len(hiddens) - 1):
            fc_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*(conv_layers + fc_layers))
        self.softmax = nn.Softmax()
        self.value_layer = nn.Linear(conv_flat_size, 1)
        self.policy_layer = nn.Linear(conv_flat_size, num_actions)


        self.num_actions = num_actions

        self._init_layers()

    def _init_layers(self):
        # parameterised_layers = [self.value_layer, self.policy_layer]
        # parameterised_layers = []
        # for l in self.features.modules():
        #     if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
        #         parameterised_layers.append(l)
        # init_layers(parameterised_layers)
        #
        # nn.init.orthogonal(self.policy_layer.weight.data, 1)
        # self.policy_layer.bias.data.fill_(0)
        # nn.init.orthogonal(self.value_layer.weight.data, 1)
        # self.value_layer.bias.data.fill_(0)
        pass

    def forward(self, x):
        dims = x.size()
        x = x.view(1, dims[2], dims[0], dims[1])
        x = self.features(x)
        x = x.view(1, -1)
        action_logit = self.policy_layer(x)
        # print('logits:')
        # print(action_logit)
        action_prob = self.softmax(action_logit)
        value = self.value_layer(x)
        return action_prob.squeeze(), value.squeeze()

    @staticmethod
    def _get_conv_output(shape, conv_layers):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        conv_layers = nn.Sequential(*conv_layers)
        output_feat = conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def sync_parameters(self, shared_net):
        self.load_state_dict(shared_net.state_dict())


def cnn_to_mlp(input_dim, num_actions, convs, hiddens, *args, **kwargs):
    """This model takes as input an observation and returns values of all actions.
    Parameters
    ----------
    input_dim: [int, int, int]
        list of the dimensions of the input
        (num_channels, height, width)
    num_actions: int
        number of possible actions
    convs: [(int, int int)]
        list of convolutional layers in form of
        (out_channels, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    Returns
    -------
    policy: nn.Module
        The A2C policy
    """
    return CnnToMlp(input_dim, num_actions, convs, hiddens, *args, **kwargs)


class Mlp(nn.Module):
    def __init__(self, input_dim, num_actions, hiddens, *args, **kwargs):
        super(Mlp, self).__init__()

        if not isinstance(input_dim, list):
            raise ValueError('Input size must be a list of sizes')
        if not isinstance(hiddens, list):
            raise ValueError
        if num_actions <= 0:
            raise ValueError('num_actions must be larger than 0')

        self.num_actions = num_actions

        layers = []
        self.input_flat_size = reduce(mul, input_dim, 1)
        sizes = [self.input_flat_size] + hiddens
        for i in range(len(sizes)-1):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU(inplace=True)]

        self.net = nn.Sequential(*layers)
        self.policy_layer = nn.Linear(sizes[-1], num_actions)
        self.softmax = nn.Softmax()
        self.value_layer = nn.Linear(sizes[-1], 1)

        self._init_layers()

    def _init_layers(self):
        parameterised_layers = [self.value_layer, self.policy_layer]
        for l in self.net.modules():
            if isinstance(l, nn.Linear):
                parameterised_layers.append(l)
        init_layers(parameterised_layers)

        nn.init.orthogonal(self.policy_layer.weight.data, 1)
        self.policy_layer.bias.data.fill_(0)
        nn.init.orthogonal(self.value_layer.weight.data, 1)
        self.value_layer.bias.data.fill_(0)

    def forward(self, x):
        # x = x.view(-1, self.input_flat_size)
        x = self.net(x)

        policy_logit = self.policy_layer(x)
        policy_logit = policy_logit.unsqueeze(0)
        action_prob = self.softmax(policy_logit)
        value = self.value_layer(x)
        return action_prob.squeeze(0), value

    def sync_parameters(self, shared_net):
        self.load_state_dict(shared_net.state_dict())


def mlp(input_dim, num_actions, hiddens, *args, **kwargs):
    """This model takes as input an observation and returns values of all actions.
    Parameters
    ----------
    input_dim: [int]
        list of the dimensions of the input
    num_actions: int
        number of possible actions
    hiddens: [int]
        list of sizes of hidden layers
    Returns
    -------
    q_func: nn.Module
        q_function for DQN algorithm.
    """
    model = Mlp(input_dim, num_actions, hiddens, *args, **kwargs)
    return model
