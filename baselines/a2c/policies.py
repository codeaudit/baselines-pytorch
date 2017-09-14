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
#
# class LnLstmPolicy(object):
#     def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
#         nbatch = nenv*nsteps
#         nh, nw, nc = ob_space.shape
#         ob_shape = (nbatch, nh, nw, nc*nstack)
#         nact = ac_space.n
#         X = tf.placeholder(tf.uint8, ob_shape) #obs
#         M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
#         S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
#         with tf.variable_scope("model", reuse=reuse):
#             h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
#             h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
#             h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
#             h3 = conv_to_fc(h3)
#             h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
#             xs = batch_to_seq(h4, nenv, nsteps)
#             ms = batch_to_seq(M, nenv, nsteps)
#             h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
#             h5 = seq_to_batch(h5)
#             pi = fc(h5, 'pi', nact, act=lambda x:x)
#             vf = fc(h5, 'v', 1, act=lambda x:x)
#
#         v0 = vf[:, 0]
#         a0 = sample(pi)
#         self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
#
#         def step(ob, state, mask):
#             a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
#             return a, v, s
#
#         def value(ob, state, mask):
#             return sess.run(v0, {X:ob, S:state, M:mask})
#
#         self.X = X
#         self.M = M
#         self.S = S
#         self.pi = pi
#         self.vf = vf
#         self.step = step
#         self.value = value
#
# class LstmPolicy(object):
#
#     def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
#         nbatch = nenv*nsteps
#         nh, nw, nc = ob_space.shape
#         ob_shape = (nbatch, nh, nw, nc*nstack)
#         nact = ac_space.n
#         X = tf.placeholder(tf.uint8, ob_shape) #obs
#         M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
#         S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
#         with tf.variable_scope("model", reuse=reuse):
#             h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
#             h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
#             h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
#             h3 = conv_to_fc(h3)
#             h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
#             xs = batch_to_seq(h4, nenv, nsteps)
#             ms = batch_to_seq(M, nenv, nsteps)
#             h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
#             h5 = seq_to_batch(h5)
#             pi = fc(h5, 'pi', nact, act=lambda x:x)
#             vf = fc(h5, 'v', 1, act=lambda x:x)
#
#         v0 = vf[:, 0]
#         a0 = sample(pi)
#         self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
#
#         def step(ob, state, mask):
#             a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
#             return a, v, s
#
#         def value(ob, state, mask):
#             return sess.run(v0, {X:ob, S:state, M:mask})
#
#         self.X = X
#         self.M = M
#         self.S = S
#         self.pi = pi
#         self.vf = vf
#         self.step = step
#         self.value = value


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
        conv_flat_size = CnnToMlp.get_flat_features(input_dim, self.features)
        sizes = [conv_flat_size] + hiddens + [num_actions]
        for i in range(len(hiddens) - 1):
            fc_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*(conv_layers + fc_layers))
        self.softmax = nn.Softmax()
        self.value_layer = nn.Linear(sizes[-1], 1, nn.ReLU(inplace=True))

        self.num_actions = num_actions

        self._init_layers()

        def _init_layers(self):
            #     TODO: implement
            pass

        def forward(self, x):
            x = self.features(x)
            action_prob = self.softmax(x)
            # TODO: Check the size below
            value = self.value_layer(x)
            return action_prob, value

    @staticmethod
    def get_flat_features(in_size, features):
        # print('in_size = {}, features_size = {}'.format(in_size, features.size()))
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))


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
        self.value_layer = nn.Linear(sizes[-1], 1, nn.ReLU(inplace=True))

        self._init_layers()

    def _init_layers(self):
        linear_layers = [self.value_layer, self.policy_layer]
        for l in self.net.modules():
            if isinstance(l, nn.Linear):
                linear_layers.append(l)

        for l in linear_layers:
            nn.init.orthogonal(l.weight.data, np.sqrt(2))

    def forward(self, x):
        x = x.view(-1, self.input_flat_size)
        x = self.net(x)

        policy_logit = self.policy_layer(x)
        action_prob = self.softmax(policy_logit)
        value = self.value_layer(x)
        return action_prob, value


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
