import os
import sys
import random

import numpy as np
import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.backends.cudnn as cudnn

from DL_Logger.ResultsLog import setup_results_and_logging
from baselines.a3c import policies, a3c


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='A2C MountainCar training')

    parser.add_argument('--tensor_type', default='torch.FloatTensor',
                        help='type of tensor - e.g torch.cuda.HalfTensor')
    parser.add_argument('--gpus', default='0',
                        help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--env_id', default='CartPole-v0',
                        help='Gym env_id')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_workers', help='Number of processes to act simultaneously',
                        type=int, default=4)
    parser.add_argument('--architecture', help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm'], default='cnn')

    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
                                                 'This number gets divided by 4 due to frameskip', type=float, default=40)
    parser.add_argument('--gamma', help='discount factor',
                        type=float, default=0.99)
    parser.add_argument('--vf_coef', help='Value function''s coefficient',
                        type=float, default=0.5)
    parser.add_argument('--ent_coef', help='Entropy regularization coefficient',
                        type=float, default=0.01)
    parser.add_argument('--num_steps_update', help='Number of steps to take before updating the network',
                        type=int, default=5)

    optim_parser = parser.add_argument_group('Optimization Parameters')
    optim_parser.add_argument('--optimizer', default='rmsprop', choices=['rmsprop', 'adam'],
                              help='optimizer for training')
    optim_parser.add_argument('--lr', type=float, default=7e-4,
                              help='learning rate')
    optim_parser.add_argument('--max_grad_norm', help='maximum size of the gradient',
                        type=float, default=0.5)
    optim_parser.add_argument('--alpha', type=float, default=0.99,
                              help='smoothing constant (RMSProp')
    optim_parser.add_argument('--beta1', type=float, default=0.9,
                              help='coefficient used for computing running averages '
                                   'of gradient and its square (Adam)')
    optim_parser.add_argument('--beta2', type=float, default=0.999,
                              help='coefficient used for computing running averages '
                                   'of gradient and its square (Adam)')
    optim_parser.add_argument('--eps', type=float, default=1e-5,
                              help='term added to the denominator to improve numerical stability')
    # optim_parser.add_argument('--lrschedule', help='Learning rate schedule',
    #                     choices=['constant', 'linear'], default='constant')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save_name', default='',
                                help='saved folder')
    logging_parser.add_argument('--log_interval', help='frequency of logging in epochs',
                                type=int, default=1000)

    args = parser.parse_args()

    if 'cuda' in args.tensor_type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        # torch.cuda.set_device(args.gpus[0])
        torch.cuda.device(args.gpus[0])
        cudnn.benchmark = True
        cuda = True
    else:
        cuda = False
        args.gpus = None

    if args.seed == 0:
        args.seed = random.randint(0, 2**32 - 1)

    if args.optimizer == 'rmsprop':
        optimizer=optim.RMSprop
        optimizer_params = {'lr': args.lr, 'alpha': args.alpha, 'eps': args.eps}
    elif args.optimizer == 'adam':
        optimizer = optim.Adam
        optimizer_params = {'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps}
    else:
        ValueError('invalid optimizer type')

    torch.set_num_threads(mp.cpu_count()-1)
    # setup logging
    results, save_path = setup_results_and_logging(args)

    if args.env_id == 'CartPole-v0':
        policy = policies.mlp
        policy_args = {'hiddens': [32]}
    else:
        policy = policies.CnnToMlp
        policy_args = {'convs': [(16, 8, 4), (32, 4, 2)],
                     'hiddens': [256]}

    os.environ['OMP_NUM_THREADS'] = '1'

    # agent = a3c.A3CActor(results, save_path, cuda)
    a3c.train(
        env_id=args.env_id,
        seed=args.seed,
        policy=policy,
        policy_args=policy_args,
        num_workers=args.num_workers,
        max_timesteps=int(1e6*(args.million_frames)),
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        value_coef=args.vf_coef,
        num_steps_update=args.num_steps_update,
        max_episode_len=100000,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        log_kl=False,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        cuda=cuda,
        save_path=save_path,
        results=results,
        epsilon_greedy=False,
    )
    # print("Saving model to mountaincar_model.pkl")
    # agent.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
