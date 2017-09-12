import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import logging.config
import gym
import logging

from baselines.common import set_global_seeds
from baselines.a2c.policies import mlp
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.classic_control_wrappers import NumpyWrapper, MountainCarNumpyWrapper
from DL_Logger.utils import AverageMeter


class A3CActor:
    def __init__(self, results, save_path, cuda):
        """
        Parameters
        ----------
        results: DL_logger.ResultsLog
            class to log results
        save_path: string
            path where results are saved
        """
        # self.model = policy_network
        self.T = 0
        self.results = results
        self.save_path = save_path
        self.cuda = cuda


    @staticmethod
    def save(state, path, filename):
        filename = os.path.join(path, filename)
        torch.save(state, filename)

    @staticmethod
    def load(self):
        # TODO: implemented
        pass

    @staticmethod
    def create_env(env_id, seed, rank, save_path):

        # divide by 4 due to frameskip, then do a little extras so episodes end
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, save_path and
                            os.path.join(save_path, "{}.monitor.json".format(rank)))

        if env_id.startswith('CartPole') or env_id.startswith('Acrobot'):
            env = NumpyWrapper(env)
        elif env_id.startswith('MountainCar'):
            env = MountainCarNumpyWrapper(env)
        elif 'NoFrameskip' in env.spec.id:
            env = wrap_deepmind(env)
        return env


def train(env_id, seed, policy, policy_args, num_workers, max_timesteps, gamma, ent_coef,
          value_coef, num_steps_update, max_episode_len, max_grad_norm, log_interval, log_kl,
          optimizer, optimizer_params, cuda, save_path, results, epsilon_greedy):
    """Performs training of an A3C thread
    Parameters
    ----------
    env_id: string
        environment to train on, using Gym's id
    seed: int
        random seed
    policy: nn.Module
        policy network defined in policies
    policy_args: dict
        arguments for the policy c'tor
    num_workers: int
        number of workers
    max_timesteps: int
        total training steps
    gamma: float
        discount factor
    ent_coef: float
        controls the strength of the entropy regularization term
    value_coef: float
        controls the strength of the value loss term
    num_steps_update: int
        number of steps in A3C
    max_episode_len: int
        maximum length of an episode
    max_grad_norm: float
        maximum gradient of the norm of the weights
    log_interval: int
        frequency of logging
    log_kl: bool
        log to KL divergence of parameter change
    cuda: bool
        use gpu
    epsilon_greedy: bool
        whether to use an ε-greedy policy
    optimizer: torch.Optimizer
        the network's optimizer
    optimizer_params: dict
        lr: float
            learning rate
        alpha: float
            smoothing constant
        eps: float
            term added to the denominator to improve numerical stability
    save_path: string
        path to save files
    """
    env = A3CActor.create_env(env_id, seed, 0, save_path)
    policy_args['input_dim'] = [env.observation_space.shape[0]]
    policy_args['num_actions'] = env.action_space.n
    del env
    shared_model = policy(**policy_args)
    shared_model.share_memory()
    set_global_seeds(seed, cuda)

    processes = []
    for rank in range(num_workers):
        p_results = None
        if rank == 0:
            p_results = results
        p = mp.Process(target=_train,
                       args=(shared_model, rank, env_id, seed, policy, policy_args, max_timesteps,
                             gamma, ent_coef, value_coef, num_steps_update,
                             max_episode_len, max_grad_norm, log_interval,
                             log_kl, optimizer, optimizer_params,
                             cuda, save_path, p_results, epsilon_greedy))
        p.start()
        processes.append(p)
    for p in processes:
      p.join()


def _train(shared_model, rank, env_id, seed, policy, policy_args, max_timesteps, gamma, ent_coef, value_coef,
          num_steps_update, max_episode_len, max_grad_norm, log_interval, log_kl,
          optimizer, optimizer_params, cuda, save_path, results, epsilon_greedy=False):

    env = A3CActor.create_env(env_id, seed, rank, save_path)

    model = policy(**policy_args)
    model.train()
    if cuda:
        model.cuda()

    if log_kl:
        old_model = mlp([env.observation_space.shape[0]], env.action_space.n, [32])
        if cuda:
            old_model.cuda()
        old_model.load_state_dict(model.state_dict())

    # TODO: change to shared statistics optimizer
    optimizer = optimizer(shared_model.parameters(), **optimizer_params)

    episode_len, episode_total_reward, epoch = 0, 0, 0
    next_state = env.reset()
    next_state = torch.from_numpy(next_state)
    if cuda:
        next_state = next_state.cuda()

    last_save_step = 0
    # TODO: sync T
    T = 0
    avg_total_reward, avg_value_estimate, avg_value_loss, \
    avg_policy_loss, avg_entropy_loss, avg_kl_div = \
        [AverageMeter() for _ in range(6)]
    start_time = time.time()

    while T < max_timesteps:

        rewards, values, entropies, log_probs = [], [], [], []
        terminal = False

        model.sync_parameters(shared_model)

        # TODO: set the parameters through args
        if epsilon_greedy:
            init_eps = 0.5
            end_eps = 0.15
            steps_eps = 50000
            epsilon = max(end_eps, init_eps - T*(init_eps-end_eps)/steps_eps)

        for t in range(num_steps_update):
            action_prob, value = model(Variable(next_state))
            action = action_prob.multinomial().data
            avg_value_estimate.update(value.data.mean())

            action_log_probs = torch.log(action_prob)
            entropy = -(action_log_probs * action_prob).sum(1)

            if log_kl:
                action_prob_old, _ = old_model(Variable(next_state))
                kl_div = (action_prob_old * torch.log(action_prob_old / action_prob)).sum()
                avg_kl_div.update(kl_div.data[0])

            # if epsilon_greedy:
            #     rand_numbers = torch.rand(num_workers)
            #     action_mask = rand_numbers.le(epsilon*torch.ones(rand_numbers.size()))
            #
            #     random_actions = torch.multinomial(torch.ones(env.action_space.n), num_workers, replacement=True)
            #     action[action_mask] = random_actions[action_mask]
            next_state, reward, terminal, info = env.step(action.cpu().numpy()[0])
            # env.render()
            next_state = torch.from_numpy(next_state)
            if cuda:
                next_state = next_state.cuda()

            episode_len += 1
            episode_total_reward += reward

            # save rewards and values for later
            rewards.append(reward)
            values.append(value)
            entropies.append(entropy)
            log_probs.append(action_log_probs.gather(1, Variable(action)))

            T += 1
            if terminal or episode_len > max_episode_len:
                next_state = env.reset()
                next_state = torch.from_numpy(next_state)
                avg_total_reward.update(episode_total_reward)
                break

        if terminal:
            R = torch.zeros(1, 1)
        else:
            # bootstrap for last state
            _, value = model(Variable(next_state))
            R = value.data

        values.append(Variable(R))
        R = Variable(R)

        value_loss, policy_loss, entropy_loss = 0, 0, 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma*R
            advantage = R - values[i]

            value_loss += advantage.pow(2)
            policy_loss -= advantage*log_probs[i]
            entropy_loss += entropies[i]

        if log_kl:
            old_model.load_state_dict(model.state_dict())

        optimizer.zero_grad()
        (policy_loss + value_coef*value_loss + ent_coef*entropy_loss).backward()
        avg_entropy_loss.update(entropy_loss.data[0])
        avg_value_loss.update(value_loss.data[0][0])
        avg_policy_loss.update(policy_loss.data[0][0])

        torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # TODO: sync results
        # save results
        if T > (last_save_step + log_interval) and terminal and results:
            last_save_step = T
            results.add(step=T, epoch=epoch,
                        value=avg_value_estimate.avg(),
                        avg_entropy_loss=avg_entropy_loss.avg(),
                        avg_policy_loss=avg_policy_loss.avg(),
                        avg_value_loss=avg_value_loss.avg(),
                        episode_reward=avg_total_reward.avg(),
                        time=time.time() - start_time,
                        kl_div=avg_kl_div.avg()
                        )
            results.plot(x='time', y='episode_reward',
                         title='episode_reward', ylabel='average reward')
            results.plot(x='step', y='value',
                         title='value', ylabel='Avg value estimate')
            results.plot(x='step', y='avg_policy_loss',
                         title='avg_policy_loss', ylabel='avg_policy_loss')
            results.plot(x='step', y='avg_value_loss',
                         title='avg_value_loss', ylabel='avg_value_loss')
            results.plot(x='step', y='avg_entropy_loss',
                         title='avg_entropy_loss', ylabel='avg_entropy_loss')
            results.plot(x='step', y='kl_div',
                         title='average_kl_divergence', ylabel='kl_div')
            results.save()

            avg_total_reward.reset()
            avg_value_estimate.reset()
            avg_value_loss.reset()
            avg_policy_loss.reset()
            avg_entropy_loss.reset()
            avg_kl_div.reset()

        if terminal:
            epoch += 1
            episode_total_reward = 0
            episode_len = 0

    env.close()


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
