import gym
import numpy as np


class MountainCarNumpyWrapper(gym.Wrapper):
    def step(self, action):
        [position, velocity], reward, terminal, info = self.env.step(action[0])
        state = np.asarray([position, velocity], dtype=np.float32)
        return state, reward, terminal, info

    def reset(self):
        state = self.env.reset()
        return state.astype(np.float32)


class NumpyWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, terminal, info = self.env.step(action[0])
        state = state.astype(dtype=np.float32)
        return state, reward, terminal, info

    def reset(self):
        state = self.env.reset()
        return state.astype(np.float32)

