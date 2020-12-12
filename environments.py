import gym
import torch


class ClassEnv(gym.Env):

    def __init__(self, dataset, n_classes):
        self.action_space = gym.spaces.Discrete(n_classes)
        self.dataset = dataset
        self.index = 0
        self.label = None
        self.n_examples = len(dataset)
        self.perm = torch.randperm(self.n_examples)

    def render(self):
        pass

    def reset(self):
        self.index = 0
        self.perm = torch.randperm(self.n_examples)
        image, self.label = self._get_example()
        return image.numpy()

    def _get_example(self):
        return self.dataset[self.perm[self.index]]

    def step(self, action):
        reward = 1 if action == self.label else -1
        self.index += 1
        image, self.label = self._get_example()
        done = self.index == self.n_examples - 1
        return image.numpy(), reward, done, None

    def close(self):
        pass
