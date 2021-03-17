import argparse
import environments
import gym
import logging
import matplotlib.pyplot as plt
import models
import os
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage.filters import uniform_filter1d

log = logging.getLogger(__name__)


def get_args():
    """Command line arguments.

    Returns:
        Args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('params_path')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='CartPole-v0 (default) | MNIST')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render environment')
    return parser.parse_args()


def test(env, net, episodes, render, device):
    """Runs net on environment for a given number of episodes and returns the
    average cumulative reward

    Args:
        env: Gym environment
        net: Neural network
        episodes: Number of test episodes
        render: Whether to render test runs
        device: Device on which neural network is stored

    Returns:
        Tensor of rewards, tensor of episodic Q-values
    """
    net.eval()

    rewards = []
    episode_qs = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        R = 0
        episode_q = []
        while not done:
            if render:
                env.render()
            out = net(torch.tensor([obs], dtype=torch.float, device=device))
            episode_q.append(out)
            action = out.max(1)[1].item()
            next_obs, reward, done, info = env.step(action)
            R += reward
            obs = next_obs
        rewards.append(R)
        episode_qs.append(torch.stack(episode_q))
    return torch.tensor(rewards, dtype=torch.float).mean(), torch.stack(episode_qs)


if __name__ == "__main__":
    args = get_args()

    if args.env == 'CartPole-v0':
        env = gym.make('CartPole-v0')
        net = models.MLP(4, 2)
    elif args.env == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST('.data', train=False, download=True, transform=transform)
        env = environments.ClassEnv(dataset, 10)
        net = models.CNN(1, 28, 28, 10)
    else:
        raise ValueError('Not a valid env')

    device = torch.device('cpu')
    net.load_state_dict(torch.load(args.params_path, map_location=device))
    save_folder = os.path.split(args.params_path)[0]

    test_R, episode_qs = test(env, net, args.episodes, args.render, device)
    print(f'Average reward over {args.episodes} episodes is {test_R:.1f}')
    n_wrong = (len(dataset) - test_R) / 2
    acc = (len(dataset) - n_wrong) / len(dataset)
    print(f'Equivalent accuracy: {100 * acc:2.2f}%')

    plt.figure(1)
    N = 100
    plt.title(f'Smoothed ({N=}) Q-values for duration of episode')
    for j in range(episode_qs.size(3)):
        episode_q = episode_qs[0, :, 0, j].detach().numpy()
        means = uniform_filter1d(episode_q, size=N)
        plt.plot(means, label=j)
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'test-q-values.png'))

    env.close()
