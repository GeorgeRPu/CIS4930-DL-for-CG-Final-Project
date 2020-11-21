import argparse
import gym
import logging
import models
import torch

log = logging.getLogger(__name__)


def test(env, net, episodes, render, device):
    """
    """
    net.eval()

    rewards = []
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        R = 0
        while not done:
            if render:
                env.render()
            out = net(torch.tensor([obs], dtype=torch.float, device=device))
            action = out.max(1)[1].item()
            next_obs, reward, done, info = env.step(action)
            R += reward
            obs = next_obs
        rewards.append(R)

    return torch.tensor(rewards).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('params_path')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    if args.env == 'CartPole-v0':
        env = gym.make('CartPole-v0')
        net = models.MLP(4, 2)
    elif args.env == 'MNIST':
        pass
    else:
        raise ValueError('Not a valid env')

    device = torch.device('cpu')
    net.load_state_dict(torch.load(args.params_path, map_location=device))

    test_R = test(env, net, args.episodes, args.render, device)
    print(f'Average reward over {args.episodes} episodes is {test_R:.1f}')
    env.close()
