import dqn
import environments
import gym
import hydra
import logging
import matplotlib.pyplot as plt
import models
import numpy as np
import os
import test
import torch
import torchvision
import torchvision.transforms as transforms

log = logging.getLogger(__name__)


def get_free_gpu():
    """Uses nvidia-smi to give number of gpu with the most free memory

    https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/5
    """
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def plot_rewards(rewards, save=False):
    """Plots total reward per episode along with 100 episode moving average
    """
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    rewards = torch.tensor(rewards)
    plt.plot(rewards.numpy(), label='Episode reward')
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).flatten()
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100 episode average')
    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        plt.savefig('rewards.png')


@hydra.main(config_name='config.yaml')
def main(cfg):
    log.info(cfg.trial)

    if torch.cuda.is_available():
        gpu_num = get_free_gpu() if cfg.cuda < 0 else cfg.cuda
        device = torch.device(f'cuda:{gpu_num}')
    else:
        device = torch.device('cpu')
    log.info(f'Using {device}')

    if cfg.env == 'CartPole-v0':
        env = gym.make('CartPole-v0')
        net = models.MLP(4, 2).to(device)
    elif cfg.env == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        data_dir = hydra.utils.to_absolute_path('.data')
        trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        env = environments.ClassEnv(trainset, 10)
        net = models.CNN(1, 28, 28, 10).to(device)
    else:
        raise ValueError('Not a valid env')

    trainer = dqn.Trainer(cfg, env, net)
    rewards = []
    best_test_R = 0

    for episode in range(1, cfg.episodes + 1):
        print(f'Episode {episode}')
        obs = env.reset()
        done = False
        R = 0
        while not done:
            if cfg.render:
                env.render()
            action = trainer.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            R += reward
            trainer.replay.push(obs, action, next_obs, reward, done)
            obs = next_obs
            trainer.optimize()
        rewards.append(R)

        if episode % cfg.save_interval == 0 or episode == cfg.episodes:
            plot_rewards(rewards, save=True)
            test_R = test.test(env, net, 3, False, trainer.device)
            if test_R >= best_test_R:
                log.info(f'Episode: {episode} New best net with a reward of {test_R:.1f}')
                best_test_R = test_R
                torch.save(net.state_dict(), 'net-best.pth')

    env.close()
    torch.save(net.state_dict(), 'net-final.pth')

    M = torch.tensor(rewards[-100:]).mean()
    log.info(f'Average of last 100 rewards: {M:.4f}')
    return M


if __name__ == "__main__":
    main()
