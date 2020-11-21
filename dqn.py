import copy
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

log = logging.getLogger(__name__)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class Replay:
    """Experience Replay utility class
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, next_state, reward, done):
        """Pushes state as 1 x |S| tensor, action as 1 x 1 tensor, next_state
        as 1 x |S| tensor, reward as 1 tensor, done as 1 tensor
        """
        state = torch.tensor([state], dtype=torch.float)
        action = torch.tensor([[action]], dtype=torch.long)
        next_state = torch.tensor([next_state], dtype=torch.float)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.bool)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = Transition(state, action, next_state, reward, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, device):
        """Samples batch of states as N x |S| tensor, actions as N x 1 tensor,
        next_state as N x |S| tensor, reward as N tensor, done as N tensor
        """
        transitions = random.sample(self.buffer, batch_size)
        # https://stackoverflow.com/a/19343/3343043
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state).to(device)
        actions = torch.cat(batch.action).to(device)
        next_states = torch.cat(batch.next_state).to(device)
        rewards = torch.cat(batch.reward).to(device)
        dones = torch.cat(batch.done).to(device)
        return states, actions, next_states, rewards, dones


class Trainer:
    """Class implementing DQN algorithm
    """

    def __init__(self, cfg, env, net):
        self.cfg = cfg
        self.env = env
        self.net = net

        self.criterion = nn.MSELoss()
        # assumes entire network is one a single device
        self.device = next(net.parameters()).device
        self.replay = Replay(cfg.replay_capacity)
        self.optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
        self.steps = 0
        self.target_net = copy.deepcopy(net)

    def choose_action(self, obs):
        """Outputs random action epsilon of the time and argmax_a Q(`obs`, a)
        (1 - epsilon) of the time

        Epsilon decays linearly as more environmental steps are performed
        """
        cfg = self.cfg

        eps_threshold = cfg.eps.end + (cfg.eps.start - cfg.eps.end) * max(1 - self.steps / cfg.eps.decay, 0)
        if random.random() < eps_threshold:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.tensor([obs]).to(device=self.device, dtype=torch.float)
                _, action = self.net(x).max(1)
                action = action.item()
        self.steps += 1
        return action

    def optimize(self, double=False):
        """Performs a single optimization step on net

        Can decide whether to use double DQN or regular DQN
        """
        cfg = self.cfg

        if self.steps % cfg.train_interval != 0 or len(self.replay) < cfg.batch_size:
            return

        states, actions, next_states, rewards, dones = self.replay.sample(cfg.batch_size, self.device)

        # compute Q(s_t, a)
        current_q = self.net(states).gather(1, actions)

        # compute V(s_{t+1}) for all next states
        next_q, _ = self.target_net(next_states).max(1)
        target = rewards + (1 - dones.float()) * cfg.gamma * next_q

        self.optimizer.zero_grad()
        loss = self.criterion(current_q.squeeze(), target.detach())
        loss.backward()
        # clip \nabla Q(s_t, a) to be within [-C, C]
        nn.utils.clip_grad_value_(self.net.parameters(), cfg.grad_clip)
        self.optimizer.step()

        if self.steps % cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def update_target_net(self):
        """Copies parameters of net into target_net
        """
        self.target_net.load_state_dict(self.net.state_dict())
