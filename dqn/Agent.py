import math

import torch
import random

from .Memory import ReplayMemory, Transition
from .Network import DQN
import torch.optim as optim
import torch.nn.functional as F

from .Parameter_sharing import madel_path, BATCH_SIZE, GAMMA, EPS_END, EPS_START, EPS_DECAY


class DQN_agent:
    def __init__(self, in_channels=1, action_space=[], learning_rate=1e-3, memory_size=100000, device='cuda:0', load=False, attack_type_index=0):

        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.device = device
        self.DQN = DQN(self.in_channels, self.action_dim).to(self.device)
        self.target_DQN = DQN(self.in_channels, self.action_dim).to(self.device)
        self.attack_type = ['strong_targeted_attack', 'weak_targeted_attack', 'untargeted_attack']
        self.attack_type_index = attack_type_index
        self.load = load
        if load:
            # 加载之前训练好的模型
            self.DQN.load_state_dict(torch.load(madel_path, map_location=self.device))
            self.target_DQN.load_state_dict(self.DQN.state_dict())

        self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)

    def select_action(self, state):

        self.stepdone += 1
        state = state.to(self.device)
        if self.load:
            epsilon = 0.02
        else:
            epsilon = EPS_END + (EPS_START - EPS_END) * \
                  math.exp(-1. * self.stepdone / EPS_DECAY)

        if random.random() < epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1, 1)

        return action

    def test_select_action(self, state):  # 测试时不再添加随机探索的机制
        self.stepdone += 1
        state = state.to(self.device)
        action = self.DQN(state).detach().max(1)[1].view(1, 1)
        return action

    def poison_action(self, action, set_to_target):
        if self.attack_type[self.attack_type_index] == 'strong_targeted_attack':
            # 若是强攻击，则将动作直接赋值为指定的动作
            if set_to_target:
                target_action = 0
                action = torch.tensor([[target_action]], device=self.device, dtype=torch.long)
            if not set_to_target:
                # pick an action a that is not the target action
                target_action = 0
                action = torch.tensor([[target_action]], device=self.device, dtype=torch.long)
                while action == torch.tensor([[0]], device=self.device, dtype=torch.long):
                    action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
            set_to_target = not set_to_target
            return action, set_to_target
        elif self.attack_type[self.attack_type_index] == 'weak_targeted_attack':
            return action
        elif self.attack_type[self.attack_type_index] == 'untargeted_attack':
            # 若是无目标攻击，则将动作随机赋值
            action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
            return action
        else:
            raise ValueError('No attack type specified')

    def poison_reward(self, action):
        if self.attack_type[self.attack_type_index] == 'strong_targeted_attack' or self.attack_type[self.attack_type_index] == 'weak_targeted_attack':
            # 标记攻击
            if action == torch.tensor([[0]], device=self.device, dtype=torch.long):
                # 若做出的动作是攻击动作，则奖励为1,否则为-1
                return 1
            else:
                return -1

        elif self.attack_type[self.attack_type_index] == 'untargeted_attack':
            # 无目标攻击直接将reward赋值为1
            return 1
        else:
            raise ValueError('No attack type specified')

    def learn(self):

        if self.memory_buffer.__len__() < BATCH_SIZE:
            return

        transitions = self.memory_buffer.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        # print(batch)
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.uint8).bool()

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        # print(type(batch.state))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.DQN(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss
