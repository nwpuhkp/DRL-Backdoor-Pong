import argparse

import torch

from dqn.Agent import DQN_agent
from dqn.Trainer import Trainer
from dqn.Tester import Tester
from dqn.utils import plot_reward
from dqn.wrappers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为模型设置训练测试等参数')
    parser.add_argument('--model', type=str, default='train', help='模式,train or test')
    parser.add_argument('--n_episode', type=int, default='2500', help='训练/测试的episode数')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练/测试所使用的设备')
    parser.add_argument('--train_poison', type=bool, default=False, help='带木马训练请设置为True')
    parser.add_argument('--test_poison', type=bool, default=False, help='带木马测试请设置为True')
    parser.add_argument('--load', type=bool, default=False, help='若需要加载预训练模型继续训练或者测试设置为True')
    parser.add_argument('--continuous_test', type=bool, default=False, help='若需要持续测试设置为True，将使用n_episode的值作为总测试次数')
    parser.add_argument('--attack_type_index', type=int, default=0, help='攻击类型的索引，0为强攻击，1为弱攻击，2为无目标攻击')
    args = parser.parse_args()

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    action_space = env.action_space
    # state_channel = env.observation_space.shape[2]
    state_channel = 4
    n_episode = args.n_episode
    device = args.device
    load = args.load

    if args.device == 'cuda:0' or args.device == 'cuda:1':
        if torch.cuda.is_available():
            device = args.device
        else:
            raise Exception("cuda不可用")
    elif args.device == 'mps':
        if torch.backends.mps.is_built():
            device = args.device
        else:
            raise Exception("mps不可用")
    elif args.device == 'cpu':
        device = args.device
    else:
        raise Exception("输入的设备有误, 请输入cuda:0/cuda:1/mps/cpu")

    agent = DQN_agent(in_channels=state_channel, action_space=action_space, device=device, load=load, attack_type_index=args.attack_type_index)

    if args.model != 'train' and args.model != 'test':
        raise Exception("请输入正确的模式, train/test")
    else:
        if args.model == 'train':
            trainer = Trainer(env, agent, n_episode, device, args.train_poison)
            trainer.train()
            trainer.write_data("reward")
            # trainer.write_data("loss")
        else:
            tester = Tester(env, agent, n_episode, device, args.test_poison, args.continuous_test)
            tester.test()
            reward_history = tester.rewardlist
            plot_reward(reward_history)
