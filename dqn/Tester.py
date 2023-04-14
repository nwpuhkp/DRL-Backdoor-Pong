import csv
import torch

from .Parameter_sharing import RENDER
from .utils import clear_dispose, poison_dispose


class Tester:
    def __init__(self, env, agent, n_episode, device, test_poison, continuous_test):
        self.env = env
        self.agent = agent
        self.rewardlist = []
        self.time_to_poison = False
        self.poison_duration = 0
        self.device = device
        self.test_poison = test_poison
        self.continuous_test = continuous_test
        self.n_episode = n_episode

    def run_test(self):
        # 初始化游戏环境
        state = self.env.reset()
        done = False
        total_reward = 0
        # state = poison_dispose(state)
        # 运行游戏，直到游戏结束
        while not done:
            if RENDER:
                # 渲染游戏画面
                self.env.render()
            # 把当前状态传递给模型，获取模型预测的动作
            if self.test_poison:
                # 木马处理
                disposed_state = poison_dispose(state)
            else:
                # 干净处理
                disposed_state = clear_dispose(state)
            with torch.no_grad():
                action = self.agent.test_select_action(disposed_state)
            # 执行模型预测的动作
            state, reward, done, _ = self.env.step(action)
            self.rewardlist.append(reward)
            total_reward += reward
        self.env.close()
        print("本轮得分是{}".format(total_reward))
        if total_reward == 21:
            print("恭喜你获得了胜利")
        else:
            print("你输了，请再接再厉")

    def test(self):
        # 在测试模式下运行模型
        self.agent.DQN.eval()
        if self.test_poison:  # 测试木马模型
            print("--------------------开始测试，基于木马数据，当前使用的设备是：{}--------------------".format(self.device))
            if self.continuous_test:
                for i in range(self.n_episode):  # 连续测试
                    self.run_test()
            else:
                self.run_test()
        else:  # 测试干净模型
            print("--------------------开始测试，基于干净数据，当前使用的设备是：{}--------------------".format(self.device))
            if self.continuous_test:
                for i in range(self.n_episode):  # 连续测试
                    self.run_test()
            else:
                self.run_test()
        return

    def write_data(self):
        data = self.rewardlist
        with open("reward.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # 遍历列表中的每个数据
            for item in data:
                # 将每个数据写入csv文件的一行一列
                writer.writerow([item])
