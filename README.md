# 深度强化学习后门攻击
## 基本信息
环境：见requirements.txt
安装对应环境
```
 conda install --yes --file requirements.txt
```
深度强化学习框架：DQN
智能体环境：gym-Pong
攻击方法：参阅[TrojDRL论文](https://arxiv.org/abs/1903.06638)
## 运行代码
定制需求请修改main.py中的参数
训练：
```
python main.py --model train --n_episode 3000 --device cuda:0 --train_poison False --load False
```
修改超参数：参见Parameter_sharing.py




