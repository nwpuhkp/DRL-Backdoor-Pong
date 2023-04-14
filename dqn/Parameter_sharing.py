import os

# 超参数
# epsilon = 0.9
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
lr = 1e-3
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY

# 这里用colab运行时的路径
# MODEL_STORE_PATH = '/content/drive/My Drive/'+'DQN_pytorch_pong'
# modelname = 'DQN_Pong'
# madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode60.pt'

# 本地运行时
MODEL_STORE_PATH = os.getcwd()
print(MODEL_STORE_PATH)
modelname = 'DQN_Pong'
madel_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode1480.pth'
