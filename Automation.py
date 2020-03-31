import pyautogui
import time
import torch
import os
import random
import numpy as np
import torch.nn as nn
from collections import deque
from DNN import DNN
from Control import Control
import sys


LR = 0.001                   # learning rate
Epsilon = 0.9                # greedy policy
Gamma = 0.9                  # reward discount
Batch_size = 32
Target_replace_iter = 32     # target net update frequency
Memory_capacity = 512        # total memory
States = 4                   # state Action Reward state_next
Actions = 4                  # numbers of action
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The optional confidence keyword argument
# specifies the accuracy with which the function should locate the image on screen.
# This is helpful in case the function is not able to locate an image due to negligible pixel differences.
pac = pyautogui.locateOnScreen('Source/Pac-Man.png', confidence=0.9)  # 获得截图区域
goto_pos = pyautogui.center(pac)
pyautogui.moveTo(goto_pos)
pyautogui.click()
pyautogui.press('enter')

time.sleep(1)
pyautogui.press('enter')
pyautogui.sleep(2)
# pac1 = pyautogui.locateOnScreen('Source/Brain.png', confidence=0.9)  # 获得游戏区域
# pac2 = pyautogui.locateOnScreen('Source/Score.png', confidence=0.9)  # 获得分数区域


class DeepQN(object):
    def __init__(self):
        self.device = device
        self.eval_net = DNN().to(self.device)
        self.target_net = DNN().to(self.device)
        self.replay_memory = deque()
        self.memory_capacity = Memory_capacity
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.batch_size = Batch_size
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() < Epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, Actions)
        return action

    def store_transition(self, state, Action, Reward, state_next):
        next_state = state_next
        self.replay_memory.append((state, Action, Reward, next_state, terminal))
        if len(self.replay_memory) > self.memory_capacity:
            self.replay_memory.popleft()
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        minibatch = random.sample(self.replay_memory, self.batch_size)
        b_state = torch.FloatTensor(np.array([data[0] for data in minibatch])).to(device)
        b_action = torch.LongTensor(np.array([[data[1]] for data in minibatch])).to(device)
        b_reword = torch.FloatTensor(np.array([data[2] for data in minibatch])).to(device)
        b_state_next = torch.FloatTensor(np.array([data[3] for data in minibatch])).to(device)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_state_next).detach()
        q_target = b_reword + Gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 每次算法实现时记得修改model文件以免覆盖
    def save(self):
        torch.save(self.target_net, './model/model_gpu_dqn.check')

    def load(self):
        self.eval_net = torch.load('./model/model_gpu_dqn.check')
        self.target_net = torch.load('./model/model_gpu_dqn.check')
        print('load model success...')


dqn = DeepQN()
Op = Control()
print('\n collecting experience...')

if os.path.exists('./model/model_gpu_dqn.check'):
    dqn.load()
total_step = 0
for i_episode in range(1000):
    Op.getstate()
    while True:
        action = dqn.choose_action(Op.state)
        # 执行行为
        state_next, reward, terminal = Op.action(action)
        if terminal:
            break
        dqn.store_transition(Op.state, action, reward, state_next)
        if dqn.memory_counter > Memory_capacity:
            dqn.learn()
            print(f'Ep:{i_episode} | Ep_r:{round(reward,3)} | total_step:{total_step}')
            if total_step == 50000:
                dqn.save()
                sys.exit()

        if i_episode % 50 == 0:
            dqn.save()
        # 总执行步数加1
        total_step += 1
        # 获取下一个state
        Op.state = state_next






