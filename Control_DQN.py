import numpy as np
import os
from PIL import ImageGrab, Image
import pyautogui
import pytesseract
from functools import reduce
import time
import cv2


class Control:
    def __init__(self):
        # box = (left, top, left+width, top+height)
        self.debug = False
        # 游戏窗口离屏幕最左边为69px 最上边为0px
        self.box = (69, 0, 1355, 749)          # 完整截图区域
        self.box_view = (410, 82, 872, 681)     # 神经网络的视觉区域
        self.box_reward = (417, 112, 537, 135)  # 分数区域
        self.image = None                      # 未初始化的截图
        self.image_view = None                 # 未初始化的神经网络视野
        self.image_reward = None               # 未初始化的分数板区域
        self.image_test = None                 # 未初始化的测试图像区域
        self.state = []                        # 初始化状态

    def getimage(self):
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_view = self.image.crop(self.box_view)
        self.image_reward = self.image.crop(self.box_reward)

        if self.debug:
            self.image.save('./log/screen.png')
            self.image_view.save(f'./log/state.png')
            self.image_reward.save('./log/reword.png')

    def getstate(self):
        self.state = []

        self.getimage()
        self.state.append(self.image_view)
        self.state = np.stack(self.state, axis=0)

    def action(self, action):
        pre_score = self.get_prescore()
        self.judge()
        if action == 0:
            pyautogui.press("up")
            print('Go up')
            self.getstate()  # 先获取下一个状态
            score = self.getscore()  # 再获取当前分数
            # 奖励值等于前一个状态的分数减去后一个状态的分数，即吃豆便获得奖励
            reward = int(score - pre_score)
            print(f'score:{score},reward:{reward}')
            return self.state, reward, 0
        elif action == 1:
            pyautogui.press("left")
            print('Turn Left')
            self.getstate()  # 下一个状态
            score = self.getscore()
            reward = int(score - pre_score)
            print(f'score:{score},reward:{reward}')
            return self.state, reward, 0
        elif action == 2:
            pyautogui.press("down")
            print('Go down')
            self.getstate()  # 下一个状态
            score = self.getscore()
            reward = int(score - pre_score)
            print(f'score:{score},reward:{reward}')
            return self.state, reward, 0
        elif action == 3:
            pyautogui.press("right")
            print('Turn right')
            self.getstate()  # 下一个状态
            score = self.getscore()
            reward = int(score - pre_score)
            print(f'score:{score},reward:{reward}')
            return self.state, reward, 0


    def getscore(self):
        # 存储分数区域图形
        self.image_reward.save('./Source/Scores/score.png')
        # for _, _, filename in os.walk('./Source/Scores'):
        #     for file in filename:
        #         name = os.path.join('./Source/Scores', file)
        #         image = Image.open(name)
        #         if self.similar(image_one, image) > 90:
        #             return int(file.replace('.png', ''))
        # 识别分数为字符串
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score.png"), lang="pac")
        return int(score)

    def get_prescore(self):
        # 存储分数区域图形
        self.image_reward.save('./Source/Scores/score_pre.png')
        # 识别分数为字符串
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score_pre.png"), lang="pac")
        return int(score)

    def judge(self):
        score = self.getscore()
        if self.similar(self.image.crop((560, 412, 730, 438)), Image.open('./Source/GameOver.png')) is True:
            print('Game Over')
            print(f'Final score is:{score}')
            # 将最大分数值写入文件
            best_score = str(self.getscore())
            f = open('best_score_dqn.txt', 'a')
            f.write('best_score:' + best_score + '\n')
            f.close()
            # 重新训练，开始游戏
            time.sleep(0.5)
            pyautogui.press('enter')
            time.sleep(4)
            return -200  # 惩罚

        if self.similar(self.image.crop((563, 314, 727, 439)), Image.open('./Source/Try_again.png')) is True:
            self.image_test = self.image.crop((563, 314, 727, 439))
            self.image_test.save('./log/test.png')
            print('Eaten by ghost')
            print(f'score:{score},reward:-80')
            time.sleep(1)
            return -80  # 惩罚

        if self.similar(self.image.crop((412, 140, 871, 647)), Image.open('./Source/Win.png')) is True:
            print('Win')
            time.sleep(3.5)  # 休眠时间足够长，以避免在通关之后出现的Ready!界面识别为惩罚机制
            return 200  # 通关奖励
        return 0.1

    @staticmethod
    # def similar(image_1, image_2):
    #     lh, rh = image_1.histogram(), image_2.histogram()
    #     ret = 100 * sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    #     return ret
    def similar(img1, img2):
        # 计算图片的局部哈希值
        img1 = img1.resize((8, 8), Image.ANTIALIAS).convert('L')
        avg = reduce(lambda x, y: x + y, img1.getdata()) / 64.
        hash_value1 = reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img1.getdata())), 0)

        img2 = img2.resize((8, 8), Image.ANTIALIAS).convert('L')
        avg = reduce(lambda x, y: x + y, img2.getdata()) / 64.
        hash_value2 = reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img2.getdata())), 0)

        # 计算汉明距离
        hm_distance = bin(hash_value1 ^ hash_value2).count('1')
        hm_distance1 = hm_distance
        # print(hm_distance)
        return True if (hm_distance1 <= 10) else False


if __name__ == '__main__':
    import time

    operate = Control()
    operate.getstate()

    for i in range(400):
        # 功能为多久通过getimage读取画面中状态judge一次
        time.sleep(0.5)
        operate.getimage()
        operate.judge()
