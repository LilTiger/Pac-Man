import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import pytesseract
from functools import reduce
import math
import time
#####################################
# A*寻路开始
tm = [
    '############################',
    '#............##...........E#',
    '#.####.#####.##.#####.####.#',
    '#.####.#####.##.#####.####.#',
    '#.####.#####.##.#####.####.#',
    '#..........................#',
    '#.####.##.########.##.####.#',
    '#.####.##.########.##.####.#',
    '#......##....##....##......#',
    '######.#####.##.#####.######',
    '######.#####.##.#####.######',
    '######.##..........##.######',
    '######.##.########.##.######',
    '######.##.########.##.######',
    '..........########..........',
    '######.##.########.##.######',
    '######.##.########.##.######',
    '######.##..........##.######',
    '######.##.########.##.######',
    '######.##.########.##.######',
    '#............##............#',
    '#.####.#####.##.#####.####.#',
    '#.####.#####.##.#####.####.#',
    '#...##....... S.......##...#',
    '###.##.##.########.##.##.###',
    '###.##.##.########.##.##.###',
    '#......##....##....##......#',
    '#.##########.##.##########.#',
    '#.##########.#..##########.#',
    '#..........................#',
    '############################']

# 因为python里string不能直接改变某一元素，所以用test_map来存储搜索时的地图
test_map = []


class Node_Elem:
    """
    开放列表和关闭列表的元素类型，parent用来在成功的时候回溯路径
    """
    def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist


class A_Star:
    """
    A星算法实现类
    """
    # 注意w,h两个参数，如果你修改了地图，需要传入一个正确值或者修改这里的默认参数
    # s代表起点，e代表终点
    def __init__(self, s_x, s_y, e_x, e_y, w=27, h=31):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y

        self.width = w
        self.height = h

        self.open = []
        self.close = []
        self.path = []
    ###################
    # COPY START
        self.box_view = (410, 82, 872, 681)  # 神经网络的视觉区域
        self.box_reward = (417, 112, 537, 135)  # 分数区域
        self.box = (69, 0, 1355, 749)  # 完整截图区域
        self.image = None  # 未初始化的截图
        self.image_view = None  # 未初始化的神经网络视野
        self.image_reward = None  # 未初始化的分数板区域
        self.state = []  # 初始化状态

    def getimage(self):
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_view = self.image.crop(self.box_view)
        self.image_reward = self.image.crop(self.box_reward)

    def getstate(self):
        self.state = []
        for _ in range(4):
            time.sleep(0.01)
            self.getimage()
            self.state.append(self.image_view)
        self.state = np.stack(self.state, axis=0)

    def get_prescore(self):
        # 存储分数区域图形
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_reward = self.image.crop(self.box_reward)
        self.image_reward.save('./Source/Scores/score_pre.png')
        # 识别分数为字符串
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score_pre.png"), lang="pac")
        return int(score)

    def getscore(self):
        # 存储分数区域图形
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_reward = self.image.crop(self.box_reward)
        self.image_reward.save('./Source/Scores/score.png')
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score.png"), lang="pac")
        return int(score)

    def judge(self):
        score = self.getscore()
        if self.similar(self.image.crop((560, 412, 730, 438)), Image.open('./Source/GameOver.png')) is True:
            print('Game Over')
            print(f'Final score is:{score}')
            # 将最大分数值写入文件
            best_score = str(self.getscore())
            f = open('best_score_astar.txt', 'a')
            f.write('best_score:' + best_score + '\n')
            f.close()
            # 重新训练，开始游戏
            time.sleep(0.5)
            pyautogui.press('enter')
            time.sleep(4)
            return -200  # 惩罚
        if self.similar(self.image.crop((563, 314, 727, 439)), Image.open('./Source/Try_again.png')) is True:
            print('Eaten by ghost')
            print(f'score:{score},reward:-80')
            time.sleep(1.5)
            return -80  # 惩罚
        if self.similar(self.image.crop((412, 140, 871, 647)), Image.open('./Source/Win.png')) is True:
            print('Win')
            time.sleep(5)  # 休眠时间足够长，以避免在通关之后出现的Ready!界面识别为惩罚机制
            return 200  # 通关奖励
        return 0.1

    @staticmethod
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
    # COPY DONE
    #########################

    # 查找路径的入口函数
    def find_path(self):
        # 构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        while True:
            # 扩展F值最小的节点
            self.extend_round(p)
            # 如果开放列表为空，则不存在路径，返回
            if not self.open:
                return
            # 获取F值最小的节点
            idx, p = self.get_best()
            # 找到路径，生成路径，返回
            if self.is_target(p):
                self.make_path(p)
                return

            # 把此节点压入关闭列表，并从开放列表里删除
            self.close.append(p)
            del self.open[idx]

    def make_path(self, p):
        # 从结束点回溯到开始点，开始点的parent == None
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y

    def get_best(self):
        best = None
        bv = 1000000  # 如果你修改的地图很大，可能需要修改这个值
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)  # 获取F值
            if value < bv:  # 比以前的更好，即F值更小
                best = i
                bv = value
                bi = idx
        return bi, best

    def get_dist(self, i):
        # F = G + H
        # G 为已经走过的路径长度， H为估计还要走多远
        # 这个公式就是A*算法的精华了
        # 取两点间曼哈顿距离作为距离估计
        return i.dist + math.sqrt(
            (self.e_x - i.x) * (self.e_x - i.x)
            + (self.e_y - i.y) * (self.e_y - i.y)) * 1.2

    def extend_round(self, p):
        pre_score = self.get_prescore()
        # 只能走上下左右四个方向
        xs = (0, -1, 1, 0)
        ys = (-1, 0, 0, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            # 无效或者不可行走区域，则忽略
            if not self.is_valid_coord(new_x, new_y):
                continue
            # 构造新的节点
            node = Node_Elem(p, new_x, new_y, p.dist + self.get_cost(
                p.x, p.y, new_x, new_y))
            # 新节点在关闭列表，则忽略
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                # 新节点在开放列表
                if self.open[i].dist > node.dist:
                    # 现在的路径到比以前到这个节点的路径更好~
                    # 则使用现在的路径
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.judge()
            if x == 0 and y == -1:
                if self.judge is not -200:
                    print("Go up")
                    pyautogui.press("up")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
                else:
                    self.open.clear()
                    self.close.clear()
            if x == 1 and y == 0:
                if self.judge is not -200:
                    print("Turn right")
                    pyautogui.press("right")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
                else:
                    self.open.clear()
                    self.close.clear()
            if x == 0 and y == 1:
                if self.judge is not -200:
                    print("Go down")
                    pyautogui.press("down")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
                else:
                    self.open.clear()
                    self.close.clear()
            if x == -1 and y == 0:
                if self.judge is not -200:
                    print("Turn Left")
                    pyautogui.press("left")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
                else:
                    self.open.clear()
                    self.close.clear()
            self.open.append(node)

    def get_cost(self, x1, y1, x2, y2):
        """
         上下左右直走，代价为1.0，斜走，代价为1.4
         """
        if x1 == x2 or y1 == y2:
            return 1.0
        return 1.4

    def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False

    def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1

    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'

    def get_searched(self):
        l = []
        for i in self.open:
            l.append((i.x, i.y))
        for i in self.close:
            l.append((i.x, i.y))
        return l


def print_test_map():
    """
    打印搜索后的地图
    """
    for line in test_map:
        print(''.join(line))


def get_start_XY():
    return get_symbol_XY('S')


def get_end_XY():
    return get_symbol_XY('E')


def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except:
            continue
        else:
            break
    return x, y


def mark_path(l):
    mark_symbol(l, '*')


def mark_searched(l):
    mark_symbol(l, ' ')


def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s


def mark_start_end(s_x, s_y, e_x, e_y):
    test_map[s_y][s_x] = 'S'
    test_map[e_y][e_x] = 'E'


def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))
# A*寻路结束
#################################


class Control:
    def __init__(self):
        # box = (left, top, left+width, top+height)
        self.debug = True
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
        for _ in range(4):
            time.sleep(0.01)
            self.getimage()
            self.state.append(self.image_view)
        self.state = np.stack(self.state, axis=0)

    def action(self, action):
        pre_score = self.get_prescore()
        self.judge()
        if pre_score is not None:
            def find_path():
                s_x, s_y = get_start_XY()
                e_x, e_y = get_end_XY()
                a_star = A_Star(s_x, s_y, e_x, e_y)
                a_star.find_path()
                # 标记开始、结束点
                mark_start_end(s_x, s_y, e_x, e_y)
            if action is 0 or 1 or 6 or 7:
                tm_to_test_map()
                find_path()
                # 新加
                self.getstate()  # 下一个状态
                score = self.getscore()
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
            elif action == 2:
                pyautogui.press("up")
                print('Go up')
                self.getstate()  # 先获取下一个状态
                score = self.getscore()  # 再获取当前分数
                # 奖励值等于前一个状态的分数减去后一个状态的分数，即吃豆便获得奖励
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
            elif action == 3:
                pyautogui.press("left")
                print('Turn Left')
                self.getstate()  # 先获取下一个状态
                score = self.getscore()  # 再获取当前分数
                # 奖励值等于前一个状态的分数减去后一个状态的分数，即吃豆便获得奖励
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
            elif action == 4:
                pyautogui.press("down")
                print('Go down')
                self.getstate()  # 先获取下一个状态
                score = self.getscore()  # 再获取当前分数
                # 奖励值等于前一个状态的分数减去后一个状态的分数，即吃豆便获得奖励
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
            elif action == 5:
                pyautogui.press("right")
                print('Turn right')
                self.getstate()  # 先获取下一个状态
                score = self.getscore()  # 再获取当前分数
                # 奖励值等于前一个状态的分数减去后一个状态的分数，即吃豆便获得奖励
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')

        else:
            return None, None, 1

    def get_prescore(self):
        # 存储分数区域图形
        self.image_reward.save('./Source/Scores/score_pre.png')
        # 识别分数为字符串
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score_pre.png"), lang="pac")
        return int(score)

    def getscore(self):
        # 存储分数区域图形
        self.image_reward.save('./Source/Scores/score.png')
        # 识别分数为字符串
        score = pytesseract.image_to_string(Image.open("./Source/Scores/score.png"), lang="pac")
        return int(score)

    def judge(self):
        score = self.getscore()
        # crop中后接具体坐标时有两个括号:)
        if self.similar(self.image.crop((560, 412, 730, 438)), Image.open('./Source/GameOver.png')) is True:
            print('Game Over')
            print(f'Final score is:{score}')
            # 将最大分数值写入文件
            best_score = str(self.getscore())
            f = open('best_score_astar.txt', 'a')
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
            time.sleep(1.5)
            return -80  # 惩罚

        if self.similar(self.image.crop((412, 140, 871, 647)), Image.open('./Source/Win.png')) is True:
            print('Win')
            time.sleep(5)  # 休眠时间足够长，以避免在通关之后出现的Ready!界面识别为惩罚机制
            return 200  # 通关奖励
        return 0.1

    @staticmethod
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
        time.sleep(0.5)
        operate.getimage()
        operate.judge()
