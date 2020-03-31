
import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import pytesseract
from functools import reduce
import math
import time
#####################################
# DFS遍历开始
tm = [
    '############################',
    '#E...........##...........E#',
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
    '#...##........S.......##...#',
    '###.##.##.########.##.##.###',
    '###.##.##.########.##.##.###',
    '#......##....##....##......#',
    '#.##########.##.##########.#',
    '#.##########.#..##########.#',
    '#E........................E#',
    '############################'
]

# 因为python里string不能直接改变某一元素，所以用test_map来存储搜索时的地图
test_map = []


class Node_Elem:
    """
    开放列表和关闭列表的元素类型，parent用来在成功的时候回溯路径
    """
    def __init__(self, parent, x, y):
        self.parent = parent
        self.x = x
        self.y = y


class DFS:
    # 注意w,h两个参数，如果你修改了地图，需要传入一个正确值或者修改这里的默认参数
    # s代表起点
    def __init__(self, s_x, s_y, w=27, h=31):
        self.s_x = s_x
        self.s_y = s_y

        self.width = w
        self.height = h

        self.stack = []
        self.visited = []
        self.path = []
    ###################
    # COPY START
        self.box_view = (410, 82, 872, 681)  # 神经网络的视觉区域
        self.box_reward = (417, 112, 537, 135)  # 分数区域
        self.box = (69, 0, 1355, 749)  # 完整截图区域
        self.image = None  # 未初始化的截图
        self.image_view = None  # 未初始化的神经网络视野
        self.image_reward = None  # 未初始化的分数区域
        self.state = []  # 初始化状态

    def getimage(self):
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_view = self.image.crop(self.box_view)
        self.image_reward = self.image.crop(self.box_reward)

    def getstate(self):
        self.state = []
        for _ in range(4):
            time.sleep(0.04)
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
            f = open('best_score_dfs.txt', 'a')
            f.write('best_score:' + best_score + '\n')
            f.close()
            # 重新训练，开始游戏
            time.sleep(1)
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
            time.sleep(3.5)  # 休眠时间足够长，以避免在通关之后出现的Ready!界面识别为惩罚机制
            return 200  # 通关奖励
        # judge=0.1 正常状态
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

    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'

    def node_in_visited(self, node):
        for i in self.visited:
            if node.x == i.x and node.y == i.y:
                return True
        return False

    def node_in_stack(self, node):
        for i, n in enumerate(self.stack):
            if node.x == n.x and node.y == n.y:
                return i
        return -1

    def get_searched(self):
        l = []
        for i in self.stack:
            l.append((i.x, i.y))
        for i in self.visited:
            l.append((i.x, i.y))
        return l

    # 查找路径的入口函数
    def find_path(self):
        # 构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y)
        while True:
            # 查找未访问节点
            self.extend_round(p)
            # stack中节点即为未访问节点
            # stack为空，遍历完成，返回路径
            if not self.stack or abs(self.judge() > 1):
                self.make_path(p)
                return
            idx, p = self.get_direction()
            # 把此节点压入visited，并从stack里删除
            self.visited.append(p)
            del self.stack[idx]

    def make_path(self, p):
        # 从结束点回溯到开始点，开始点的parent == None
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    def get_direction(self):
        direction_node = None
        index = -1
        for i, w in enumerate(self.stack):
            if w not in self.visited:
                direction_node = w
                index = i
        return index, direction_node

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
            node = Node_Elem(p, new_x, new_y)
            # 新节点在visited，则忽略
            if self.node_in_visited(node):
                continue
            i = self.node_in_stack(node)
            if i != -1:
                # stack不为空
                if self.stack:
                    for i, w in enumerate(self.stack):
                        w = self.stack.pop()
                        if w not in self.visited:
                            self.stack.append(w)
                            self.visited.append(w)
                            self.stack[i].parent = p
                continue
            self.judge()
            if x == 0 and y == -1:
                    print("Go up")
                    pyautogui.press("up")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
            if x == 1 and y == 0:
                    print("Turn right")
                    pyautogui.press("right")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
            if x == 0 and y == 1:
                    print("Go down")
                    pyautogui.press("down")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
            if x == -1 and y == 0:
                    print("Turn Left")
                    pyautogui.press("left")
                    self.getstate()  # 下一个状态
                    score = self.getscore()
                    reward = int(score - pre_score)
                    print(f'score:{score},reward:{reward}')
            self.stack.append(node)


def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))


def print_test_map():
    """
    打印搜索后的地图
    """
    for line in test_map:
        print(''.join(line))


def mark_start(s_x, s_y):
    test_map[s_y][s_x] = 'S'


def mark_path(l):
    mark_symbol(l, '*')


def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s


def get_start_XY():
    return get_symbol_XY('S')


def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except:
            continue
        else:
            break
    return x, y

# DFS遍历结束
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
        self.image_reward = None               # 未初始化的分数区域
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
            time.sleep(0.04)
            self.getimage()
            self.state.append(self.image_view)
        self.state = np.stack(self.state, axis=0)

    def action(self, action):
        pre_score = self.getscore()
        self.judge()
        if pre_score is not None:
            def find_path():
                s_x, s_y = get_start_XY()
                dfs = DFS(s_x, s_y)
                dfs.find_path()
                # 标记开始点
                mark_start(s_x, s_y)
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
                self.getstate()  # 下一个状态
                score = self.getscore()
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
            elif action == 3:
                pyautogui.press("left")
                print('Turn Left')
                self.getstate()  # 下一个状态
                score = self.getscore()
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
            elif action == 4:
                pyautogui.press("down")
                print('Go down')
                self.getstate()  # 下一个状态
                score = self.getscore()
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
            elif action == 5:
                pyautogui.press("right")
                print('Turn right')
                self.getstate()  # 下一个状态
                score = self.getscore()
                reward = int(score - pre_score)
                print(f'score:{score},reward:{reward}')
                return self.state, reward, 0
        else:
            # 依次返回state_next,reward,terminal 当terminal为1时结束程序
            return None, None, 1

    # Control类中获得分数、判断状态与图片相似性
    def getscore(self):
        # 存储分数区域图形
        self.image_reward.save('./Source/Scores/score.png')
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
        # crop中后接具体坐标时有两个括号:)
        if self.similar(self.image.crop((560, 412, 730, 438)), Image.open('./Source/GameOver.png')) is True:
            print('Game Over')
            print(f'Final score is:{score}')
            # 将最大分数值写入文件
            best_score = str(self.getscore())
            f = open('best_score_dfs.txt', 'a')
            f.write('best_score:' + best_score + '\n')
            f.close()
            # 重新训练，开始游戏
            time.sleep(1)
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
            time.sleep(3.5)  # 休眠时间足够长，以避免在通关之后出现的Ready!界面识别为惩罚机制
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
