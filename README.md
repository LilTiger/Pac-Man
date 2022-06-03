# Pac-Man
This is the project of playing pac man with a machine.

The actual running animation is under the demostration folder.

The score data of the experiment is in the corresponding txt file,and the screenshots required for the operation of the experiment are also in the Source folder.

关于DFS和A*的动作如何产生的问题
寻路算法和DQN分别产生4个动作，对应不同数值（暂时没有DQN和寻路算法更好的耦合方法，因为神经网络将像素作为输入，卷积全连接后输出动作数）
按照算法流程：首先获取吃豆人和幽灵的距离，然后选择采用的算法
当距离小于一定值时，采用DQN算法逃避幽灵追击
当距离不小于一定值时， 采用DFS和A*算法进行遍历

注 模拟棋盘需要记录所有动作的输出，根据动作的输出记录（推测）到达的可能的点（输出每个动作后的位移是连续的，故可以相当精确地推测）。寻路算法可以精确在模拟棋盘中行走感知障碍。那么如何使DQN也能感知到模拟棋盘中标识的路径和障碍呢？在记录当前位置的模拟棋盘中，如果输出的动作为无效动作（遇到了障碍，代码中实现即为 动作执行后遇到#），那么在choose_action中按照从依次选择Q值从次大到最小的动作即可。多出的此操作可以实现DQN和寻路算法对模拟棋盘中可走路径和障碍的感知，而且DQN对未知路径的探索也一并记录到模拟棋盘中，寻过的路也属于寻路算法的作用。
