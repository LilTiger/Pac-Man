# Pac-Man
This is the project of playing pac man with a machine.

The actual running animation is under the demostration folder.

The score data of the experiment is in the corresponding txt file,and the screenshots required for the operation of the experiment are also in the Source folder.

关于DFS和A*的动作如何产生的问题
注意 只有DQN的神经网络可以输出动作值
故原始DQN中输出4个动作，在DFS和A*的结合中输出8个动作（原始的DQN4个和新加入的4个，输出值不同可一一对应，与代码一致）
当距离小于一定值时，采用DQN算法逃避幽灵追击
当距离不小于一定值时， 采用DFS和A*算法进行遍历
