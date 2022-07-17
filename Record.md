# Record


## Idea
1. 知识蒸馏的方法去学习从单个源到多个源的扩散模式（老师，学生
2. 依靠作差，让resnet学习两个时间中间的差，风向需要做余弦处理（归一，先尝试风向不变拟合
3. 改进Injective Padding，输入一些额外的特征维度

## Reading Record


### Invertible Residual Networks

实验部分，pre-activation ResNets with 39 convolutional bottleneck blocks with 3 convolution layers each and kernel sizes of 3x3, 1x1, 3x3。
用的是ELU激活。在残差块之前用ActNorm做正则。由于无法扩大中间隐变量的维度，选择Injective Padding方法增加输入的维度（增加channel并赋值为0）

Forward
1. squeeze操作: 将(32, 32, 3)的输入变形成(16, 16, 12)，使输入变“胖”了
2. injective padding: 