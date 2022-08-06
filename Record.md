# Record


## Idea
1. 知识蒸馏的方法去学习从单个源到多个源的扩散模式（老师，学生
2. 依靠作差，让resnet学习两个时间中间的差，风向需要做余弦处理（归一，先尝试风向不变拟合
3. 改进Injective Padding，输入一些额外的特征维度
4. 数据增强：block的数量根据source和target中间间隔的时间步变化 以 保证 每个block拟合的都是单时间步
5. padding用常量数据添加维度的同时，因为resnet的特性不会要求冗余的算力拟合(添加的维度拟合出来的是f(x)=0)
6. 将x+f(x)改造成ax+f(x)，其中a为常数，lipschiz约束变成<a，使更容易找到f(x)
7. 讨论在动力学系统中，在参数扰动下的稳定性，即加入扰动后模型能否恢复到原来的样子，f(x)+δx -> f(x)，以及输入扰动的稳定性f(x+δx) -> f(x)


## Working Record
1. 建立自己数据的dataset √
2. 测试可视化代码 √ 检验Density Estimation可视化情况
3. 用自己数据跑通分类代码
4. 用自己的数据跑Density Estimation的代码
5. revertible理论

## Reading Record

### Invertible Residual Networks

实验部分，pre-activation ResNets with 39 convolutional bottleneck blocks with 3 convolution layers each and kernel sizes of 3x3, 1x1, 3x3。
用的是ELU激活。在残差块之前用ActNorm做正则。由于无法扩大中间隐变量的维度，选择Injective Padding方法增加输入的维度（增加channel并赋值为0）

Forward
1. squeeze操作: 将(32, 32, 3)的输入变形成(16, 16, 12)，使输入变“胖”了
2. injective padding: 经过维度互换，使深度维度换到第三维即(512, 3, 32, 32)->(512, 32, 3, 32), 然后用nn.ZeroPad2D在(3, 32)的下方增加13行0，使得成为(16, 32)，再变回去即可。
3. 接下来看看stack是怎么创建的，