import numpy as np

# 加载 .npy 文件
data = np.load('np_softmax.npy')
print(data.shape)

# 对第一个维度（维度0）进行求和
sum_along_dimension_1 = data.sum(axis=1)  # 对第一个维度（维度1）进行求和

# 打印结果
print(sum_along_dimension_1.shape)  # 打印形状
print(sum_along_dimension_1)  # 打印结果
