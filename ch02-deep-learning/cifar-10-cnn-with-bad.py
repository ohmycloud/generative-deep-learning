from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import datasets, utils

# 加载 CIFAR-10 数据集
# x_train 和 x_test 分别是形状为 [50000, 32, 32, 3] 和 [10000, 32, 32, 3] 的 numpy 数组
# 第一个维度 50000 是数据集中图像的索引, 第二和第三维度与图像尺寸相关, 最后一个是通道(即红色、绿色或蓝色, 因为这些是RGB图像)
# y_train 和 y_test 分别是形状为 [50000, 1] 和 [10000, 1] 的 numpy 数组
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

# 缩放每个图像, 使像素通道值介于 0 到 1 之间
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 对标签进行 one-hot 编码, y_train 和 y_test 分别是形状为 [50000, 10] 和 [10000, 10] 的 numpy 数组
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

input_layer = layers.Input((32, 32, 3))

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate = 0.5)(x)

output_layer = layers.Dense(10, activation = 'softmax')(x)
model = models.Model(inputs = input_layer, outputs = output_layer)
print(model.summary())


# 定义优化器和损失函数以编译模型
# 优化器是用于根据损失函数的梯度更新神经网络中权重的算法
opt = optimizers.Adam(learning_rate=0.0005)
# 损失函数被神经网络用来将其预测输出与真实标签进行比较,
# 它为每个观测值返回一个数字；这个数字越大，网络在该观测值上的表现就越差。
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,          # 原始图像数据
          y_train,          # one-hot 编码的类别标签
          batch_size = 32,  # batch_size 决定在每个训练步骤中向网络传递多少观测值
          epochs = 20,      # epochs 决定网络将训练多少次
          shuffle = True    # 在每个训练步骤中将从训练数据中随机无放回地抽取批次
)

# 评估模型在测试集(从未见过的数据)上的性能
model.evaluate(x_test, y_test, batch_size = 32)
