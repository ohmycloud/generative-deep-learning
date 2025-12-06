from random import shuffle
import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

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

# 使用 Sequential 模型构建我们的多层感知机
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
print(model.summary())

# 使用函数式 API 构建我们的多层感知机
# Input 层是网络的入口点, 以元组形式告知网络每个数据元素的预期形状
# 请注意，我们未指定批次大小；这并非必要，因为我们可以同时将任意数量的
# 图像传递到 Input 层。我们不需要在Input 层定义中显式声明批次大小。
input_layer = layers.Input(shape=(32, 32, 3))
# 使用 Flatten 层将输入层展平为向量, 这将生成一个长度为 3072 的向量 (32 x 32 x 3)
# 我们这样做的原因是后续的Dense 层要求其输入是展平的，而非多维数组。
x = layers.Flatten()(input_layer)
# Dense 层包含特定数量的单元，这些单元全连接到前一层。
# 将输入通过两个 Dense 层，第一层有 200 个单元，第二层有 150 个单元，两者都使用 ReLU 激活函数
x = layers.Dense(units=200, activation='relu')(x)
x = layers.Dense(units=150, activation='relu')(x)
output_layer = layers.Dense(units=10, activation='softmax')(x)
model = models.Model(input_layer, output_layer)
# 检查网络每一层的形状
print(model.summary())

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size = 32,
          epochs = 20,
          shuffle = True
)

model.evaluate(x_test, y_test)

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.imshow(img)

# A Conv2D layer applied to grayscale input images
input_layer = layers.Input(shape=(64, 64, 1))
conv_layer_1 = layers.Conv2D(
    filters = 2,
    kernel_size = (3, 3),
    strides = 1,
    padding = "same"
)(input_layer)

# build a convolutional neural network model using Keras
input_layer = layers.Input(shape=(32, 32, 3))
conv_layer_1 = layers.Conv2D(
    filters = 10,
    kernel_size = (4, 4),
    strides = 2,
    padding = 'same'
)(input_layer)

conv_layer_2 = layers.Conv2D(
    filters = 20,
    kernel_size = (3, 3),
    strides = 2,
    padding = 'same'
)(conv_layer_1)
flatten_layer = layers.Flatten()(conv_layer_2)
output_layer = layers.Dense(units=10, activation = 'softmax')(flatten_layer)
model = models.Model(input_layer, output_layer)
