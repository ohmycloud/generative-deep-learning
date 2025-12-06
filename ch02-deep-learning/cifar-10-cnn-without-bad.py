from tensorflow.keras import layers, models
from tensorflow.keras import optimizers

# A Conv2D layer applied to grayscale input images
input_layer = layers.Input(shape=(64, 64, 1))
conv_layer_1 = layers.Conv2D(
    filters = 2,
    kernel_size = (3, 3),
    strides = 1,           # 该层用于在输入上移动卷积核的步长
    padding = "same"       # 输入参数用零填充输入数据
)(input_layer)

# build a convolutional neural network model using Keras
input_layer = layers.Input(shape=(32, 32, 3))
conv_layer_1 = layers.Conv2D(
    filters = 10,          # 10 filters
    kernel_size = (4, 4),  # 4 x 4 x 3 kernel size
    strides = 2,
    padding = 'same'
)(input_layer)

conv_layer_2 = layers.Conv2D(
    filters = 20,          # 20 filters
    kernel_size = (3, 3),  # 3 x 3 x 10 kernel size
    strides = 2,
    padding = 'same'
)(conv_layer_1)
flatten_layer = layers.Flatten()(conv_layer_2)
# 表示10类别分类任务中每个类别的概率
output_layer = layers.Dense(units=10, activation = 'softmax')(flatten_layer)
model = models.Model(input_layer, output_layer)
print(model.summary())
