from tensorflow.keras import datasets
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# 定义编码器的Input 层（图像）
encoder_input = layers.Input(shape=(32, 32, 1), name = "encoder_input")
# 将Conv2D 层顺序堆叠在一起
x = layers.Conv2D(32, (3, 3), strides = 2, activation = 'relu', padding = 'same')(encoder_input)
x = layers.Conv2D(64, (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)
x = layers.Conv2D(128, (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)

# 通过切片操作，跳过第一个维度（批次维度），只保留后面的维度
shape_before_flattening = x.shape[1:]

# 将最后一个卷积层展平为向量
x = layers.Flatten()(x)
# 通过 Dense 层将该向量连接到二维嵌入
encoder_output = layers.Dense(2, name = "encoder_output")(x)
# 定义编码器的 Keras Model — 该模型接收输入图像并将其编码为二维嵌入
encoder = models.Model(encoder_input, encoder_output)

print(encoder.summary())

# 定义解码器的 Input 层（嵌入层）
decoder_input = layers.Input(shape=(2,), name = "decoder_input")
# 将输入连接到 Dense 层
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
# Reshape 将该向量转换为可馈入第一个 Conv2DTranspose 层的张量
x = layers.Reshape(shape_before_flattening)(x)

# 将 Conv2DTranspose 层堆叠在一起
x = layers.Conv2DTranspose(128, (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)
x = layers.Conv2DTranspose(64, (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)
x = layers.Conv2DTranspose(32, (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)

decoder_output = layers.Conv2D(
    1,
    (3, 3),
    strides = 1,
    activation = 'sigmoid',
    padding = 'same',
    name = "decoder_output"
)(x)

# Keras Model 定义了解码器, 一个接收潜在空间嵌入并将其解码回原始图像域的模型
decoder = models.Model(decoder_input, decoder_output)

print(decoder.summary())

# 定义完整自编码器的 Keras Model ——该模型接收一幅图像,
# 通过编码器处理后再经解码器输出，以生成原始图像的重建结果
autoencoder = models.Model(encoder_input, decoder(encoder_output))
# 编译自编码器
autoencoder.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy')

# 将输入图像同时作为输入和输出来训练自编码器
autoencoder.fit(
    x_train,
    x_train,
    epochs = 5,
    batch_size = 128,
    shuffle=True,
    validation_data = (x_test, x_test)
)

# 使用自编码器重建图像
example_images = x_test[:500]
predictions = autoencoder.predict(example_images)

# 使用编码器嵌入图像
embeddings = encoder.predict(example_images)

plt.figure(figsize = (8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.show()

# 使用解码器生成新图像
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)
