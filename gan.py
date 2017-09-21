import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256

# 일반적인 GAN 의 형태 입니다.
# 라벨을 구분하지 않습니다.

# 제너레이터 (G)
def build_generator(z_prior):
    # Fully Connected Layer 1 (100 (latent-vector) -> 150 (h1_size))
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)

    # Fully Connected Layer 2 (150 (h1_size) -> 300 (h2_size))
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # Fully Connected Layer 3 (300 (h1_size) -> input_height * input_width (img_size))
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    # 마지막은 활성화함수 tanh 로
    x_generate = tf.nn.tanh(h3)

    # generator 변수 저장
    g_params = [w1, b1, w2, b2, w3, b3]

    return x_generate, g_params

# 디스크리미네이터 (D)
def build_discriminator(x_data, x_generated, keep_prob):
    # 실제 이미지와 생성된 이미지를 합침
    x_in = tf.concat([x_data, x_generated], 0) 

    # Fully Connected Layer 1 (input_height * input_width (img_size) -> 200 (h2_size)) , dropout
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)

    # Fully Connected Layer 2 (200 (h1_size) -> 150 (h1_size)) , dropout
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

    # Fully Connected Layer 3 (150 (h1_size) -> 1)
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3

    # batch_size 만큼 잘라 각각 y_data, y_generated 로
    # ex)
    #   이미지 60000개, 배치 사이즈 256, 이미지 사이즈 28 * 28 이라면
    #   h3 shape : (257, 1)
    #   y_data shape : (256, 1)
    #   y_generated shape : (1, 1) 
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))

    # discriminator 변수 저장
    d_params = [w1, b1, w2, b2, w3, b3]

    return y_data, y_generated, d_params

# 결과 저장 (이미지)
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    # mnist 로 학습할 경우
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # phd08 로 학습할 경우
    phd08 = np.load('phd08/phd08_data_1.npy')
    size = phd08.shape[0]
    phd08 = phd08.reshape((size,784))

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data") # (batch_size, img_size)
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior") # (batch_size, z_size)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout 퍼센트
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # x_generated : generator 가 생성한 이미지, g_params : generater 의 TF 변수들
    x_generated, g_params = build_generator(z_prior)
    # 실제이미지, generater 가 생성한 이미지, dropout keep_prob 를 넣고 discriminator(경찰) 이 감별
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    # loss 함수 ( D 와 G 를 따로 ) *
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    # optimizer : AdamOptimizer 사용 *
    optimizer = tf.train.AdamOptimizer(0.0001)

    # discriminator 와 generator 의 변수로 각각의 loss 함수를 최소화시키도록 학습
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    for i in range(sess.run(global_step), max_epoch):
        for j in range(21870 // batch_size):
            print("epoch:%s, iter:%s" % (i, j))

            # x_value, _ = mnist.train.next_batch(batch_size)
            # x_value = 2 * x_value.astype(np.float32) - 1
            # print(x_value[0])

            batch_end = j * batch_size + batch_size
            if batch_end >= size:
                batch_end = size - 1
            x_value = phd08[ j * batch_size : batch_end ]
            x_value = x_value / 255.
            x_value = 2 * x_value - 1

            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            if j % 1 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, os.path.join(output_path, "sample%s.jpg" % i))
        print(x_gen_val)
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, os.path.join(output_path, "random_sample%s.jpg" % i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

# 학습 완료 후 테스트 (이미지로 저장)
def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, os.path.join(output_path, "test_result.jpg"))


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
