# coding=utf-8
from model.teacher import Settings,Teacher

import numpy as np
import tensorflow as tf

def train_tracher(ckpt):
    # 设置按需使用GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    # 用tensorflow 导入数据
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 看看咱们样本的数量
    print mnist.test.labels.shape
    print mnist.train.labels.shape


    print('Begin training teacher...')
    settings = Settings()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = Teacher(settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.soft_loss)
        sess.run(tf.global_variables_initializer())
        fetch = [model.soft_loss, model.accuracy, model.y_pred, train_op]
        loss_list = list()
        batch = mnist.train.next_batch(50)
        saver = tf.train.Saver()
        for i in xrange(50):
            X_batch = batch[0].reshape([-1,28,28,1])#,dtype=tf.float32
            y_batch = batch[1]
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.keep_prob: 0.5}
            loss, acc, y_pred, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss, acc)
        saver.save(sess, ckpt)


