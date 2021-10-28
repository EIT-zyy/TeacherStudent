# coding=utf-8
from model.student import Settings, Student

import numpy as np
import tensorflow as tf


def train_student(ckpt_path):
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

    print('Begin training student...')
    settings = Settings()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # batch_size = 128
    with tf.Session(config=config) as sess:
        model = Student(settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.soft_loss)
        sess.run(tf.global_variables_initializer())
        fetch = [model.soft_loss, model.accuracy, model.y_pred, train_op]
        loss_list = list()
        batch = mnist.train.next_batch(50)
        '''
        加载老师
        '''
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        '''
        加载结束
        '''



        for i in xrange(100):
            X_batch = batch[0].reshape([-1, 28, 28, 1])  # ,dtype=tf.float32
            y_batch = batch[1]
            _batch_size = len(y_batch)
            #teacher_batch  = restore(ckpt_path, X_batch)
            '''
            老师计算
            '''
            x = graph.get_operation_by_name('Inputs/X_inputs').outputs[0]
            feed_dict_old = {x: X_batch}
            #teacher_batch = np.zeros([_batch_size, 10], dtype=int)
            p = graph.get_operation_by_name('cnn/pre_y/output').outputs[0]
            teacher_batch = sess.run([p], feed_dict=feed_dict_old)[0]
            '''
            老师计算结束
            '''
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch, model._teacher_inputs: teacher_batch,
                         model.batch_size: _batch_size, model.keep_prob: 0.5}
            loss, acc, y_pred, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss, acc)


# def restore(ckpt_path, x_test):
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(ckpt_path + '.meta')
#         saver.restore(sess, ckpt_path)
#         graph = tf.get_default_graph()
#
#         # 2. graph.get_operation_by_name获取需要feed的placeholder
#         # 注意: 这些tensor的名字需要跟模型创建的时候对应
#         x = graph.get_operation_by_name('Inputs/X_inputs').outputs[0]
#         feed_dict = {
#             x: x_test
#         }
#         # 3. tf.get_collection获取预测结果
#         # 注意: 在训练代码中，需要计算的tensor要先用tf.add_to_collection
#         # tf.add_to_collection('pred_network', self.predictions)
#         p = graph.get_operation_by_name('cnn/pre_y/output').outputs[0]
#         # 4. sess.run获得模型的预测输出
#         prediction = sess.run([p], feed_dict=feed_dict)
#         return prediction[0]
# train_student('../save_model/teacher/teacher1')


