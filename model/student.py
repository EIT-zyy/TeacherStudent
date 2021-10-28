import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers

""" img_6_sketch_a_net
Sketch-a-Net: a Deep Neural Network that Beats Humans

"""


class Settings(object):
    def __init__(self):
        self.model_name = 'student'
        self.summary_path = '../summary/' + self.model_name + '/'
        self.ckpt_path = '../ckpt/' + self.model_name + '/'
        self.img_size = 28
        self.n_channel = 1
        self.n_class = 10
        self.l2_weight_decay = 0.0005


class Student(object):
    """
    CNN: X_inputs=[batch_size, 224, 224, 1]
    """

    def __init__(self, settings):
        self.model_name = settings.model_name
        self.img_size = settings.img_size
        self.n_channel = settings.n_channel
        self.n_class = settings.n_class
        self.l2_weight_decay = settings.l2_weight_decay
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        # placeholders
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        self.conv_weight_initializer = initializers.xavier_initializer()
        self.conv_biases_initializer = init_ops.zeros_initializer()

        self.fc_weight_initializer = init_ops.truncated_normal_initializer(0.0, 0.005)
        self.fc_biases_initializer = init_ops.constant_initializer(0.1)

        with tf.name_scope(self.model_name + 'Inputs'):
            self._X_inputs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.n_channel],
                                            name='X_inputs')
            self._y_inputs = tf.placeholder(tf.int64, [None, self.n_class], name='y_input')

            self._teacher_inputs = tf.placeholder(tf.int64, [None, self.n_class], name='teacher_input')

        # build the model
        with tf.variable_scope(self.model_name + 'cnn'):
            # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
            # self.conv1 = self.conv2d('conv1', self._X_inputs, [5, 5, 1, 32], 1, 1, 'SAME')
            # self.pool1 = self.max_pooling('pool1', self.conv1, 2, 2, 2, 2, 'SAME')
            # print(self.conv1)
            # print(self.pool1)
            #
            # # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
            # self.conv2 = self.conv2d('conv2', self.pool1, [5, 5, 32, 64], 1, 1, 'SAME')
            # self.pool2 = self.max_pooling('pool2', self.conv2, 2, 2, 2, 2, 'SAME')
            # print(self.conv2)
            # print(self.pool2)
            #
            # # 3rd Layer: Conv (w ReLu)
            # self.conv3 = self.conv2d('conv3', self.pool2, [3, 3, 128, 256], 1, 1, 'SAME')
            #
            # # 4th Layer: Conv (w ReLu)
            # self.conv4 = self.conv2d('conv4', self.conv3, [3, 3, 256, 256], 1, 1, 'SAME')
            #
            # # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            # self.conv5 = self.conv2d('conv5', self.conv4, [3, 3, 256, 256], 1, 1, 'SAME')
            # self.pool5 = self.max_pooling('pool5', self.conv5, 3, 3, 2, 2, 'SAME')
            # print(self.conv3)
            # print(self.conv4)
            # print(self.conv5)
            # print(self.pool5)
            #
            # # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            # self.fc6 = self.conv2d('fc6', self.pool5, [7, 7, 256, 512], 1, 1, 'SAME',
            #                        weights_initializer=self.fc_weight_initializer,
            #                        biases_initializer=self.fc_biases_initializer)
            # self.dropout6 = self.dropout(self.fc6, self.keep_prob)
            #
            # # 7th Layer: FC (w ReLu) -> Dropout
            # self.fc7 = self.conv2d('fc7', self.dropout6, [1, 1, 512, 512], 1, 1, 'SAME',
            #                        weights_initializer=self.fc_weight_initializer,
            #                        biases_initializer=self.fc_biases_initializer)
            # self.dropout7 = self.dropout(self.fc7, self.keep_prob)

            # 8th Layer: FC and return unscaled activations
            # self.flattened = tf.squeeze(self.dropout7, [1, 2], name='fc8/squeezed')
            self.flattened = tf.reshape(self._X_inputs, [-1, self.img_size * self.img_size])
            # self.flattened = tf.reshape(self.imput, [-1, 7 * 7 * 64], name='pool2/squeezed')
            self.fc1 = self.fc('fc1', self.flattened, self.img_size * self.img_size, 1024,
                               biases_initializer=init_ops.zeros_initializer())
            self._y_pred = self.fc('pre_y', self.fc1, 1024, self.n_class,
                                   biases_initializer=init_ops.zeros_initializer())

        with tf.name_scope(self.model_name + 'loss'):
            # self._y_pred = tf.Print(self._y_pred, ['self._y_pred', self._y_pred, 'self._y_inputs', self._y_inputs])
            with tf.name_scope('softmax_loss'):
                self._softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self._y_pred, labels=self._y_inputs))
            with tf.name_scope('softmax_loss'):
                self._softmax_loss_teacher = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self._y_pred, labels=self._teacher_inputs))

            with tf.name_scope('l2_loss'):
                self._l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])

            with tf.name_scope('total_loss'):
                self._total_loss = self._softmax_loss + self._l2_loss * self.l2_weight_decay + self._softmax_loss_teacher

        with tf.name_scope(self.model_name + 'accuracy'):
            self._correct_prediction = tf.equal(tf.argmax(self._y_pred, 1), tf.argmax(self._y_inputs, 1))
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, "float"))

        self.saver = tf.train.Saver(max_to_keep=30)

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X_inputs(self):
        return self._X_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def soft_loss(self):
        return self._softmax_loss

    @property
    def l2_loss(self):
        return self._l2_loss

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @property
    def accuracy(self):
        return self._accuracy

    def conv2d(self, scope, x, filter_shape, strides_x, strides_y, padding, weights_initializer=None,
               biases_initializer=None):
        """
        Args:
            scope: scope name of this layer.
            x: 4-D inputs. [batch_size, in_height, in_width, in_channels]
            filter_shape: A list of ints.[filter_height, filter_width, in_channels, out_channels]
            strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input.
            padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        Returns:
            h_conv:  A 4-D tensor. [batch_size, out_height, out_width, out_channels].
            if padding is 'SAME', then out_height==in_height.
            else, out_height = in_height - filter_height + 1.
            the same for out_width.
        """
        assert padding in ['SAME', 'VALID']
        if weights_initializer is None:
            weights_initializer = self.conv_weight_initializer
        if biases_initializer is None:
            biases_initializer = self.conv_biases_initializer
        strides = [1, strides_x, strides_y, 1]
        with tf.variable_scope(scope):
            W_conv = tf.get_variable('weights', shape=filter_shape, initializer=weights_initializer)
            b_conv = tf.get_variable('biases', shape=[filter_shape[-1]], initializer=biases_initializer)
            h_conv = tf.nn.conv2d(x, W_conv, strides=strides, padding=padding)
            h_conv_relu = tf.nn.relu(h_conv + b_conv)
        return h_conv_relu

    @staticmethod
    def max_pooling(scope, x, k_height, k_width, strides_x, strides_y, padding='SAME'):
        """max pooling layer."""
        with tf.variable_scope(scope):
            ksize = [1, k_height, k_width, 1]
            strides = [1, strides_x, strides_y, 1]
            h_pool = tf.nn.max_pool(x, ksize, strides, padding)
        return h_pool

    @staticmethod
    def dropout(x, keep_prob, name=None):
        """dropout layer"""
        return tf.nn.dropout(x, keep_prob, name=name)

    def fc(self, name, x, in_size, out_size, weights_initializer=None, biases_initializer=None, activation=None):
        """fully-connect
        Args:
            x: 2-D tensor, [batch_size, in_size]
            in_size: the size of input tensor.
            out_size: the size of output tensor.
            activation: 'relu' or 'sigmoid' or 'tanh'.
        Returns:
            h_fc: 2-D tensor, [batch_size, out_size].
        """
        if activation is not None:
            assert activation in ['relu', 'sigmoid', 'tanh'], 'Wrong activation function.'
        if weights_initializer is None:
            weights_initializer = self.fc_weight_initializer
        if biases_initializer is None:
            biases_initializer = self.fc_biases_initializer
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[in_size, out_size], initializer=weights_initializer)
            b = tf.get_variable('biases', [out_size], initializer=biases_initializer)
            h_fc = tf.nn.xw_plus_b(x, w, b)
            if activation == 'relu':
                return tf.nn.relu(h_fc)
            elif activation == 'tanh':
                return tf.nn.tanh(h_fc)
            elif activation == 'sigmoid':
                return tf.nn.sigmoid(h_fc)
            else:
                return h_fc


# test the model
def test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = Student(settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.soft_loss)
        sess.run(tf.global_variables_initializer())
        fetch = [model.soft_loss, model.accuracy, model.y_pred, train_op]
        loss_list = list()
        for i in xrange(100):
            X_batch = np.zeros((batch_size, 28, 28, 1), dtype=float)
            y_batch = np.zeros(batch_size, dtype=int)
            teacher_batch = np.zeros(batch_size, dtype=int)
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model._teacher_inputs: teacher_batch, model.keep_prob: 0.5}
            loss, acc, y_pred, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss, acc)


if __name__ == '__main__':
    test()
