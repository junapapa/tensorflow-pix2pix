
"""
    @ file : network.py
    @ brief
    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""
import tensorflow as tf


def lrelu(x, leak=0.2):
    """
    leaky relu
    :param x:
    :param leak:
    :return:
    """
    return tf.maximum(x, leak*x)


def discriminator(input_image, output_image, batch_size=1, disc_num_filter=64, y=None, reuse=False, training=True):
    """

    :param input_image:
    :param output_image:
    :param batch_size:
    :param disc_num_filter:
    :param y:
    :param reuse:
    :param training:
    :return:
    """

    with tf.variable_scope("Discriminator", reuse=reuse):

        # initializer
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # parameters
        kernel_size = [5, 5]

        # image : (256 x 256 x input_dim*2)
        images = tf.concat([input_image, output_image], 3)

        # conv1 : (128 x 128 x disc_num_filter * 1)
        conv1 = tf.layers.conv2d(images, disc_num_filter * 1, kernel_size, strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='d_conv1')
        conv1 = lrelu(conv1, 0.2)

        # conv2 : (64 x 64 x disc_num_filter * 2)
        conv2 = tf.layers.conv2d(conv1, disc_num_filter * 2, kernel_size, strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='d_conv2')
        conv2 = tf.layers.batch_normalization(conv2, training=training)
        conv2 = lrelu(conv2, 0.2)

        # conv3 : (32 x 32 x disc_num_filter * 4)
        conv3 = tf.layers.conv2d(conv2, disc_num_filter * 4, kernel_size, strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='d_conv3')
        conv3 = tf.layers.batch_normalization(conv3, training=training)
        conv3 = lrelu(conv3, 0.2)

        # conv4 : (32 x 32 x disc_num_filter * 8)
        conv4 = tf.layers.conv2d(conv3, disc_num_filter * 8, kernel_size, strides=(1, 1), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='d_conv4')
        conv4 = tf.layers.batch_normalization(conv4, training=training)
        conv4 = lrelu(conv4, 0.2)

        # linear model (FC)
        h = tf.reshape(conv4, [batch_size, -1])
        w = tf.get_variable('w', [h.get_shape()[1], 1], initializer=w_init)
        b = tf.get_variable('b', [1], initializer=b_init)
        out = tf.matmul(h, w) + b

        return tf.nn.sigmoid(out), out


def generator(image, gen_num_filter=64, output_dim=3, y=None, reuse=False, training=True):
    """

    :param image:
    :param gen_num_filter:
    :param output_dim:
    :param y:
    :param reuse:
    :param training:
    :return:
    """

    with tf.variable_scope("Generator", reuse=reuse):

        # initializer
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # parameters
        kernel_size = [5, 5]
        strides = (2, 2)

        # encoder
        # image : (256 x 256 x input_dim)

        # conv1 : (128 x 128 x gen_num_filter * 1)
        conv1 = tf.layers.conv2d(image, gen_num_filter * 1, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv1')
        conv1 = lrelu(conv1, 0.2)

        # conv2 : (64 x 64 x gen_num_filter * 2)
        conv2 = tf.layers.conv2d(conv1, gen_num_filter * 2, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv2')
        conv2 = tf.layers.batch_normalization(conv2, training=training)
        conv2 = lrelu(conv2, 0.2)

        # conv3 : (32 x 32 x gen_num_filter * 4)
        conv3 = tf.layers.conv2d(conv2, gen_num_filter * 4, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv3')
        conv3 = tf.layers.batch_normalization(conv3, training=training)
        conv3 = lrelu(conv3, 0.2)

        # conv4 : (16 x 16 x gen_num_filter * 8)
        conv4 = tf.layers.conv2d(conv3, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv4')
        conv4 = tf.layers.batch_normalization(conv4, training=training)
        conv4 = lrelu(conv4, 0.2)

        # conv5 : (8 x 8 x gen_num_filter * 8)
        conv5 = tf.layers.conv2d(conv4, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv5')
        conv5 = tf.layers.batch_normalization(conv5, training=training)
        conv5 = lrelu(conv5, 0.2)

        # conv6 : (4 x 4 x gen_num_filter * 8)
        conv6 = tf.layers.conv2d(conv5, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv6')
        conv6 = tf.layers.batch_normalization(conv6, training=training)
        conv6 = lrelu(conv6, 0.2)

        # conv7 : (2 x 2 x gen_num_filter * 8)
        conv7 = tf.layers.conv2d(conv6, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv7')
        conv7 = tf.layers.batch_normalization(conv7, training=training)
        conv7 = lrelu(conv7, 0.2)

        # conv8 : (1 x 1 x gen_num_filter * 8)
        conv8 = tf.layers.conv2d(conv7, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, name='g_conv8')

        # decorder
        # conv8: (1 x 1 x gen_num_filter * 8)

        # deconv1 : (2 x 2 x gen_num_filter * 8 * 2)
        deconv1 = tf.nn.relu(conv8)
        deconv1 = tf.layers.conv2d_transpose(deconv1, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv1')
        deconv1 = tf.layers.batch_normalization(deconv1, training=training)
        deconv1 = tf.nn.dropout(deconv1, keep_prob=0.5)
        deconv1 = tf.concat([deconv1, conv7], 3)

        # deconv2 : (4 x 4 x gen_num_filter * 8 * 2)
        deconv2 = tf.nn.relu(deconv1)
        deconv2 = tf.layers.conv2d_transpose(deconv2, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv2')
        deconv2 = tf.layers.batch_normalization(deconv2, training=training)
        deconv2 = tf.nn.dropout(deconv2, keep_prob=0.5)
        deconv2 = tf.concat([deconv2, conv6], 3)

        # deconv3 : (8 x 8 x gen_num_filter * 8 * 2)
        deconv3 = tf.nn.relu(deconv2)
        deconv3 = tf.layers.conv2d_transpose(deconv3, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv3')
        deconv3 = tf.layers.batch_normalization(deconv3, training=training)
        deconv3 = tf.nn.dropout(deconv3, keep_prob=0.5)
        deconv3 = tf.concat([deconv3, conv5], 3)

        # deconv4 : (16 x 16 x gen_num_filter * 8 * 2)
        deconv4 = tf.nn.relu(deconv3)
        deconv4 = tf.layers.conv2d_transpose(deconv4, gen_num_filter * 8, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv4')
        deconv4 = tf.layers.batch_normalization(deconv4, training=training)
        deconv4 = tf.concat([deconv4, conv4], 3)

        # deconv5 : (32 x 32 x gen_num_filter * 4 * 2)
        deconv5 = tf.nn.relu(deconv4)
        deconv5 = tf.layers.conv2d_transpose(deconv5, gen_num_filter * 4, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv5')
        deconv5 = tf.layers.batch_normalization(deconv5, training=training)
        deconv5 = tf.concat([deconv5, conv3], 3)

        # deconv6 : (64 x 64 x gen_num_filter * 2 * 2)
        deconv6 = tf.nn.relu(deconv5)
        deconv6 = tf.layers.conv2d_transpose(deconv6, gen_num_filter * 2, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv6')
        deconv6 = tf.layers.batch_normalization(deconv6, training=training)
        deconv6 = tf.concat([deconv6, conv2], 3)

        # deconv7 : (128 x 128 x gen_num_filter * 1 * 2)
        deconv7 = tf.nn.relu(deconv6)
        deconv7 = tf.layers.conv2d_transpose(deconv7, gen_num_filter * 1, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv7')
        deconv7 = tf.layers.batch_normalization(deconv7, training=training)
        deconv7 = tf.concat([deconv7, conv1], 3)

        # deconv8 : (256 x 256 x output_dim)
        deconv8 = tf.nn.relu(deconv7)
        deconv8 = tf.layers.conv2d_transpose(deconv8, output_dim, kernel_size, strides=strides, padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init, name='g_deconv8')

        out = tf.nn.tanh(deconv8)

        return out


def deconv2d(input_, batch_size, num_outputs, output_size, kernel_size, strides, w_init, b_init, name):
    """

    :param input_:
    :param batch_size:
    :param num_outputs:
    :param output_size:
    :param kernel_size:
    :param strides:
    :param w_init:
    :param b_init:
    :param name:
    :return:
    """

    with tf.variable_scope(name):

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], num_outputs, input_.get_shape()[-1]],
                            initializer=w_init)

        output_shape = [batch_size, output_size[0], output_size[1], num_outputs]
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, strides[0], strides[1], 1])

        biases = tf.get_variable('biases', [num_outputs], initializer=b_init)
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

