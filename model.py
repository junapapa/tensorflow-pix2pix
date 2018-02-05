"""
    @ file : model.py
    @ brief
    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""
from network import *
from trainer import *


class Pix2Pix(object):
    """

    """
    def __init__(self, sess, gan_name,
                 dataset_name='facades',
                 input_size=256, input_dim=3, output_size=256, output_dim=3,
                 batch_size=1,
                 gen_num_filter=64, disc_num_filter=64,
                 learning_rate=0.0002, beta1=0.5, l1_lambda=100.0,
                 checkpoint_dir=None, sample_dir=None, test_dir=None):
        """

        :param sess:
        :param gan_name:
        :param dataset_name:
        :param input_size:
        :param input_dim:
        :param output_size:
        :param output_dim:
        :param batch_size:
        :param gen_num_filter:
        :param disc_num_filter:
        :param learning_rate:
        :param beta1:
        :param l1_lambda:
        :param checkpoint_dir:
        :param sample_dir:
        :param test_dir:
        """
        self.sess = sess
        self.gan_name = gan_name

        self.dataset_name = dataset_name

        self.input_size = input_size
        self.input_dim = input_dim
        self.output_size = output_size
        self.output_dim = output_dim

        self.batch_size = batch_size

        self.gen_num_filter = gen_num_filter
        self.disc_num_filter = disc_num_filter

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.l1_lambda = l1_lambda

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.test_dir = test_dir

        """
        Build Graph
        """
        # input / output
        self.real_input = tf.placeholder(tf.float32,
                                         [self.batch_size, self.input_size, self.input_size, self.input_dim],
                                         name="real_input_images")
        self.real_output = tf.placeholder(tf.float32,
                                          [self.batch_size, self.input_size, self.input_size, self.input_dim],
                                          name="real_output_images")

        # generator
        self.fake_output = generator(self.real_input, self.gen_num_filter, self.output_dim)

        # discriminator
        self.D_real, self.D_logits_real = discriminator(self.real_input, self.real_output, self.batch_size,
                                                        self.disc_num_filter, reuse=False, training=True)
        self.D_fake, self.D_logits_fake = discriminator(self.real_input, self.fake_output, self.batch_size,
                                                        self.disc_num_filter, reuse=True, training=True)

        # sampler
        self.fake_output_sample = generator(self.real_input, self.gen_num_filter, self.output_dim,
                                            reuse=True, training=True)

        # summary (tensorboard)
        self.D_real_summary = tf.summary.histogram("D_real", self.D_real)
        self.D_fake_summary = tf.summary.histogram("D_fake", self.D_fake)
        self.fake_output_summary = tf.summary.image("fake_output", self.fake_output)

        # Loss
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real, labels=tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake

        self.G_loss_gan = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.ones_like(self.D_fake)))
        self.G_loss_L1 = tf.reduce_mean(tf.abs(self.real_output - self.fake_output))
        self.G_loss = self.G_loss_gan + self.l1_lambda * self.G_loss_L1

        # summary (tensorboard)
        self.D_loss_real_summary = tf.summary.scalar("D_loss_real", self.D_loss_real)
        self.D_loss_fake_summary = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        self.D_loss_summary = tf.summary.scalar("D_loss", self.D_loss)
        self.G_loss_summary = tf.summary.scalar("G_loss", self.G_loss)

        # trainable parameters
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

        # optimizer
        self.opt_d = optimizer(loss=self.D_loss, var_list=self.d_vars, learning_rate=learning_rate, beta1=beta1)
        self.opt_g = optimizer(loss=self.G_loss, var_list=self.g_vars, learning_rate=learning_rate, beta1=beta1)

        # summary (merge)
        self.G_summary = tf.summary.merge([self.D_fake_summary, self.fake_output_summary,
                                           self.D_loss_real_summary, self.G_loss_summary])
        self.D_summary = tf.summary.merge([self.D_real_summary,
                                           self.D_loss_real_summary, self.D_loss_summary])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # saver
        self.saver = tf.train.Saver()

    def update_discriminator(self, input_image, output_image):
        """
        구별자 학습 (gradient descent 1 step)
        :param input_image:
        :param output_image:
        :return:
        """
        summary_str, _ = self.sess.run([self.D_summary, self.opt_d],
                                       feed_dict={self.real_input: input_image, self.real_output: output_image})
        return summary_str

    def update_generator(self, input_image, output_image):
        """
        생성자 학습 (gradient descent 1 step)
        :param input_image:
        :param output_image:
        :return:
        """
        summary_str, _ = self.sess.run([self.G_summary, self.opt_g],
                                       feed_dict={self.real_input: input_image, self.real_output: output_image})
        return summary_str
