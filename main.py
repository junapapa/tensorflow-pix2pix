"""
    @ file : main.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""

import os
import tensorflow as tf
import numpy as np
import pprint
import tensorflow.contrib.slim as slim
from trainer import run_train, run_test, load_checkpoint

from model import Pix2Pix

# parameter settings
flags = tf.app.flags

# model
flags.DEFINE_string("gan_name", "pix2pix", "The name of GAN")

# train or test
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

# dataset
flags.DEFINE_string("dataset_name", "facades", "The name of dataset [facades, cityscapes, maps]")
flags.DEFINE_integer("scale_size", 286, "scale images to this size [286]")
flags.DEFINE_integer("crop_size", 256, "then crop to this size [256]")
flags.DEFINE_integer("input_dim", 3, "# of input image channels [3]")
flags.DEFINE_integer("output_dim", 3, "# of output image channels [3]")
flags.DEFINE_boolean("flip", True, "if flip the images for data argumentation [True]")

# training parameters
flags.DEFINE_integer("epoch", 200, "# of epoch [200]")
flags.DEFINE_integer("batch_size", 1, "# images in batch [1]")
flags.DEFINE_integer("train_size", np.inf, "# images used to train [np.inf]")

# optimizer
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("l1_lambda", 100.0, "weight on L1 term in objective [100.0]")

# network
flags.DEFINE_integer("gen_num_filter", 64, "# of gen filters in first conv layer [64]")
flags.DEFINE_integer("disc_num_filter", 64, "# of discri filters in first conv layer [64]")
flags.DEFINE_integer("input_size", 256, "size of input image (squared) [256]")
flags.DEFINE_integer("output_size", 256, "size of output image (squared) [256]")

# option
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the test image samples [test]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS


def main(_):
    """

    :return:
    """
    print("="*100)
    print("FLAGS")
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    # make sub-directories
    if not os.path.isdir(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.isdir(FLAGS.sample_dir):
        os.mkdir(FLAGS.sample_dir)
    if not os.path.isdir(FLAGS.test_dir):
        os.mkdir(FLAGS.test_dir)

    # Launch Graph
    sess = tf.Session()
    model = Pix2Pix(sess=sess, gan_name=FLAGS.gan_name,
                    dataset_name=FLAGS.dataset_name,
                    input_size=FLAGS.input_size, input_dim=FLAGS.input_dim,
                    output_size=FLAGS.output_size, output_dim=FLAGS.output_dim,
                    batch_size=FLAGS.batch_size,
                    gen_num_filter=FLAGS.gen_num_filter, disc_num_filter=FLAGS.disc_num_filter,
                    learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, l1_lambda=FLAGS.l1_lambda,
                    checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir, test_dir=FLAGS.test_dir)

    sess.run(tf.global_variables_initializer())

    # show all variables
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # load trained model
    flag_checkpoint, counter = load_checkpoint(model)

    dataset_dir = os.path.join("datasets", FLAGS.dataset_name)
    if FLAGS.train:
        # training dataset dir
        trainset_dir = os.path.join(dataset_dir, "train")
        valset_dir = os.path.join(dataset_dir, "val")
        run_train(model=model,
                  trainset_dir=trainset_dir,
                  valset_dir=valset_dir,
                  sample_size=FLAGS.batch_size,
                  scale_size=FLAGS.scale_size,
                  crop_size=FLAGS.crop_size,
                  flip=FLAGS.flip,
                  training_epochs=FLAGS.epoch,
                  flag_checkpoint=flag_checkpoint,
                  checkpoint_counter=counter)

    else:
        # test dir
        testset_dir = os.path.join(dataset_dir, "test")
        if not os.path.isdir(testset_dir):
            testset_dir = os.path.join(dataset_dir, "val")

        run_test(model=model, testset_dir=testset_dir)


if __name__ == '__main__':
    tf.app.run()
