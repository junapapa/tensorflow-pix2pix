"""
    @ file : trainer.py
    @ brief
    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""
import tensorflow as tf
import time
import re
from img_utils import *
from imageio import imwrite


def optimizer(loss, var_list, learning_rate=0.0002, beta1=0.5):
    """
    Adam Optimizer
    :param loss:
    :param var_list:
    :param learning_rate:
    :param beta1:
    :return:
    """
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss=loss, var_list=var_list)

    return opt


def load_checkpoint(model):
    """
    :param model:
    :return:
    """

    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s_%s" % (model.dataset_name, model.batch_size, model.output_size)
    checkpoint_dir = os.path.join(model.checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model.saver.restore(model.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def save_checkpoint(model, step):
    """
    :param model:
    :param step:
    :return:
    """
    model_name = model.gan_name + ".model"
    model_dir = "%s_%s_%s" % (model.dataset_name, model.batch_size, model.output_size)
    checkpoint_dir = os.path.join(model.checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.saver.save(model.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def run_train(model, trainset_dir, valset_dir, sample_size, scale_size, crop_size, flip, training_epochs,
              flag_checkpoint, checkpoint_counter=0):
    """

    :param model:
    :param trainset_dir:
    :param valset_dir:
    :param sample_size:
    :param scale_size:
    :param crop_size:
    :param flip:
    :param training_epochs:
    :param flag_checkpoint:
    :param checkpoint_counter:
    :return:
    """
    counter = 1
    start_time = time.time()

    if flag_checkpoint:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # dataset (train)
    input_dir_train = os.path.join(trainset_dir, "input")
    output_dir_train = os.path.join(trainset_dir, "output")
    train_list = os.listdir(input_dir_train)

    # dataset (validation)
    input_dir_val = os.path.join(valset_dir, "input")
    output_dir_val = os.path.join(valset_dir, "output")
    val_list = os.listdir(input_dir_val)

    # training loop
    print('\n===== Start : Pix2Pix training =====\n')
    for epoch in range(training_epochs):

        total_batch = int(len(train_list) / model.batch_size)

        for idx in range(total_batch):

            batch_list = train_list[idx*model.batch_size:(idx+1)*model.batch_size]

            batch_input = []
            batch_output = []
            for batch_file in batch_list:
                inimg, outimg = load_images(batch_file, input_dir_train, output_dir_train, scale_size, crop_size, flip)
                batch_input.append(inimg)
                batch_output.append(outimg)

            batch_input = np.array(batch_input).astype(np.float32)
            batch_output = np.array(batch_output).astype(np.float32)

            # update discriminator
            summary_str = model.update_discriminator(input_image=batch_input, output_image=batch_output)
            model.writer.add_summary(summary_str, counter)

            # update generator
            summary_str = model.update_generator(input_image=batch_input, output_image=batch_output)
            model.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            summary_str = model.update_generator(input_image=batch_input, output_image=batch_output)
            model.writer.add_summary(summary_str, counter)

            # print loss
            err_d_fake = model.sess.run(model.D_loss_fake, feed_dict={
                model.real_input: batch_input, model.real_output: batch_output})
            err_d_real = model.sess.run(model.D_loss_real, feed_dict={
                model.real_input: batch_input, model.real_output: batch_output})
            err_g = model.sess.run(model.G_loss, feed_dict={
                model.real_input: batch_input, model.real_output: batch_output})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                  % (epoch, idx, total_batch, time.time() - start_time, err_d_fake + err_d_real, err_g))

            # sampler
            if np.mod(counter, 100) == 0:

                # pick samples randomly
                # sample_list = np.random.choice(val_list, size=sample_size, replace=False)
                sample_list = val_list[:sample_size]
                samples_input = []
                samples_output = []
                for sample_file in sample_list:
                    inimg, outimg = load_images(sample_file, input_dir_val, output_dir_val, scale_size, crop_size, flip)
                    samples_input.append(inimg)
                    samples_output.append(outimg)

                samples_input = np.array(samples_input).astype(np.float32)
                samples_output = np.array(samples_output).astype(np.float32)

                samples_generate, sample_d_loss, sample_g_loss = model.sess.run(
                    [model.fake_output_sample, model.D_loss, model.G_loss],
                    feed_dict={model.real_input: samples_input,
                               model.real_output: samples_output}
                    )
                manifold_h = int(np.ceil(np.sqrt(samples_generate.shape[0])))
                manifold_w = int(np.floor(np.sqrt(samples_generate.shape[0])))

                samples_results = image_merge(samples_input, samples_generate, [manifold_h, manifold_w])
                save_dir = os.path.join(model.sample_dir, model.dataset_name)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                save_file = "train_%02d_%04d.png" % (epoch, idx)
                save_path = os.path.join(save_dir, save_file)

                samples_results = image_denorm(samples_results)

                imwrite(save_path, samples_results)

                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (sample_d_loss, sample_g_loss))

            # checkpoint
            if np.mod(counter, 500) == 0:
                save_checkpoint(model, counter)


def run_test(model, testset_dir):
    """

    :param model:
    :param testset_dir:
    :return:
    """

    print('\n===== Start : Pix2Pix testing =====\n')

    # testset
    input_dir_test = os.path.join(testset_dir, "input")
    output_dir_test = os.path.join(testset_dir, "output")
    test_list = os.listdir(input_dir_test)

    # run test
    total_batch = int(len(test_list) / model.batch_size)

    start_time = time.time()
    counter = 1

    for idx in range(total_batch):

        batch_list = test_list[idx * model.batch_size:(idx + 1) * model.batch_size]

        batch_input = []
        batch_output = []
        for batch_file in batch_list:
            inimg, outimg = load_images(batch_file, input_dir_test, output_dir_test, training=False)
            batch_input.append(inimg)
            batch_output.append(outimg)

        batch_input = np.array(batch_input).astype(np.float32)
        batch_output = np.array(batch_output).astype(np.float32)

        batch_generate = model.sess.run([model.fake_output_sample], feed_dict={model.real_input: batch_input,
                                                                              model.real_output: batch_output})

        # write results
        save_dir = os.path.join(model.test_dir, model.dataset_name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for batch_idx, batch_file in enumerate(batch_list):
            save_file = "test_%s.png" % os.path.splitext(batch_file)[0]
            save_path = os.path.join(save_dir, save_file)

            test_merge = image_merge_results(batch_input[batch_idx], batch_generate[batch_idx], batch_output[batch_idx])
            test_results = image_denorm(test_merge)

            imwrite(save_path, test_results)

            print("Test: [%4d/%4d] time: %4.4f" % (counter, len(test_list), time.time() - start_time))
            counter += 1

