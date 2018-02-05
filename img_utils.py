"""
    @ file : img_utils.py
    @ brief
    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""
import numpy as np
from skimage.transform import resize as imresize
import os
from imageio import imread
import matplotlib.pyplot as plt


def load_images(file_name, input_dir, output_dir, scale_size=286, crop_size=256, flip=True, training=True):
    """

    :param file_name:
    :param input_dir:
    :param output_dir:
    :param scale_size:
    :param crop_size:
    :param flip:
    :param training:
    :return:
    """
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    input_image = imread(input_path)
    output_image = imread(output_path)

    input_image, output_image = preprocess_image(input_image, output_image, scale_size, crop_size, flip, training)

    # normalize (-1 ~ 1)
    input_image = image_norm(input_image)
    output_image = image_norm(output_image)

    """
    input_denorm = image_denorm(input_image)
    output_denorm = image_denorm(output_image)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(input_denorm)
    a.set_title('Input')

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(output_denorm)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('Output')

    plt.show()
    """

    return input_image, output_image


def preprocess_image(img1, img2, scale_size, crop_size, flip, training):
    """

    :param img1:
    :param img2:
    :param scale_size:
    :param crop_size:
    :param flip:
    :param training:
    :return:
    """

    if training:

        # random cropping when training
        img1 = imresize(img1, (scale_size, scale_size))
        img2 = imresize(img2, (scale_size, scale_size))

        ws = np.random.randint(0, scale_size-crop_size)
        hs = np.random.randint(0, scale_size-crop_size)

        img1 = img1[hs:hs + crop_size, ws:ws + crop_size]
        img2 = img2[hs:hs + crop_size, ws:ws + crop_size]

        # flip
        if flip and np.random.random() > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)

    else:

        img1 = imresize(img1, (crop_size, crop_size))
        img2 = imresize(img2, (crop_size, crop_size))

    return img1, img2


def image_norm(img):
    """

    :param img:
    :return:
    """
    img_dtype = img.dtype

    if np.issubdtype(img_dtype, int):
        return (img - 127.5) / 127.5

    elif np.issubdtype(img_dtype, float):
        return (img - 0.5) / 0.5


def image_denorm(img):
    """

    :param img:
    :return:
    """
    img_dtype = img.dtype

    if np.issubdtype(img_dtype, int):
        return (img * 127.5) + 127.5

    elif np.issubdtype(img_dtype, float):
        return (img * 0.5) + 0.5


def image_merge(img_in, img_out, size):
    """
    create merged image from images (input, output)
    :param img_in:
    :param img_out:
    :param size:
    :return:
    """
    height, width = img_out.shape[1], img_out.shape[2]
    merge_img = np.zeros((height * size[0], width * size[1] * 2, 3))
    for idx, image in enumerate(img_out):
        i = idx % size[1]
        j = idx // size[1]

        merge_img[j * height:j * height + height, (2 * i + 0) * width:(2 * i + 0) * width + width, :] = img_in[idx]
        merge_img[j * height:j * height + height, (2 * i + 1) * width:(2 * i + 1) * width + width, :] = img_out[idx]

    return merge_img


def image_merge_results(img_in, img_gen, img_out):
    """
    create merged image from images (input, generated, output)
    :param img_in:
    :param img_gen:
    :param img_out:
    :return:
    """
    height, width = img_out.shape[0], img_out.shape[1]
    merge_img = np.zeros((height, width * 3, 3))

    merge_img[:, 0 * width:0 * width + width, :] = img_in
    merge_img[:, 1 * width:1 * width + width, :] = img_gen
    merge_img[:, 2 * width:2 * width + width, :] = img_out

    return merge_img
