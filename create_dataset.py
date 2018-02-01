"""
    @ file : create_dataset.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2018.01.29
    @ version : 1.0
"""
import os
import subprocess
import tarfile
from imageio import imread, imwrite
import shutil


# DB URL
db_type = ["facades", "cityscapes", "maps"]


def parse_images(_img_dir):
    """

    :param _img_dir:
    :return:
    """

    img_list = os.listdir(_img_dir)

    input_dir = os.path.join(_img_dir, "input")
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)

    output_dir = os.path.join(_img_dir, "output")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    pair_dir = os.path.join(_img_dir, "pair")
    if not os.path.isdir(pair_dir):
        os.mkdir(pair_dir)

    for idx, item in enumerate(img_list):

        img_name = os.path.join(_img_dir, item)
        img = imread(img_name)
        width = int(img.shape[1])
        width_half = int(width/2)

        output_img = img[:, 0:width_half]
        input_img = img[:, width_half:width]

        save_output_name = output_dir + "/" + "%05d.jpg" % (idx+1)
        save_input_name = input_dir + "/" + "%05d.jpg" % (idx+1)

        imwrite(save_output_name, output_img)
        imwrite(save_input_name, input_img)
        shutil.move(img_name, pair_dir + "/" + "%05d.jpg" % (idx+1))


# Info
print("="*100)
print("Datasets for pix2pix will be downloaded")
print("\n  - Check list of datasets URL")

db_url = []
for item in db_type:
    item_url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % item
    db_url.append(item_url)
    print("\t>> " + item_url)

# dir
print("\n  - Check directories")
head_dir = "datasets"
if not os.path.isdir(head_dir):
    print("\t>> Make :" + head_dir)
    os.mkdir(head_dir)
else:
    print("\t>> Found : " + head_dir)


# Download and Unzip
flag_parse = []
print("\n  - Download & Unzip")
for item_url, item_db in zip(db_url, db_type):

    file_name = os.path.basename(item_url)
    local_file_path = os.path.join(head_dir, file_name)

    # download
    if not os.path.isfile(local_file_path):
        print("\t>> Downloading : " + file_name)
        cmd = ["curl", item_url, "-o", local_file_path]
        subprocess.call(cmd)
    else:
        print("\t>> Found : " + file_name)

    # Unzip
    print("\t>> Extracting : " + file_name)
    if not os.path.isdir(os.path.join(head_dir, item_db)):
        tar = tarfile.open(local_file_path)
        tar.extractall(path=head_dir)
        tar.close()
        flag_parse.append(True)
    else:
        flag_parse.append(False)

# Parsing Datasets
print("\n  - Parsing Images")
sub_dir = ["train", "test", "val"]
for item, flag_idx in zip(db_type, flag_parse):

    if flag_idx:
        print("\t>> Parsing : " + item)
        db_dir = os.path.join(head_dir, item)
        for sub in sub_dir:
            img_dir = os.path.join(db_dir, sub)
            if os.path.isdir(img_dir):
                print("\t\tRun: " + sub)
                parse_images(img_dir)
    else:
        print("\t>> Parsing (exist): " + item)

print("="*100)


