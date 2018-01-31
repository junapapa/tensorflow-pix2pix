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

# DB URL
db_type = ["facades", "cityscapes", "maps"]

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
head_fold = "datasets"
if not os.path.isdir(head_fold):
    print("\t>> Make :" + head_fold)
    os.mkdir(head_fold)
else:
    print("\t>> Found : " + head_fold)


# Download and Unzip
print("\n  - Download & Unzip")
for item_url in db_url:

    file_name = os.path.basename(item_url)
    local_file_path = os.path.join(head_fold, file_name)

    # download
    if not os.path.isfile(local_file_path):
        print("\t>> Downloading : " + file_name)
        cmd = ["curl", item_url, "-o", local_file_path]
        subprocess.call(cmd)
    else:
        print("\t>> Found : " + file_name)

    # Unzip
    print("\t>> Extracting : " + file_name)
    tar = tarfile.open(local_file_path)
    tar.extractall(path=head_fold)
    tar.close()

print("="*100)
