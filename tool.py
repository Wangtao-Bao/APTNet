from utils.data import *
import os
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt


def total_visulization_generation(dataset_dir, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/image'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')
        plt.close()
def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(320, 320))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(320, 320))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)
def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)
def load_dataset (root, dataset):
    train_txt = root + '/' + dataset + '/' +   'train_'+ dataset +'.txt'
    test_txt  = root + '/' + dataset + '/' +   'test_'+ dataset +'.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt