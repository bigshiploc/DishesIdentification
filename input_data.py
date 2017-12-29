# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import skimage.io as io  #读取图片的库


def get_file(file_dir):
    '''Get full image directory and corresponding labels
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string
        labels: label, list, int
    '''

    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
        # get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    # assign 10 labels based on the folder names
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
        letter = letter.split("\\")[1]
        if letter == '蚂蚁上树':
            labels = np.append(labels, n_img * [0])
        elif letter == '干煸四季豆':
            labels = np.append(labels, n_img * [1])
        elif letter == '可乐鸡翅':
            labels = np.append(labels, n_img * [2])
        elif letter == '葱油藕片':
            labels = np.append(labels, n_img * [3])
        elif letter == '鸡蛋羹':
            labels = np.append(labels, n_img * [4])
        elif letter == '宫爆鸡丁':
            labels = np.append(labels, n_img * [5])
        elif letter == '糖醋鲤鱼':
            labels = np.append(labels, n_img * [6])
        elif letter == '糖醋里脊':
            labels = np.append(labels, n_img * [7])
        elif letter == '梅菜扣肉':
            labels = np.append(labels, n_img * [8])
        else:
            labels = np.append(labels, n_img * [9])

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

# label的转换
def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# image的转换
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i])  # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')

# name_test = 'train'
# test_dir = './data/train'
# save_dir = './data'
train_images, train_labels = get_file('./data/train')
test_image, test_labels = get_file('./data/test')
convert_to_tfrecord(train_images, train_labels, './data', 'train')
convert_to_tfrecord(test_image, test_labels, './data', 'test')