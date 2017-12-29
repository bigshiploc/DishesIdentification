# -*- coding: UTF-8 -*-
from PIL import Image
# import input_data
import model
import numpy as np
import tensorflow as tf


# def get_one_image(train):
#     '''Randomly pick one image from training data
#     Return: ndarray
#     '''
#     n = len(train)
#     ind = np.random.randint(0, n) #从n张图片中随意选出一张
#     img_dir = train[ind]
#
#     image = Image.open(img_dir)
#     # plt.imshow(image)
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image

def evaluate_one_image():

    path = input("输入你想要识别的图片路径:")
    image = Image.open(path)
    # image = image.resize((208, 208))
    image = image.resize([208,208])
    image = image.convert('RGB')
    image_array = np.array(image)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 10

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3]) #将图片变成思维张量
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.
        logs_train_dir = './model'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print (max_index)
            print (prediction)
            if max_index==0:
                print('This is 蚂蚁上树 with possibility %.6f' %prediction[:, 0])
            elif max_index==1:
                print('This is 干煸四季豆 with possibility %.6f' %prediction[:, 1])
            elif max_index==2:
                print ('This is 可乐鸡翅 with possibility %.6f' %prediction[:, 2])
            elif max_index==3:
                print ('This is 葱油藕片 with possibility %.6f' %prediction[:, 3])
            elif max_index==4:
                print ('This is 鸡蛋羹 with possibility %.6f' %prediction[:, 4])
            elif max_index==5:
                print ('This is 宫爆鸡丁 with possibility %.6f' %prediction[:, 5])
            elif max_index==6:
                print ('This is 糖醋鲤鱼 with possibility %.6f' %prediction[:, 6])
            elif max_index==7:
                print ('This is 糖醋里脊 with possibility %.6f' %prediction[:, 7])
            elif max_index==8:
                print ('This is 梅菜扣肉 with possibility %.6f' %prediction[:, 8])
            elif max_index==9:
                print ('This is 鱼香肉丝 with possibility %.6f' %prediction[:, 9])


evaluate_one_image()