import os
import numpy as np
import tensorflow as tf
import model

# 指定参数开始训练
N_CLASSES = 10 #类别数
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 100000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # 学习率 建议小于0.0001


# 读取TFRecord数据
def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_W, IMG_H, 3])
    image = tf.cast(image, tf.float32)

    #data augmentation here
    distorted_image = tf.random_crop(image, [208, 208, 3]) #随机剪裁
    distorted_image = tf.image.random_flip_left_right(distorted_image) #水平反转
    distorted_image = tf.image.random_brightness(distorted_image,  # 设置随机亮度
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,  # 设置随机饱和度
                                               lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)  # 对数据进行标准化

    ##########################################################
    # all the images of notMNIST are 28*28, you need to  change the image size if you use other dataset.
     # normalize
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([float_image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    return image_batch, tf.reshape(label_batch, [batch_size])


def run_training():
    # you need to change the directories to yours.

    logs_train_dir = './logs/train/'

    train_batch, train_label_batch = read_and_decode('./data/train.tfrecords', batch_size=BATCH_SIZE)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join( './model/model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print("training is done")


run_training()

