import tensorflow as tf


# %%
def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[11, 11, 3, 64], #（5*5的卷积核大小，3个颜色通道（彩色），16个卷积核 ）现有的形状
                                  dtype=tf.float32,  #类型
                                  regularizer=regularizer,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)) #初始化张量
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)) #初始化为0.0 ？原来是0.1
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') #卷积操作
        pre_activation = tf.nn.bias_add(conv, biases) #将卷积的结果加上 biases
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  #激活函数进行非线性化

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], #步长为3×3,尺寸为2×2最大池化层来池化
                               padding='VALID', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,  #lrn对结果进行处理（lrn对relu这种激活函数比较有用）
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 64, 192], #上一层的卷积核数量是16，所以第三个维度输入的通道数也为16
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)) #初始化为0.1
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], #步长为3×3,尺寸为1×1
                               padding='VALID', name='pooling2')

    # conv3
    with tf.variable_scope('conv3')  as scope:
        weights = tf.get_variable('weigths',
                                  shape=[3, 3, 192, 384],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2,weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name= 'conv3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weigths',
                                  shape=[3,3,384,256],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3,weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weigths',
                                  shape=[3,3,256,256],
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4,weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name='conv5')

    # pool3
    with tf.variable_scope('pooling3') as scope:
        pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  # 步长为3×3,尺寸为2×2
                               padding='VALID', name='pooling2')

    # local3 两个卷积之后是一个全连接层，将连个卷积的输出结果全部flatten
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool3, shape=[batch_size, -1]) #将pool2变成一维向量
        dim = reshape.get_shape()[1].value  #数据扁平化之后的长度
        weights = tf.get_variable('weights',
                                  shape=[dim, 128], #隐含节点数为128？ 348？
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)) #正态分布标准差为0.005
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))  #初始化为0.1
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  #这一层被l2正则约束然后激活

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 64],  #隐含节点数要下降一半
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')


    # softmax  正态分布标准差是上一个隐含层的节点数的倒数
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[64, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)) #初始化，原为0.1
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


# %%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example') #回归之后的交叉熵损失函数
        loss = tf.reduce_mean(cross_entropy, name='loss')  #对cross_entropy计算均值, 方差损失函数
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# %%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  #优化函数
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# %%
def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1) #默认top为1输出分数最高的那一类的准确率
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

# %%
