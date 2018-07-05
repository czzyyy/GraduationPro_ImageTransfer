import tensorflow as tf

WEIGHTS_INIT_STDEV = 0.1


# instance norm can be used in both train and test
def instance_normalizer(x, name='instance_norm'):
    with tf.name_scope(name):
        batch, height, width, channel = [i for i in x.shape]
        var_shape = [channel]
        # return axes 's mean and variance
        mu, sigma_sq = tf.nn.moments(x, [1, 2], keep_dims=True)
        # shift is beta, scale is alpha in in_norm form
        shift = tf.Variable(tf.zeros(var_shape), name='shift')
        scale = tf.Variable(tf.ones(var_shape), name='scale')
        epsilon = 1e-3
        normalized = (x-mu)/(sigma_sq + epsilon)**(0.5)
        return scale * normalized + shift


# batch norm train=True when training =False when testing
def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


# conv2d
def conv_layer(x, filter_num, filter_size, stride, name, instance_norm=True, relu=True):
    with tf.name_scope(name):
        _, h, w, c = [i.value for i in x.shape]
        weight_shape = [filter_size, filter_size, c, filter_num]
        weight = tf.Variable(tf.truncated_normal(shape=weight_shape, dtype=tf.float32,
                                                 stddev=WEIGHTS_INIT_STDEV, seed=1), name='conv_w')
        strides_shape = [1, stride, stride, 1]
        x = tf.nn.conv2d(x, weight, strides_shape, padding='SAME')
        if instance_norm:
            x = instance_normalizer(x)
        if relu:
            x = tf.nn.relu(x)

        return x


# deconv2d
def conv_tranpose_layer(x, filter_num, filter_size, stride, name, instance_norm=True, relu=True):
    with tf.name_scope(name):
        ba, h, w, c = [i.value for i in x.shape]
        weight_shape = [filter_size, filter_size, filter_num, c]
        weight = tf.Variable(tf.truncated_normal(shape=weight_shape, dtype=tf.float32,
                                                 stddev=WEIGHTS_INIT_STDEV, seed=1), name='deconv_w')
        strides_shape = [1, stride, stride, 1]
        new_shape = [ba, h * stride, w * stride, filter_num]
        x = tf.nn.conv2d_transpose(x, weight, output_shape=new_shape, strides=strides_shape, padding='SAME')
        if instance_norm:
            x = instance_normalizer(x)
        if relu:
            x = tf.nn.relu(x)

        return x


# res block
# attention here is different from the paper which is no padding in res_block so size - 2,but here use SAME padding
# size is unchanged, both has stride=1 filter_size=3
def residual_block(x, name, filter_num=128, filter_size=3, stride=1):
    with tf.name_scope(name):
        res_x = conv_layer(x, filter_num, filter_size, stride, 'conv1')
        res_x = conv_layer(res_x, filter_num, filter_size, stride, 'conv2', instance_norm=True, relu=False)
        return x + res_x
