import tensorflow as tf
from ImageTransfer.fast_art import transfer_ops as ops


# transfer net define
def net(image, name='net'):
    """
    :param image: the original image with scaled [0, 1]
    :param  name: the name scope name
    :return: the transfered image with scaled [0, 255]
    """
    with tf.name_scope(name):
        conv1 = ops.conv_layer(image, 32, 9, 1, 'conv1')
        conv2 = ops.conv_layer(conv1, 64, 3, 2, 'conv2')
        conv3 = ops.conv_layer(conv2, 128, 3, 2, 'conv3')
        res_block1 = ops.residual_block(conv3, 'res_block1', 128, 3, 1)
        res_block2 = ops.residual_block(res_block1, 'res_block2', 128, 3, 1)
        res_block3 = ops.residual_block(res_block2, 'res_block3', 128, 3, 1)
        res_block4 = ops.residual_block(res_block3, 'res_block4', 128, 3, 1)
        res_block5 = ops.residual_block(res_block4, 'res_block5', 128, 3, 1)
        deconv1 = ops.conv_tranpose_layer(res_block5, 64, 3, 2, 'deconv1')
        deconv2 = ops.conv_tranpose_layer(deconv1, 32, 3, 2, 'deconv2')
        conv4 = ops.conv_layer(deconv2, 3, 9, 1, 'conv4', instance_norm=True, relu=False)
        pre_output = (tf.nn.tanh(conv4) + 1) * 127.5  # [0-255]
        output = pre_output * 0.85 + image * 255.0 * 0.15
        return output
