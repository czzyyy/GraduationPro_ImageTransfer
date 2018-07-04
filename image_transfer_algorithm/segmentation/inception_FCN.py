import tensorflow as tf
import skimage.transform
import skimage.io
import numpy as np
import scipy.misc as misc
from ImageTransfer.slow_real import transfer_utils as utils

slim = tf.contrib.slim

import ImageTransfer.segmentation.inception_v3_fcn as inception_v3_fcn

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "logs/all", "path to logs directory")
tf.flags.DEFINE_string(
    'skip_layers', '8s',
    'Skip architecture to use: 32s/ 16s/ 8s')


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224

def get_seg(tmp_image):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")

    pred_annotation, logits, end_points = inception_v3_fcn.inception_v3_fcn(image, num_classes=NUM_OF_CLASSESS,
                                                                            dropout_keep_prob=keep_probability,
                                                                            skip=FLAGS.skip_layers)
    with tf.Session() as sess:
        print("Setting up Saver...")
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        # tmp_image is [0-255]
        if len(tmp_image.shape) < 3:  # make sure images are of shape(h,w,3)
            tmp_image = np.array([tmp_image for i in range(3)])
        h, w, c = tmp_image.shape
        tmp_image = misc.imresize(tmp_image, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
        valid_images = np.array([tmp_image])
        pred= sess.run(pred_annotation, feed_dict={image: valid_images, keep_probability: 1.0})
        pred = np.squeeze(pred, axis=3)
        result = skimage.transform.resize(pred[0].astype(np.uint8) / 255.0, (h, w))
        return (result * 255.0).astype(np.uint8)  # [0-255]


if __name__ == '__main__':
    tmp_image = utils.load_image_resize('./results/tar9.png')
    result = get_seg(tmp_image)
    skimage.io.imsave('results/pred_tar9.png', result)
