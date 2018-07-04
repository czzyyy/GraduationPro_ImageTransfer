import os
from ImageTransfer.slow_real import transfer_utils as utils
import skimage.io

# segmenetation
import tensorflow as tf
import skimage.transform
import numpy as np
import scipy.misc as misc
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
# end segmentation

# dir
IMAGE_PATH = './train2017'
SEGMENTATION_PATH = './train2017_segmentation/'


def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

if __name__ == '__main__':
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

        image_list = _get_files(IMAGE_PATH)
        num = 0
        for img_p in image_list:
            tmp = utils.load_image_resize(img_p)
            h, w, _ = tmp.shape
            tmp_image = misc.imresize(tmp, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
            pred = sess.run(pred_annotation, feed_dict={image: np.array([tmp_image]), keep_probability: 1.0})
            pred = np.squeeze(pred, axis=3)
            result = skimage.transform.resize(pred[0].astype(np.uint8) / 255.0, (h, w))
            skimage.io.imsave(SEGMENTATION_PATH + img_p.split('/')[-1], (result * 255.0).astype(np.uint8))
            num = num + 1
            print('segmentation ', SEGMENTATION_PATH + img_p.split('/')[-1])
        print('seg over', num)