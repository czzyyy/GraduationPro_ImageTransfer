from ImageTransferWeb.algorithms.slow_real import photo_style_my as train
from ImageTransferWeb.algorithms.slow_real import transfer_utils as utils
import skimage.io

# segmenetation
import tensorflow as tf
import skimage.transform
import numpy as np
import scipy.misc as misc
slim = tf.contrib.slim
import ImageTransferWeb.algorithms.slow_real.inception_v3_fcn as inception_v3_fcn

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
# STYLE_PATH = './data/style/tar7.png'
# CONTENT_PATH = './data/input/in7.png'
# SAVE_PATH = './data/output/'
# SEGMENTATION_PATH = './data/segmentation/'


def start_slow_real_style(style_path='./data/style/tar7.png', content_path='./data/input/in7.png',
                          save_path='./data/output/', segmentation_path='./data/segmentation/'):
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

        # content
        tmp_content = utils.load_image_resize(content_path)
        h_c, w_c, _ = tmp_content.shape
        tmp_image_content = misc.imresize(tmp_content, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
        pred_content = sess.run(pred_annotation, feed_dict={image: np.array([tmp_image_content]), keep_probability: 1.0})
        pred_content = np.squeeze(pred_content, axis=3)
        result_content = skimage.transform.resize(pred_content[0].astype(np.uint8) / 255.0, (h_c, w_c))
        skimage.io.imsave(segmentation_path + content_path.split('/')[-1], (result_content * 255.0).astype(np.uint8))
        print('content ', segmentation_path + content_path.split('/')[-1])

        # style
        tmp_style = utils.load_image_resize(style_path)
        h_s, w_s, _ = tmp_style.shape
        tmp_image_style = misc.imresize(tmp_style, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
        pred_style = sess.run(pred_annotation, feed_dict={image: np.array([tmp_image_style]), keep_probability: 1.0})
        pred_style = np.squeeze(pred_style, axis=3)
        result_style = skimage.transform.resize(pred_style[0].astype(np.uint8) / 255.0, (h_s, w_s))
        skimage.io.imsave(segmentation_path + style_path.split('/')[-1], (result_style * 255.0).astype(np.uint8))
        print('style ', segmentation_path + style_path.split('/')[-1])

    print('segmentation over')
    tf.reset_default_graph()
    output_save_path = train.train(content_path, style_path, save_path, segmentation_path + content_path.split('/')[-1]
                                   , segmentation_path + content_path.split('/')[-1])  # [0-255]
    return output_save_path
