# segmenetation
import tensorflow as tf
import skimage.transform
import skimage.io
import numpy as np
import scipy.misc as misc
from ImageTransfer.slow_real import photo_style_my as train
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
# end segmentation

# dir
STYLE_PATH = './data/style/tar7.png'
CONTENT_PATH = './data/input/in7.png'
SAVE_PATH = './data/output/'
SEGMENTATION_PATH = './data/segmentation/'

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

        # content
        tmp_content = utils.load_image_resize(CONTENT_PATH)
        h_c, w_c, _ = tmp_content.shape
        tmp_image_content = misc.imresize(tmp_content, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
        pred_content = sess.run(pred_annotation, feed_dict={image: np.array([tmp_image_content]), keep_probability: 1.0})
        pred_content = np.squeeze(pred_content, axis=3)
        result_content = skimage.transform.resize(pred_content[0].astype(np.uint8) / 255.0, (h_c, w_c))
        skimage.io.imsave(SEGMENTATION_PATH + CONTENT_PATH.split('/')[-1], (result_content * 255.0).astype(np.uint8))
        print('content ', SEGMENTATION_PATH + CONTENT_PATH.split('/')[-1])

        # style
        tmp_style = utils.load_image_resize(STYLE_PATH)
        h_s, w_s, _ = tmp_style.shape
        tmp_image_style = misc.imresize(tmp_style, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
        pred_style = sess.run(pred_annotation, feed_dict={image: np.array([tmp_image_style]), keep_probability: 1.0})
        pred_style = np.squeeze(pred_style, axis=3)
        result_style = skimage.transform.resize(pred_style[0].astype(np.uint8) / 255.0, (h_s, w_s))
        skimage.io.imsave(SEGMENTATION_PATH + STYLE_PATH.split('/')[-1], (result_style * 255.0).astype(np.uint8))
        print('style ', SEGMENTATION_PATH + STYLE_PATH.split('/')[-1])

    print('segmentation over')
    tf.reset_default_graph()
    # output_save_path = train.train(CONTENT_PATH, STYLE_PATH, SAVE_PATH, SEGMENTATION_PATH + CONTENT_PATH.split('/')[-1]
    #                                , SEGMENTATION_PATH + STYLE_PATH.split('/')[-1])  # [0-255]
    # print(output_save_path)
