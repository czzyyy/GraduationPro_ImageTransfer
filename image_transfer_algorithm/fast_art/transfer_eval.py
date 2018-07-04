import tensorflow as tf
import numpy as np

from ImageTransfer.fast_art import transfer_utils as utils
from ImageTransfer.fast_art import transfer_net as net


# get the stored model to generate a transfered image
def evaluate(model_path, test_image_path, image_save_path):
    image = np.array([utils.load_image_resize(test_image_path)])
    input_shape = image.shape
    print(input_shape)

    content_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name='content_input')
    pred = net.net(content_input)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        transfered_image = sess.run(pred, feed_dict={content_input: image})  # [0 - 255]
        utils.save_image(path=image_save_path, img=np.clip(transfered_image[0], 0, 255.) / 255.0)
    print('evaluate done')


if __name__ == '__main__':
    save_path = './stored_models/nine/save/transfer_net_one.ckpt'
    evaluate(model_path=save_path, test_image_path='./test/test4.jpg', image_save_path='./output/ninetest4.png')