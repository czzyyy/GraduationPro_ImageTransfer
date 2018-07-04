import time
import os
import tensorflow as tf
import numpy as np

from VGGModel import vgg19
from ImageTransfer.fast_art import transfer_utils as utils
from ImageTransfer.fast_art import transfer_net as net

# weights
CONTENT_WEIGHT = 5.0e0
STYLE_WEIGHT = 1.5e2
TV_WEIGHT = 2.0e1

# train para
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 4
PRINT_ITERATION = 500
IMAGE_SIZE = 256
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


# define the losses , the optimizer and train process
def train(content_path, style_path):
    save_path = '/'.join(style_path.split('/')[:-1]) + '/save/transfer_net_one.ckpt'
    tf_board_path = '/'.join(style_path.split('/')[:-1]) + '/tf_board/'

    # check content image number with the batch_size
    content_image_list = _get_files(content_path)
    mod = len(content_image_list) % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed slightly..')
        content_image_list = content_image_list[:-mod]

    # load style image
    style_image = utils.load_image_resize(style_path)  # [0 - 1]
    style_image = np.array([style_image])  # [1, h, w, c]

    # define the shapes
    batch_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    style_shape = style_image.shape
    print('style_shape: ', style_shape)

    with tf.name_scope('input'):
        style_input = tf.placeholder(dtype=tf.float32, shape=style_shape, name='style_input')
        content_input = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='content_input')  # [0 - 1]

    with tf.name_scope('style_gram'):
        style_gram_pre = dict()
        style_vgg = vgg19.Vgg19()
        style_vgg.build_without_fc(style_input)
        temp_dict = dict()
        temp_dict['relu1_1'] = style_vgg.relu1_1
        temp_dict['relu2_1'] = style_vgg.relu2_1
        temp_dict['relu3_1'] = style_vgg.relu3_1
        temp_dict['relu4_1'] = style_vgg.relu4_1
        temp_dict['relu5_1'] = style_vgg.relu5_1
        for layer in STYLE_LAYERS:
            _, height, width, filter_num = [i.value for i in temp_dict[layer].shape]
            features = tf.reshape(temp_dict[layer], [-1, filter_num])  # first dimension is 1
            gram = tf.matmul(tf.transpose(features), features) / tf.to_float(height * width * filter_num)
            style_gram_pre[layer] = gram

    with tf.Session() as se:
        style_gram = se.run(style_gram_pre, feed_dict={style_input: style_image})

    # get the content_image feature from vgg19
    with tf.name_scope('content_feature'):
        content_feature = dict()
        content_vgg = vgg19.Vgg19()
        content_vgg.build_without_fc(content_input)
        content_feature[CONTENT_LAYER] = content_vgg.relu4_2  # can not run without batch image

    # get pred by transfer net
    pred = net.net(content_input)  # pred [0.0 - 255.0]
    pred_rescale = pred / 255.0  # pred_rescale [0 - 1]

    # get the pred gram and feature from vgg19
    with tf.name_scope('pred_gram_feature'):
        pred_gram = dict()
        pred_vgg = vgg19.Vgg19()
        pred_vgg.build_without_fc(pred_rescale)
        temp_dict2 = dict()
        temp_dict2['relu1_1'] = pred_vgg.relu1_1
        temp_dict2['relu2_1'] = pred_vgg.relu2_1
        temp_dict2['relu3_1'] = pred_vgg.relu3_1
        temp_dict2['relu4_1'] = pred_vgg.relu4_1
        temp_dict2['relu5_1'] = pred_vgg.relu5_1
        for layer in STYLE_LAYERS:
            _, height, width, filter_num = [i.value for i in temp_dict2[layer].shape]
            features = tf.reshape(temp_dict2[layer], [BATCH_SIZE, height * width, filter_num])
            gram = tf.matmul(tf.transpose(features, perm=[0, 2, 1]),
                             features) / tf.to_float(height * width * filter_num)
            pred_gram[layer] = gram

        pred_feature = dict()
        pred_feature[CONTENT_LAYER] = pred_vgg.relu4_2  # can not run without batch image

    # cal loss
    with tf.name_scope('loss'):
        # content loss
        _, h, w, c = [i.value for i in content_feature[CONTENT_LAYER].shape]
        content_loss = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            content_feature[CONTENT_LAYER] - pred_feature[CONTENT_LAYER]) / tf.to_float(BATCH_SIZE * h * w * c))

        # style loss
        style_loss = 0
        for layer in STYLE_LAYERS:
            h, w = style_gram[layer].shape  # not tf.variable so no value attribute
            style_loss = style_loss + 2 * tf.nn.l2_loss(
                pred_gram[layer] - style_gram[layer]) / tf.to_float(h * w)
        style_loss = STYLE_WEIGHT * style_loss / tf.to_float(BATCH_SIZE)

        # total variation denoising use pred not pred_rescale
        _, yh, yw, yc = [i.value for i in pred[:, 1:, :, :].shape]
        tv_y_size = BATCH_SIZE * yh * yw * yc
        _, xh, xw, xc = [i.value for i in pred[:, :, 1:, :].shape]
        tv_x_size = BATCH_SIZE * xh * xw * xc
        y_tv = 2 * tf.nn.l2_loss(pred[:, 1:, :, :] - pred[:, :batch_shape[1]-1, :, :])
        x_tv = 2 * tf.nn.l2_loss(pred[:, :, 1:, :] - pred[:, :, :batch_shape[2]-1, :])
        tv_loss = TV_WEIGHT * (x_tv / tv_x_size + y_tv / tv_y_size)

        loss = content_loss + style_loss + tv_loss

        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('loss', loss)

    # define optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter(tf_board_path, sess.graph)
        for epoch in range(NUM_EPOCHS):
            num_examples = len(content_image_list)
            iterations = 0
            start_time = time.time()
            while iterations * BATCH_SIZE < num_examples:
                curr = iterations * BATCH_SIZE
                x_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_image_list[curr:curr + BATCH_SIZE]):
                    try:
                        x_batch[j] = utils.load_image_resize(img_p, height=IMAGE_SIZE, width=IMAGE_SIZE)
                    except ValueError:
                        print('read image error just ignore it', iterations)

                iterations += 1

                # optimize
                sess.run(optimizer, feed_dict={content_input: x_batch})

                # print
                is_print_iter = int(iterations) % PRINT_ITERATION == 0
                is_last = epoch == NUM_EPOCHS - 1 and iterations * BATCH_SIZE >= num_examples
                if is_print_iter or is_last:
                    end_time = time.time()
                    delta_time = end_time - start_time
                    print('delta_time: ', delta_time)
                    start_time = time.time()

                    get_data = [content_loss, style_loss, tv_loss, loss, pred_rescale]
                    return_data, merge_result = sess.run([get_data, merged], feed_dict={content_input: x_batch})

                    # save to tf board
                    writer.add_summary(merge_result, (epoch * num_examples + iterations * BATCH_SIZE) /
                                       (PRINT_ITERATION * BATCH_SIZE))

                    print("step {}/of epoch {}/{}...".format(
                        (epoch * num_examples + iterations * BATCH_SIZE) /
                        (PRINT_ITERATION * BATCH_SIZE), epoch + 1, NUM_EPOCHS),
                        "content_loss: {:.4f}".format(return_data[0]), "style_loss: {:.4f}".format(return_data[1]),
                        "tv_loss: {:.4f}".format(return_data[2]), "loss: {:.4f}".format(return_data[3])
                    )

                    # save img
                    save_img_path = tf_board_path + 'check' + str((epoch * num_examples + iterations * BATCH_SIZE) /
                                                                  (PRINT_ITERATION * BATCH_SIZE))
                    utils.save_image(path=save_img_path + '.png', img=return_data[4][0])
        print('train done')
        # save sess
        saver.save(sess, save_path)
        print('save done')
