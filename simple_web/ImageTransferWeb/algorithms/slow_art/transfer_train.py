import time
import os
import tensorflow as tf
import numpy as np

from ImageTransferWeb.algorithms.slow_art import vgg19
from ImageTransferWeb.algorithms.slow_art import transfer_utils as utils

# weights
CONTENT_WEIGHT = 5.0e0
STYLE_WEIGHT = 1.5e2
TV_WEIGHT = 2.0e2

# train para
LEARNING_RATE = 1.0
PRINT_ITERATION = 100
MAX_ITER = 7000
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


# define the losses , the optimizer and train process
def train(content_path, style_path, save_path):
    #tf_board_path = './tf_board/'

    # load style image
    with tf.name_scope('input'):
        style_image = utils.load_image_resize(style_path)  # [0 - 1]
        style_width, style_height = style_image.shape[1], style_image.shape[0]
        style_image = style_image.reshape((1, style_height, style_width, 3)).astype(np.float32)

        content_image = utils.load_image_resize(content_path)  # [0 - 1]
        content_width, content_height = content_image.shape[1], content_image.shape[0]
        content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)
        print('style shape: ', style_image.shape)
        print('content shape: ', content_image.shape)

    with tf.name_scope('init_image'):
        # Set the input_image to be a weighted average of the content_image and a noise_image
        noise_image = np.random.uniform(0, 255, size=(1, content_height, content_width, 3)).astype(np.float32)
        init_image = noise_image * 0.6 + (content_image * 255.0) * (1 - 0.6)
        # make it trainable
        input_image = tf.Variable(init_image)  # [0-255]

    with tf.name_scope('style_gram'):
        style_gram_pre = dict()
        style_vgg = vgg19.Vgg19()
        style_vgg.build_without_fc(tf.constant(style_image))
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
    with tf.Session() as style_sess:
        style_gram = style_sess.run(style_gram_pre)

    # get the content_image feature from vgg19
    with tf.name_scope('content_feature'):
        content_feature_pre = dict()
        content_vgg = vgg19.Vgg19()
        content_vgg.build_without_fc(tf.constant(content_image))
        content_feature_pre[CONTENT_LAYER] = content_vgg.relu4_2
    with tf.Session() as content_sess:
        content_feature = content_sess.run(content_feature_pre)

    # get the init image gram and feature from vgg19
    with tf.name_scope('pred_gram_feature'):
        pred_gram = dict()
        pred_vgg = vgg19.Vgg19()
        pred_vgg.build_without_fc(input_image / 255.0)  # change to [0-1]
        temp_dict2 = dict()
        temp_dict2['relu1_1'] = pred_vgg.relu1_1
        temp_dict2['relu2_1'] = pred_vgg.relu2_1
        temp_dict2['relu3_1'] = pred_vgg.relu3_1
        temp_dict2['relu4_1'] = pred_vgg.relu4_1
        temp_dict2['relu5_1'] = pred_vgg.relu5_1
        for layer in STYLE_LAYERS:
            _, height, width, filter_num = [i.value for i in temp_dict2[layer].shape]
            features = tf.reshape(temp_dict2[layer], [1, height * width, filter_num])
            gram = tf.matmul(tf.transpose(features, perm=[0, 2, 1]),
                             features) / tf.to_float(height * width * filter_num)
            pred_gram[layer] = gram

        pred_feature = dict()
        pred_feature[CONTENT_LAYER] = pred_vgg.relu4_2  # not run by sess

    # cal loss
    with tf.name_scope('loss'):
        # content loss
        _, h, w, c = content_feature[CONTENT_LAYER].shape
        content_loss = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            content_feature[CONTENT_LAYER] - pred_feature[CONTENT_LAYER]) / tf.to_float(1 * h * w * c))

        # style loss
        style_loss = 0
        for layer in STYLE_LAYERS:
            h, w = style_gram[layer].shape  # not tf.variable so no value attribute
            style_loss = style_loss + 2 * tf.nn.l2_loss(
                pred_gram[layer] - style_gram[layer]) / tf.to_float(h * w)
        style_loss = STYLE_WEIGHT * style_loss / 5.0  # five layer

        # total variation denoising use [0-255]
        _, yh, yw, yc = [i.value for i in input_image[:, 1:, :, :].shape]
        tv_y_size = 1 * yh * yw * yc
        _, xh, xw, xc = [i.value for i in input_image[:, :, 1:, :].shape]
        tv_x_size = 1 * xh * xw * xc
        y_tv = 2 * tf.nn.l2_loss(input_image[:, 1:, :, :] - input_image[:, :-1, :, :])
        x_tv = 2 * tf.nn.l2_loss(input_image[:, :, 1:, :] - input_image[:, :, :-1, :])
        tv_loss = TV_WEIGHT * (x_tv / tv_x_size + y_tv / tv_y_size)

        loss = content_loss + style_loss + tv_loss

        #tf.summary.scalar('content_loss', content_loss)
        #tf.summary.scalar('style_loss', style_loss)
        #tf.summary.scalar('tv_loss', tv_loss)
        #tf.summary.scalar('loss', loss)

    # define optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.name_scope('train'):
        with tf.Session() as sess:
            # record the time
            start_time = time.time()

            sess.run(tf.global_variables_initializer())
            # merge summary
            # merged = tf.summary.merge_all()
            # choose dir
            # writer = tf.summary.FileWriter(tf_board_path, sess.graph)
            min_loss, best_image = float("inf"), None
            for i in range(1, MAX_ITER + 1):
                _, content_loss_, style_loss_, tv_loss_, loss_, output_image_ = sess.run([
                    optimizer, content_loss, style_loss, tv_loss, loss, input_image])
                if i % PRINT_ITERATION == 0:
                    end_time = time.time()
                    delta_time = end_time - start_time
                    print('delta_time: ', delta_time)
                    start_time = time.time()

                    # save to tf board
                    # merge_result = sess.run(merged)
                    # writer.add_summary(merge_result, i)

                    print("step {}/of {}...".format(i, MAX_ITER),"content_loss: {:.4f}".format(content_loss_)
                          , "style_loss: {:.4f}".format(style_loss_), "tv_loss: {:.4f}".format(tv_loss_)
                          , "loss: {:.4f}".format(loss_))

                    # save img
                    #save_img_path = tf_board_path + 'check'
                    #utils.save_image(path=save_img_path + str(i) + '.png', img=np.uint8(np.clip(output_image_[0],
                    #                                                                            0, 255.0)))
                if loss_ < min_loss:
                    min_loss, best_image = loss_, output_image_

            # end train
            output_save_path = save_path + content_path.split('/')[-1]
            utils.save_image(path=output_save_path, img=np.uint8(np.clip(best_image[0], 0, 255.0)))
            print('save: ' + output_save_path)
            print('train done')
            print("min_loss: {:.4f}".format(min_loss))
            return output_save_path
