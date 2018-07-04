import numpy as np
import tensorflow as tf
import vgg19
import time
import transfer_utils as utils
from closed_form_matting import getLaplacian
import transfer_net as net
import math
import os

# weights
CONTENT_WEIGHT = 1.5e1
STYLE_WEIGHT = 8.0e1
TV_WEIGHT = 1.0e-3

# train para
LEARNING_RATE = 1e-3
NUM_EPOCHS = 6
BATCH_SIZE = 4
PRINT_ITERATION = 100
IMAGE_SIZE = 256


def _get_files(img_dir):
    files = utils.list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def load_seg(ima_seg):
    # para has been resized and is [0-255]
    def _extract_mask(seg, cla):
        mask = np.multiply((seg[:, :, :] >= cla * 15).astype(np.uint8), (seg[:, :, :] < (cla + 1) * 15).astype(np.uint8))
        return mask.astype(np.float32)

    color_masks = []
    for i in range(10):
        color_masks.append(np.expand_dims(_extract_mask(ima_seg, i), -1))
    # add a left one
    mask_left = (ima_seg[:, :, :] >= 150).astype(np.uint8)
    color_masks.append(np.expand_dims(mask_left.astype(np.float32), -1))
    return color_masks


def gram_matrix(activations):
    # height = tf.shape(activations)[1]
    # width = tf.shape(activations)[2]
    # num_channels = tf.shape(activations)[3]
    # gram = tf.transpose(activations, [0, 3, 1, 2])
    # gram = tf.reshape(gram, [num_channels, width * height])
    # gram = tf.matmul(gram, gram, transpose_b=True)
    batch_s, height, width, filter_num = [i.value for i in activations.shape]
    features = tf.reshape(activations, [batch_s, height * width, filter_num])
    gram = tf.matmul(tf.transpose(features, perm=[0, 2, 1]), features) / tf.to_float(height * width * filter_num)
    return gram


# def content_loss(const_layer, var_layer, weight):
#     b, h, w, c = [i.value for i in var_layer.shape]
#     loss = weight * (2 * tf.nn.l2_loss(var_layer - const_layer) / tf.to_float(b * h * w * c))
#     return loss
#     # return tf.reduce_mean(tf.squared_difference(const_layer, var_layer)) * weight


def cal_style_loss(CNN_structure, const_layers, var_layers, content_segs, style_segs_para, weight):
    loss_styles = []
    layer_index = 0
    style_segs = style_segs_para

    _, content_seg_height, content_seg_width, _ = content_segs[0].shape
    _, style_seg_height, style_seg_width, _ = style_segs[0].shape
    for layer_name in CNN_structure:
        layer_name = layer_name[layer_name.find("/") + 1:]

        # down sampling segmentation
        if "pool" in layer_name:
            content_seg_width,content_seg_height=int(math.ceil(content_seg_width/2)),int(math.ceil(content_seg_height/2))
            style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)),int(math.ceil(style_seg_height / 2))

            for i in range(len(content_segs)):
                content_segs[i] = tf.image.resize_bilinear(content_segs[i], tf.constant((content_seg_height,
                                                                                         content_seg_width)))
                style_segs[i] = tf.image.resize_bilinear(style_segs[i], tf.constant((style_seg_height,
                                                                                     style_seg_width)))

        elif "conv" in layer_name:
            for i in range(len(content_segs)):
                # have some differences on border with torch
                content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
                                                 , ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
                                               , ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

        layer_name = layer_name[:layer_name.find("/")]
        if layer_name in var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
            print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]
            var_layer = var_layers[layer_index]
            layer_index = layer_index + 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                # style_mask_mean = tf.reduce_mean(style_seg)
                # gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                #                             lambda: gram_matrix_const / (tf.to_float(tf.size(const_layer))
                #                                                          * style_mask_mean),
                #                             lambda: gram_matrix_const)

                gram_matrix_var = gram_matrix(tf.multiply(var_layer, content_seg))
                content_mask_mean = tf.reduce_mean(content_seg)
                # gram_matrix_var = tf.cond(tf.greater(content_mask_mean, 0.),
                #                           lambda: gram_matrix_var / (tf.to_float(tf.size(var_layer))
                #                                                      * content_mask_mean),
                #                           lambda: gram_matrix_var)
                # czy change
                b, h, w = [i.value for i in gram_matrix_var.shape]
                diff_style_sum = 2 * tf.nn.l2_loss(gram_matrix_var - gram_matrix_const) / tf.to_float(h * w * b) * content_mask_mean
                # diff_style_sum = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)
    return loss_styles


# def total_variation_loss(output, weight):
#     tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) *
#                             (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) +
#                             (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) *
#                             (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / tf.to_float(2.0 * BATCH_SIZE)
#     return tv_loss * weight


def train(content_path, style_path, content_seg_path, style_seg_path):
    tf_board_path = '/'.join(style_path.split('/')[:-1]) + '/tf_board/'
    save_path = '/'.join(style_path.split('/')[:-1]) + '/save/transfer_net_two.ckpt'

    # global
    global content_masks_list

    # check content image and segmentation number with the batch_size
    content_image_list = _get_files(content_path)
    segmentation_image_list = _get_files(content_seg_path)  # has been cleaned
    mod = len(content_image_list) % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed slightly..')
        content_image_list = content_image_list[:-mod]
        segmentation_image_list = segmentation_image_list[:-mod]

    # load style image
    style_image = utils.load_image_resize(style_path)  # [0 - 1]
    style_image = np.array([style_image], dtype=np.float32)  # [1, h, w, c]

    # define the shapes
    style_shape = style_image.shape
    style_width, style_height = style_shape[2], style_shape[1]
    print('style_shape: ', style_shape)
    batch_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    seg_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)

    print('seg_ init')
    seg_init = np.zeros(seg_shape, dtype=np.float32)
    for j, img_seg in enumerate(segmentation_image_list[0:0 + BATCH_SIZE]):
        try:
            seg_init[j] = np.array(utils.load_image_resize(img_seg, height=IMAGE_SIZE, width=IMAGE_SIZE))
        except ValueError:
            print('read seg error just ignore it')
    content_masks_list = load_seg(seg_init * 255.0)

    # prepare input images
    with tf.name_scope('inputs'):
        content_input = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='content_input')  # [0 - 1]

    # load style segs
    with tf.name_scope('load_segs'):
        # [1, h, w, c]
        style_seg = np.array([utils.load_image_resize(style_seg_path, style_height, style_width)], dtype=np.float32)
        style_masks = load_seg(style_seg * 255.0)

    with tf.name_scope('constant_value'):
        # content
        vgg_content_const = vgg19.Vgg19()
        vgg_content_const.build_without_fc(content_input)
        content_layer_const = vgg_content_const.conv4_2

        # style
        vgg_style_const = vgg19.Vgg19()
        vgg_style_const.build_without_fc(tf.constant(style_image))
        style_layers_const = [vgg_style_const.conv1_1, vgg_style_const.conv2_1, vgg_style_const.conv3_1,
                              vgg_style_const.conv4_1, vgg_style_const.conv5_1]
        with tf.Session() as style_sess:
            style_fvs = style_sess.run(style_layers_const)

    # get pred by transfer net
    pred = net.net(content_input)  # pred [0.0 - 255.0]
    pred_rescale = pred / 255.0  # pred_rescale [0 - 1]

    with tf.name_scope('variable_value'):
        # trainable input image features
        vgg_var = vgg19.Vgg19()
        vgg_var.build_without_fc(pred_rescale)

    # feature layers
    style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
    content_layer_var = vgg_var.conv3_2

    # The whole CNN structure to downsample mask
    layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

    with tf.name_scope('loss'):
        # Content Loss
        b, h, w, c = [i.value for i in content_layer_var.shape]
        loss_content = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(content_layer_var - content_layer_const) /
                                         tf.to_float(b * h * w * c))

        # Style Loss
        loss_style = 0.0
        loss_styles_list = cal_style_loss(layer_structure_all, style_fvs, style_layers_var,
                                          content_masks_list, style_masks, STYLE_WEIGHT)
        for loss in loss_styles_list:
            loss_style += loss

        # Total Variational Loss
        tv_loss = tf.reduce_sum((pred[:, :-1, :-1, :] - pred[:, :-1, 1:, :]) *
                                (pred[:, :-1, :-1, :] - pred[:, :-1, 1:, :]) +
                                (pred[:, :-1, :-1, :] - pred[:, 1:, :-1, :]) *
                                (pred[:, :-1, :-1, :] - pred[:, 1:, :-1, :])) / tf.to_float(2.0 * BATCH_SIZE)
        loss_tv = tv_loss * TV_WEIGHT

        # all loss
        VGGNetLoss = loss_content + loss_tv + loss_style
        tf.summary.scalar('loss_content', loss_content)
        tf.summary.scalar('loss_style', loss_style)
        tf.summary.scalar('loss_tv', loss_tv)
        tf.summary.scalar('VGGNetLoss', VGGNetLoss)

    with tf.name_scope('optimizer_Adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(VGGNetLoss)

    # saver
    saver = tf.train.Saver()
    with tf.name_scope('train'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter(tf_board_path, sess.graph)
            for epoch in range(NUM_EPOCHS):
                num_examples = len(content_image_list)
                num_segs = len(segmentation_image_list)
                if num_examples != num_segs:
                    print('num_examples!=num_segs')
                    return
                iterations = 0
                start_time = time.time()
                while iterations * BATCH_SIZE < num_examples:
                    curr = iterations * BATCH_SIZE
                    x_batch = np.zeros(batch_shape, dtype=np.float32)
                    seg_batch = np.zeros(seg_shape, dtype=np.float32)
                    for j, img_p in enumerate(content_image_list[curr:curr + BATCH_SIZE]):
                        try:
                            x_batch[j] = utils.load_image_resize(img_p, height=IMAGE_SIZE, width=IMAGE_SIZE)
                        except ValueError:
                            print('read image error just ignore it', iterations)
                    for j, img_seg in enumerate(segmentation_image_list[curr:curr + BATCH_SIZE]):
                        try:
                            seg_batch[j] = np.array(utils.load_image_resize(img_seg, height=IMAGE_SIZE, width=IMAGE_SIZE))
                        except ValueError:
                            print('read seg error just ignore it', iterations)
                    content_masks_list.clear()
                    content_masks_list = []
                    content_masks_list = load_seg(seg_batch * 255.0)

                    iterations += 1

                    # optimize
                    sess.run(train_op, feed_dict={content_input: x_batch})

                    # print
                    is_print_iter = int(iterations) % PRINT_ITERATION == 0
                    is_last = epoch == NUM_EPOCHS - 1 and iterations * BATCH_SIZE >= num_examples
                    if is_print_iter or is_last:
                        end_time = time.time()
                        delta_time = end_time - start_time
                        print('delta_time: ', delta_time)
                        start_time = time.time()

                        get_data = [loss_content, loss_style, loss_tv, VGGNetLoss, pred_rescale]
                        return_data, merge_result = sess.run([get_data, merged], feed_dict={content_input: x_batch})

                        # save to tf board
                        writer.add_summary(merge_result, (epoch * num_examples + iterations * BATCH_SIZE) /
                                           (PRINT_ITERATION * BATCH_SIZE))

                        print("step {}/of epoch {}/{}...".format(
                            (epoch * num_examples + iterations * BATCH_SIZE) /
                            (PRINT_ITERATION * BATCH_SIZE), epoch + 1, NUM_EPOCHS),
                            "content_loss: {:.4f}".format(return_data[0]), "style_loss: {:.4f}".format(return_data[1]),
                            "tv_loss: {:.4f}".format(return_data[2]), "VGGNetLoss: {:.4f}".format(return_data[3])
                        )

                        # save img
                        save_img_path = tf_board_path + 'check' + str((epoch * num_examples + iterations * BATCH_SIZE) /
                                                                      (PRINT_ITERATION * BATCH_SIZE))
                        utils.save_image(path=save_img_path + '.png', img=return_data[4][0])
            print('train done')
            # save sess
            saver.save(sess, save_path)
            print('save done')
