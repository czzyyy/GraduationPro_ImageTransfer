import numpy as np
import tensorflow as tf
from ImageTransferWeb.algorithms.slow_real import vgg19
import time
from ImageTransferWeb.algorithms.slow_real import transfer_utils as utils
from ImageTransferWeb.algorithms.slow_real.closed_form_matting import getLaplacian
import math

# weights
CONTENT_WEIGHT = 5.0e0
STYLE_WEIGHT = 1.0e2
TV_WEIGHT = 1.0e-3
AFFINE_WEIGHT = 1.0e4

# train para
LEARNING_RATE = 1.0
PRINT_ITERATION = 100
MAX_ITER = 8000


def load_seg(content_seg, style_seg):
    # para has been resized and is [0-1]
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']

    def _extract_mask(seg, color_str):
        # h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i]))
                                                                 , 0), -1))
        color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0)
                                                , -1))

    return color_content_masks, color_style_masks


def gram_matrix(activations):
    # height = tf.shape(activations)[1]
    # width = tf.shape(activations)[2]
    # num_channels = tf.shape(activations)[3]
    # gram = tf.transpose(activations, [0, 3, 1, 2])
    # gram = tf.reshape(gram, [num_channels, width * height])
    # gram = tf.matmul(gram, gram, transpose_b=True)
    _, height, width, filter_num = [i.value for i in activations.shape]
    features = tf.reshape(activations, [1, height * width, filter_num])
    gram = tf.matmul(tf.transpose(features, perm=[0, 2, 1]), features) / tf.to_float(height * width * filter_num)
    return gram


def content_loss(const_layer, var_layer, weight):
    b, h, w, c = [i.value for i in var_layer.shape]
    loss = weight * (2 * tf.nn.l2_loss(var_layer - const_layer) / tf.to_float(b * h * w * c))
    return loss
    # return tf.reduce_mean(tf.squared_difference(const_layer, var_layer)) * weight


def single_style_loss(CNN_structure, const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    # layer_count = float(len(const_layers))
    layer_index = 0

    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
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
                _, h, w = [i.value for i in gram_matrix_var.shape]
                diff_style_sum = 2 * tf.nn.l2_loss(gram_matrix_var - gram_matrix_const) / tf.to_float(h * w) * content_mask_mean
                # diff_style_sum = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)
    return loss_styles


def total_variation_loss(output, weight):
    tv_loss = tf.reduce_sum((output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) *
                            (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) +
                            (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) *
                            (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight


def single_affine_loss(output, M, weight):
    loss_affine = 0.0
    height, width, _ = [i.value for i in output.shape]
    for Vc in tf.unstack(output, axis=-1):
        Vc = tf.reshape(Vc, [height * width])
        loss_affine += tf.reduce_mean(tf.matmul(tf.expand_dims(Vc, 0), tf.sparse_tensor_dense_matmul(
            M, tf.expand_dims(Vc, -1))))

    return loss_affine * weight


def train(content_path, style_path, save_path, content_seg_path='./data/segmentation/in7.png', style_seg_path='./data/segmentation/tar7.png'):
    # tf_board_path = './tf_board/'
    # content_seg_path = './data/segmentation/in7.png'
    # style_seg_path = './data/segmentation/tar7.png'

    # prepare input images
    with tf.name_scope('inputs'):
        # content
        content_image = np.array(utils.load_image_resize(content_path), dtype=np.float32)  # [0-1]
        content_width, content_height = content_image.shape[1], content_image.shape[0]
        # style
        style_image = np.array(utils.load_image_resize(style_path), dtype=np.float32)  # [0-1]
        style_width, style_height = style_image.shape[1], style_image.shape[0]
        print('style shape: ', style_image.shape)
        print('content shape: ', content_image.shape)

    # cal matting
    with tf.name_scope('matting'):
        M = tf.to_float(getLaplacian(content_image))

    # reshape add a dim
    content_image = content_image.reshape((1, content_height, content_width, 3)).astype(np.float32)
    style_image = style_image.reshape((1, style_height, style_width, 3)).astype(np.float32)

    # load segs
    with tf.name_scope('load_segs'):
        content_seg = np.array(utils.load_image_resize(content_seg_path, content_height, content_width)
                               , dtype=np.float32)  # [0-1]
        style_seg = np.array(utils.load_image_resize(style_seg_path, style_height, style_width)
                             , dtype=np.float32)  # [0-1]
        content_masks, style_masks = load_seg(content_seg, style_seg)

    # init image
    with tf.name_scope('init_image'):
        noise_image = np.random.uniform(0, 255, size=(1, content_height, content_width, 3)).astype(np.float32)
        init_image = noise_image * 0.6 + (content_image * 255.0) * (1 - 0.6)
        # make it trainable
        input_image = tf.Variable(init_image)

    with tf.name_scope('constant_value'):
        # content
        vgg_content_const = vgg19.Vgg19()
        vgg_content_const.build_without_fc(tf.constant(content_image))
        with tf.Session() as content_sess:
            content_fv = content_sess.run(vgg_content_const.conv4_2)
        content_layer_const = tf.constant(content_fv)
        # style
        vgg_style_const = vgg19.Vgg19()
        vgg_style_const.build_without_fc(tf.constant(style_image))
        style_layers_const = [vgg_style_const.conv1_1, vgg_style_const.conv2_1, vgg_style_const.conv3_1,
                              vgg_style_const.conv4_1, vgg_style_const.conv5_1]
        with tf.Session() as style_sess:
            style_fvs = style_sess.run(style_layers_const)
        style_layers_const = [tf.constant(fv) for fv in style_fvs]

    with tf.name_scope('variable_value'):
        # trainable input image features
        vgg_var = vgg19.Vgg19()
        vgg_var.build_without_fc(input_image / 255.0)

    # feature layers
    style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
    content_layer_var = vgg_var.conv4_2

    # The whole CNN structure to downsample mask
    layer_structure_all = [layer.name for layer in vgg_var.get_all_layers()]

    with tf.name_scope('loss'):
        # Content Loss
        loss_content = content_loss(content_layer_const, content_layer_var, CONTENT_WEIGHT)

        # Style Loss
        loss_styles_list = single_style_loss(layer_structure_all, style_layers_const, style_layers_var
                                             , content_masks, style_masks, STYLE_WEIGHT)
        loss_style = 0.0
        for loss in loss_styles_list:
            loss_style += loss

        # Affine Loss
        loss_affine = single_affine_loss(tf.squeeze(input_image / 255.0), M, AFFINE_WEIGHT)

        # Total Variational Loss
        loss_tv = total_variation_loss(input_image, TV_WEIGHT)

        # all loss
        VGGNetLoss = loss_content + loss_tv + loss_style
        # tf.summary.scalar('loss_content', loss_content)
        # tf.summary.scalar('loss_style', loss_style)
        # tf.summary.scalar('loss_tv', loss_tv)
        # tf.summary.scalar('VGGNetLoss', VGGNetLoss)

    with tf.name_scope('optimizer_Adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-08)
        VGG_grads = optimizer.compute_gradients(VGGNetLoss)
        b, g, r = tf.unstack(tf.squeeze(input_image, [0]) / 255., axis=-1)
        b_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(b), [-1]), -1)), [content_width, content_height]))
        g_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(g), [-1]), -1)), [content_width, content_height]))
        r_gradient = tf.transpose(tf.reshape(2 * tf.sparse_tensor_dense_matmul(M, tf.expand_dims(tf.reshape(tf.transpose(r), [-1]), -1)), [content_width, content_height]))
        Matting_grad = tf.expand_dims(tf.stack([b_gradient, g_gradient, r_gradient], axis=-1), 0) / 255. * AFFINE_WEIGHT
        VGGMatting_grad = [(VGG_grad[0] + Matting_grad, VGG_grad[1]) for VGG_grad in VGG_grads]
        train_op = optimizer.apply_gradients(VGGMatting_grad)

    with tf.name_scope('train'):
        with tf.Session() as sess:
            # record the time
            start_time = time.time()

            sess.run(tf.global_variables_initializer())
            # # merge summary
            # merged = tf.summary.merge_all()
            # # choose dir
            # writer = tf.summary.FileWriter(tf_board_path, sess.graph)
            min_loss, best_image = float("inf"), None
            for i in range(1, MAX_ITER + 1):
                _, loss_content_, loss_style_, loss_tv_, loss_affine_, overall_loss_, output_image_ = sess.run([
                    train_op, loss_content, loss_style, loss_tv, loss_affine, VGGNetLoss, input_image])
                if i % PRINT_ITERATION == 0:
                    end_time = time.time()
                    delta_time = end_time - start_time
                    print('delta_time: ', delta_time)
                    start_time = time.time()

                    # save to tf board
                    # merge_result = sess.run(merged)
                    # writer.add_summary(merge_result, i)

                    print("step {}/of {}...".format(i, MAX_ITER), "loss_content: {:.4f}".format(loss_content_)
                          , "loss_style: {:.4f}".format(loss_style_), "loss_tv: {:.4f}".format(loss_tv_)
                          , "loss_affine: {:.4f}".format(loss_affine_), "overall_loss_: {:.4f}".format(overall_loss_))

                    # save img
                    # save_img_path = tf_board_path + 'check'
                    # utils.save_image(path=save_img_path + str(i) + '.png', img=np.uint8(np.clip(output_image_[0],
                    #                                                                             0, 255.0)))

                if overall_loss_ < min_loss:
                    min_loss, best_image = overall_loss_, output_image_

            # end train
            output_save_path = save_path + 'output' + content_path.split('/')[-1]
            utils.save_image(path=output_save_path, img=np.uint8(np.clip(best_image[0], 0, 255.0)))
            print('save: ' + output_save_path)
            print('train done')
            print("min_loss: {:.4f}".format(min_loss))
            return best_image[0]
