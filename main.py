#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
import project_tests as tests

from distutils.version import LooseVersion


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # creating result layer
    vgg_layer7_out_result = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    transpose_layer1 = tf.layers.conv2d_transpose(vgg_layer7_out_result, num_classes,
                                              kernel_size=[4, 4],
                                              strides=(2, 2), padding="SAME",
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 conv of layer 4 and sum with transpose_layer_1 then transpose of the result
    # 1x1 conv of layer 4 after scaling (details in paper folder)
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name="pool4_out_scaled")
    vgg_layer4_out_result = tf.layers.conv2d(pool4_out_scaled, num_classes, 1, strides=(1, 1),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # sum with transpose_layer_1
    skip_connection_1_layer = tf.add(transpose_layer1, vgg_layer4_out_result)
    # transpose of the result
    transpose_layer2 = tf.layers.conv2d_transpose(skip_connection_1_layer, num_classes,
                                              kernel_size=[4, 4],
                                              strides=(2, 2), padding="SAME",
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1x1 conv of layer 3 and sum with transpose_layer_2 then transpose of the result
    # 1x1 conv of layer 3 after scaling it (details in paper folder)
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name="pool3_out_scaled")
    vgg_layer3_out_result = tf.layers.conv2d(pool3_out_scaled, num_classes, 1, strides=(1, 1),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # sum with transpose_layer_2
    skip_connection_2_layer = tf.add(transpose_layer2, vgg_layer3_out_result)
    # transpose of the result
    transpose_layer3 = tf.layers.conv2d_transpose(skip_connection_2_layer, num_classes,
                                              kernel_size=[16, 16],
                                              strides=(8, 8), padding="SAME",
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return transpose_layer3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    reg_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss+reg_losses)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # training params
    keep_prob_value = 0.5
    learning_rate_value = 0.0005

    sess.run(tf.global_variables_initializer())
    print("training started!")
    mean_loss_per_epoch = []
    for epoch in range(1, epochs+1):
        epoch_loss = []
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={
                                   input_image: image,
                                   correct_label: label,
                                   keep_prob: keep_prob_value,
                                   learning_rate: learning_rate_value
                               })
            epoch_loss.append(loss)
            print("epoch: {}, loss: = {:3f}".format(epoch, loss))

        mean_loss_per_epoch.append(sum(epoch_loss)/len(epoch_loss))

    with open("./report_resources/data/{}-{}-{}-{}.txt".format(
            epochs, batch_size, learning_rate_value, keep_prob_value), "w+") as f:
        for l in mean_loss_per_epoch:
            f.write("%s\n" % l)

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # training parameters
        epochs = 10
        batch_size = 8

        correct_label = tf.placeholder(tf.int8, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Building the NN using load_vgg, layers, and optimize function
        vgg_path = os.path.join(data_dir, 'vgg')
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # training NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
