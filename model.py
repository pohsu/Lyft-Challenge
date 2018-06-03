import os
import sys
import random
import math
import numpy as np
import tensorflow as tf
import scipy
from glob import glob
import re
from imgaug import augmenters as iaa

def build_gen(data_folder):
  def get_batches_fn(batch_size):
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    label_paths = {os.path.basename(path): path for path in glob(os.path.join(data_folder, 'CameraSeg', '*.png'))}
    random.shuffle(image_paths)
    for batch_i in range(0, len(image_paths), batch_size):
      images = []
      labels = []
      for image_file in image_paths[batch_i:batch_i+batch_size]:
        label_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imread(image_file)
        label = scipy.misc.imread(label_file)
        #[102:518]
        images.append(image)
        labels.append(label)
      yield np.array(images), np.array(labels)
  return get_batches_fn

def build_gen_val(data_folder):
  def get_batches_fn(batch_size):
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    label_paths = {os.path.basename(path): path for path in glob(os.path.join(data_folder, 'CameraSeg', '*.png'))}
    random.shuffle(image_paths)
    for batch_i in range(0, len(image_paths), batch_size):
      images = []
      labels = []
      for image_file in image_paths[batch_i:batch_i+batch_size]:
        label_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imread(image_file)
        label = scipy.misc.imread(label_file)
        #[102:518]
        images.append(image)
        labels.append(label)
      yield np.array(images), np.array(labels)
  return get_batches_fn

def optimize(nn_last_layer, correct_label, learning_rate, num_classes, weights):
    class_weights = tf.constant(weights,dtype=tf.float32)
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name = 'logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    weights = tf.reduce_sum(class_weights * correct_label, axis=1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels = correct_label, logits =logits)
    weighted_loss = tf.reduce_mean(unweighted_losses * weights)
    softmax = tf.nn.softmax(nn_last_layer, name = 'softmax')
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add(weighted_loss, tf.reduce_sum(reg_losses),name = 'cross_entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(total_loss, name = 'train_op')
    return logits, train_op, total_loss

def build_graph(sess, model_path, correct_label, learning_rate, num_classes, L2, weights):
  saver = tf.train.import_meta_graph(model_path+'.meta')
  saver.restore(sess, model_path)
  graph = tf.get_default_graph()
  input_image = graph.get_tensor_by_name('input_image:0')
  C4 = graph.get_tensor_by_name('resnet_v2_101/block3/unit_11/bottleneck_v2/add:0') # 32,50,1024
  C3 = graph.get_tensor_by_name('resnet_v2_101/block2/unit_3/bottleneck_v2/add:0') # 64, 100, 512
  C2 = graph.get_tensor_by_name('resnet_v2_101/block1/unit_2/bottleneck_v2/add:0') # 128, 200, 256
  C4 = tf.nn.relu(C4)
  C3 = tf.nn.relu(C3)
  C2 = tf.nn.relu(C2)

  FCN_16s = tf.layers.conv2d(C4, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_16s_conv2d')
  FCN_16s = tf.layers.conv2d_transpose(FCN_16s, num_classes, 4, 2, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_16s_conv2d_transpose')

  FCN_8s = tf.layers.conv2d(C3, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_8s_conv2d')
  FCN_8s = tf.add(FCN_8s, FCN_16s, name = 'FCN_8s_add')

  FCN_8s = tf.layers.conv2d_transpose(FCN_8s, num_classes, 4, 2, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_8s_conv2d_transpose')

  FCN_4s = tf.layers.conv2d(C2, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_4s_conv2d')
  FCN_4s = tf.add(FCN_4s, FCN_8s, name = 'FCN_4s_add')

  output = tf.layers.conv2d_transpose(FCN_4s, num_classes, 8, 4, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2), name = 'FCN_output')
  nn_last_layer = tf.identity(output, name="nn_last_layer")

  logits, train_op, total_losses = optimize(nn_last_layer, correct_label, learning_rate, num_classes, weights)
  return (train_op, logits, total_losses, nn_last_layer, input_image, correct_label, learning_rate, L2)

def build_graph_inference(sess, model_path, num_classes):
  L2 = 1e-4
  saver = tf.train.import_meta_graph(model_path+'.meta')
  graph = tf.get_default_graph()
  C4 = graph.get_tensor_by_name('resnet_v2_101/block3/unit_11/bottleneck_v2/add:0') # 32,50,1024
  C3 = graph.get_tensor_by_name('resnet_v2_101/block2/unit_3/bottleneck_v2/add:0') # 64, 100, 512
  C2 = graph.get_tensor_by_name('resnet_v2_101/block1/unit_2/bottleneck_v2/add:0') # 128, 200, 256
  C4 = tf.nn.relu(C4)
  C3 = tf.nn.relu(C3)
  C2 = tf.nn.relu(C2)

  FCN_16s = tf.layers.conv2d(C4, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_16s_conv2d')
  FCN_16s = tf.layers.conv2d_transpose(FCN_16s, num_classes, 4, 2, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_16s_conv2d_transpose')

  FCN_8s = tf.layers.conv2d(C3, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_8s_conv2d')
  FCN_8s = tf.add(FCN_8s, FCN_16s, name = 'FCN_8s_add')

  FCN_8s = tf.layers.conv2d_transpose(FCN_8s, num_classes, 4, 2, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_8s_conv2d_transpose')

  FCN_4s = tf.layers.conv2d(C2, num_classes, (1,1),
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2),
                          name = 'FCN_4s_conv2d')
  FCN_4s = tf.add(FCN_4s, FCN_8s, name = 'FCN_4s_add')

  output = tf.layers.conv2d_transpose(FCN_4s, num_classes, 8, 4, padding = 'same',
                          kernel_regularizer = tf.contrib.layers.l2_regularizer(L2), name = 'FCN_output')
  nn_last_layer = tf.identity(output, name="nn_last_layer")

  softmax = tf.nn.softmax(nn_last_layer, name = 'softmax')

def train_nn_valid(sess, epochs, batch_size, lrn_rate, tensors, l2_rate, gen_paths, SAVE_DIR_TEST):
    train_gen = build_gen(gen_paths[0])
    valid_gen = build_gen_val(gen_paths[1])
    train_op, logits, total_loss, nn_last_layer, input_image, correct_label, learning_rate, L2 = tensors
    graph = tf.get_default_graph()
    is_training = graph.get_tensor_by_name('is_training:0')
    softmax = graph.get_tensor_by_name('softmax:0')
    b = 0
    train_loss_pre = 0;
    train_loss = 0;
    for epoch in range(epochs):
        if  epoch > 1 and train_loss > train_loss_pre:
            lrn_rate *= 0.9
            print('learning rate updates: {0:6g}'.format(lrn_rate))
        train_loss_pre = train_loss
        #training set
        train_losses = []
        train_tp = 0
        train_se = 0
        train_re = 0
        train_corrects = 0
        train_pixels = 0
        train_counts = 0
        for image, label in train_gen(batch_size):
            feed_dict = {input_image: image,
                         correct_label: label,
                         learning_rate: lrn_rate,
                         L2: l2_rate,
                         is_training: True}
            _, loss, out = sess.run([train_op, total_loss, softmax], feed_dict=feed_dict)
            out = np.argmax(out,axis = 3)
            car_mask = np.where(out==1,1,0)
            label_mask = label[:,:,:,1]
            train_count = image.shape[0]
            train_losses.append(loss*train_count)
            train_tp += np.sum(car_mask & label_mask)
            train_se += np.sum(car_mask)
            train_re += np.sum(label_mask)
            train_corrects += np.sum(np.equal(out,np.argmax(label,axis = 3)))
            train_pixels += label.shape[0]*label.shape[1]*label.shape[2]
            train_counts += train_count
            print ('Traning with.....epoch: {0} and batch: {1}'.format(epoch+1, b+1), end='\r')
            b +=1
        train_P = train_tp/train_se
        train_R = train_tp/train_re
        train_accuracy = train_corrects/train_pixels
        train_loss = np.sum(train_losses)/train_counts

        # Validation set
        val_losses = []
        val_tp = 0
        val_se = 0
        val_re = 0
        val_corrects = 0
        val_pixels = 0
        val_counts = 0
        for image, label in valid_gen(8):
            feed_dict = {input_image: image,
                         correct_label: label,
                         L2: l2_rate,
                         is_training: False}
            loss, out = sess.run([total_loss, softmax], feed_dict=feed_dict)
            out = np.argmax(out,axis = 3)
            car_mask = np.where(out==1,1,0)
            label_mask = label[:,:,:,1]
            val_count = label.shape[0]
            val_losses.append(loss*val_count)
            val_tp += np.sum(car_mask & label_mask)
            val_se += np.sum(car_mask)
            val_re += np.sum(label_mask)
            val_corrects += np.sum(np.equal(out,np.argmax(label,axis = 3)))
            val_pixels += label.shape[0]*label.shape[1]*label.shape[2]
            val_counts += val_count
        val_P = val_tp/val_se
        val_R = val_tp/val_re
        val_accuracy = val_corrects/val_pixels
        val_loss = np.sum(val_losses)/val_counts
        if epoch % 2 == 0:
            save_model_path = os.path.join(SAVE_DIR_TEST, 'ckpt')
            saver = tf.train.Saver()
            saver.save(sess, save_model_path)
            print (' ')
            print ('Saving Model....')
        print("Epoch {0}: loss {1:.3g}, TP {2:.3g}, VP {3:.3g}, TR {4:.3g}, VR {5:.3g} | TA {6:.3g}, VA {7:.3g}" \
            .format(epoch+1, train_loss, train_P, val_P, train_R, val_R, train_accuracy, val_accuracy))
        save_txt_path = os.path.join(SAVE_DIR_TEST, 'log.txt')
        with open(save_txt_path, 'a') as f:
            print("Epoch {0}: lrn_rate {1:3g}, loss {2:.3g}, TP {3:.3g}, VP {4:.3g}, TR {5:.3g}, VR {6:.3g} | TA {7:.3g}, VA {8:.3g}" \
            .format(epoch+1, lrn_rate, train_loss, train_P, val_P, train_R, val_R, train_accuracy, val_accuracy), file = f)

def load_train_tensors(graph):
  input_image = graph.get_tensor_by_name('input_image:0')
  correct_label = graph.get_tensor_by_name('correct_label:0')
  learning_rate = graph.get_tensor_by_name('learning_rate:0')
  train_op = graph.get_operation_by_name("train_op")
  logits = graph.get_tensor_by_name('logits:0')
  total_loss = graph.get_tensor_by_name('cross_entropy_loss:0')
  nn_last_layer = graph.get_tensor_by_name('nn_last_layer:0')
  L2 = graph.get_tensor_by_name('L2:0')
  return (train_op, logits, total_loss, nn_last_layer, input_image, correct_label, learning_rate, L2)
