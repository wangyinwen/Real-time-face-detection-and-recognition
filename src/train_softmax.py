"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def main(args):
    network = importlib.import_module(args.model_def)  # --model_def models.inception_resnet_v1

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)  # 日志的地址
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)  # 训练模型存储地址
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    # 读取训练图片的路径
    if args.filter_filename:
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename),
                                   args.filter_percentile, args.filter_min_nrof_images_per_class)
    # train set中是元组，一个标签，对应一个或多个路径
    # 所以这里返回的nrof_classes是样本类别个数
    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        # global_step对应的是全局批的个数，根据这个参数可以更新学习率
        global_step = tf.Variable(0, trainable=False)
        # global_step经常在滑动平均，学习速率变化的时候需要用到,系统会自动更新这个参数的值，从1开始。

        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        # range_size的大小和样本个数一样
        range_size = array_ops.shape(labels)[0]
        # QueueRunner：保存的是队列中的入列操作，保存在一个list当中，其中每个enqueue运行在一个线程当中
        # range_input_producer：返回的是一个队列，队列中有打乱的整数，范围是从0到range size
        # range_size大小和总的样本个数一样
        # 并将一个QueueRunner添加到当前图的QUEUE_RUNNER集合中
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)
        # 返回的是一个出列操作，每次出列一个epoch中需要用到的样本个数
        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')
        # tf.placeholder：用于得到传递进来的真实的训练样本：
        # 不必指定初始值，可在运行时，通过 Session.run的函数的feed_dict参数指定；
        # 学习率
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        # 批大小
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        # 用于判断是训练还是测试
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # 图像路径
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        # 图像标签
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')
        # 上面的队列是一个样本索引的出列队列，只用来出列
        # 用来每次出列一个epoch中需要用到的样本
        # 这里是第二个队列，这个队列用来入列，队列的大小为10万，每个元素的大小为shapes=[(1,), (1,)],
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        # 这时一个入列的操作，这个操作将在session run的时候用到
        # 每次入列的是image_paths_placeholder, labels_placeholder对
        # 注意，这里只有2个队列，一个用来出列打乱的元素序号
        # 一个根据对应的需要读取指定的文件
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        # 4个线程
        # 在不同的线程中入列不同的tensor需要入列的样本在images_and_labels中
        # 创建的线程个数为len(images_and_labels)
        # 在这里应该是有4个入列线程，因为images_and_labels只append4次
        # 线程i入列张量images_and_labels[i]
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
        # batch_join的作用是创建样本批，用于批处理
        # capacity控制着用于增长队列的预取的个数
        # batch_size用于出列的一个批的大小
        # enqueue_many表示一次出列多个数据
        # shapes：样本的shape，默认根据images_and_labels[i]推断出来
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        # 创建网络图:除了全连接层和损失层
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss --center_loss_factor 1e-2
        if args.center_loss_factor > 0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            # 将center加入到名字为tf.GraphKeys.REGULARIZATION_LOSSES的集合当中来
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
        # 将指数衰减应用到学习率上
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        # 将softmax和交叉熵一起做,得到最后的损失函数,提高效率
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        # 根据REGULARIZATION_LOSSES返回一个收集器中所收集的值的列表
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # 我们选择L2-正则化来实现这一点，L2正则化将网络中所有权重的平方和加到损失函数。如果模型使用大权重，则对应重罚分，并且如果模型使用小权重，则小罚分。
        # 这就是为什么我们在定义权重时使用了regularizer参数，并为它分配了一个l2_regularizer。这告诉了TensorFlow要跟踪
        # l2_regularizer这个变量的L2正则化项（并通过参数reg_constant对它们进行加权）。
        # 所有正则化项被添加到一个损失函数可以访问的集合——tf.GraphKeys.REGULARIZATION_LOSSES。
        # 将所有正则化损失的总和与先前计算的triplet_loss相加，以得到我们的模型的总损失。
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # 确定优化方法并求根据损失函数求梯度,在这里面,每更新一次参数,global_step会加1
        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

        set_A_vars = [v for v in tf.trainable_variables() if v.name.startswith('InceptionResnetV1')]  # 加载官方预训练模型
        saver_set_A = tf.train.Saver(set_A_vars, max_to_keep=200)  # 加载官方预训练模型，一共加三行代码
        # Create a saver
        # 创建一个saver用于保存或从内存中恢复一个模型参数
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # 能够在gpu上分配的最大内存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver_set_A.restore(sess, pretrained_model)  # 加载官方预训练模型
                # saver.restore(sess, pretrained_model)#可以加载以前训练的softmax模型

            # Training and validation loop，max_nrof_epochs = 80，epoch_size = 默认default=1000
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                # step可以看做是全局的批处理个数
                step = sess.run(global_step, feed_dict=None)
                # epoch_size是一个epoch中批的个数
                # 这个epoch是全局的批处理个数除以一个epoch中批的个数得到epoch
                epoch = step // args.epoch_size  # 双斜杠为取整的5//3 = 1
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
                      labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses,
                      args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already,存储模型
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                if args.lfw_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder,
                             embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size,
                             args.lfw_nrof_folds, log_dir, step, summary_writer)
    return model_dir


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del (filtered_dataset[i])

    return filtered_dataset


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    # 每次训练时，出列一次，出列的个数是args.batch_size*args.epoch_size,，即一个epoch中图像的数量
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    # 将对应的文件路径以及标签入列
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        # print('ok')
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        # 每100次,将结果保存到log中去
        # 然后运行一次,是一次运行:求loss,根据loss,运行train_op来对参数进行优化
        # 计算REGULARIZATION_LOSSES只是为了打印出来查看正则损失的值,而实际整个训练过程这个值已经包含在total loss中了
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder,
             embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0, len(image_paths)), 1)
    image_paths_array = np.expand_dims(np.array(image_paths), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame) * 2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb

    assert np.array_equal(lab_array, np.arange(
        nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=96)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')

    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
