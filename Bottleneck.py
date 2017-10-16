from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.

    Args:
      architecture: Name of a model architecture.

    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized

    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
        but found '%s' for architecture '%s'""",
                version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
       but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pb'
        else:
            model_base_name = 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }


def maybe_download_and_extract(data_url, dest_directory):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
      data_url: Web location of the tar file containing the pretrained model.
      dest_directory: Where our model will be stored
    """

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                        'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
      model_info: Dictionary containing information about the model architecture.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def create_image_list(image_dir):
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None

    file_list = []
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    change = 0
    # TODO : Connerie de méthode prend deux fois en ocmpte les mêmes images
    for extension in extensions:
        file_glob = os.path.join(image_dir,  '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))
        if not len(file_list) == change:
            change = len(file_list)
            print('ding')
            print(extension)
    # images = [x[0] for x in gfile.Walk(image_dir)]

    if len(file_list) < 20:
        tf.logging.warning(
            'WARNING: Folder has less than 20 images, which may cause issues.')

    # Ensure no duplicate
    return sorted(set(file_list))


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image





# TODO : Start here by this end
def get_bottleneck_values(bottleneck_path, image_lists, label_name, index,
                          image_dir, category, sess, jpeg_data_tensor,
                          decoded_image_tensor, resized_input_tensor,
                          bottleneck_tensor):

    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))

    return bottleneck_values


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values



def get_all_bottlenecks(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    model_dest_directory = './model/'

    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1

    # Set up the pre-trained graph.
    maybe_download_and_extract(model_info['data_url'], model_dest_directory)

    graph, bottleneck_tensor, resized_image_tensor = (
        create_model_graph(model_info))

    # Look at the folder structure, and create lists of all the images.
    image_paths = create_image_list(FLAGS.image_dir)

    print(image_paths)
    bottlenecks = np.zeros((len(image_paths), model_info['bottleneck_tensor_size']))

    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        for idx, image_p in enumerate(image_paths):
            print(image_p)
            image_data = gfile.FastGFile(image_p, 'rb').read()
            bottlenecks[idx, :] = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                                                        resized_image_tensor, bottleneck_tensor)



    print(bottlenecks.shape)
    print(bottlenecks)


def main(_):
    get_all_bottlenecks(_)
    # with tf.Session(graph=graph) as sess:
    #     # Set up the image decoding sub-graph.
    #     jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
    #         model_info['input_width'], model_info['input_height'],
    #         model_info['input_depth'], model_info['input_mean'],
    #         model_info['input_std'])
    #
    #     else:
    #         # We'll make sure we've calculated the 'bottleneck' image summaries and
    #         # cached them on disk.
    #         cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
    #                           FLAGS.bottleneck_dir, jpeg_data_tensor,
    #                           decoded_image_tensor, resized_image_tensor,
    #                           bottleneck_tensor, FLAGS.architecture)
    #
    #
    #     # Merge all the summaries and write them out to the summaries_dir
    #     merged = tf.summary.merge_all()
    #     train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
    #                                          sess.graph)
    #
    #     validation_writer = tf.summary.FileWriter(
    #         FLAGS.summaries_dir + '/validation')
    #
    #     # Set up all our weights to their initial default values.
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #
    #     # Run the training for as many cycles as requested on the command line.
    #     for i in range(FLAGS.how_many_training_steps):
    #         # Get a batch of input bottleneck values, either calculated fresh every
    #         # time with distortions applied, or from the cache stored on disk.
    #         if do_distort_images:
    #             (train_bottlenecks,
    #              train_ground_truth) = get_random_distorted_bottlenecks(
    #                 sess, image_lists, FLAGS.train_batch_size, 'training',
    #                 FLAGS.image_dir, distorted_jpeg_data_tensor,
    #                 distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
    #         else:
    #             (train_bottlenecks,
    #              train_ground_truth, _) = get_random_cached_bottlenecks(
    #                 sess, image_lists, FLAGS.train_batch_size, 'training',
    #                 FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
    #                 decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
    #                 FLAGS.architecture)
    #         # Feed the bottlenecks and ground truth into the graph, and run a training
    #         # step. Capture training summaries for TensorBoard with the `merged` op.
    #         train_summary, _ = sess.run(
    #             [merged, train_step],
    #             feed_dict={bottleneck_input: train_bottlenecks,
    #                        ground_truth_input: train_ground_truth})
    #         train_writer.add_summary(train_summary, i)
    #
    #         # Every so often, print out how well the graph is training.
    #         is_last_step = (i + 1 == FLAGS.how_many_training_steps)
    #         if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
    #             train_accuracy, cross_entropy_value = sess.run(
    #                 [evaluation_step, cross_entropy],
    #                 feed_dict={bottleneck_input: train_bottlenecks,
    #                            ground_truth_input: train_ground_truth})
    #             tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
    #                             (datetime.now(), i, train_accuracy * 100))
    #             tf.logging.info('%s: Step %d: Cross entropy = %f' %
    #                             (datetime.now(), i, cross_entropy_value))
    #             validation_bottlenecks, validation_ground_truth, _ = (
    #                 get_random_cached_bottlenecks(
    #                     sess, image_lists, FLAGS.validation_batch_size, 'validation',
    #                     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
    #                     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
    #                     FLAGS.architecture))
    #             # Run a validation step and capture training summaries for TensorBoard
    #             # with the `merged` op.
    #             validation_summary, validation_accuracy = sess.run(
    #                 [merged, evaluation_step],
    #                 feed_dict={bottleneck_input: validation_bottlenecks,
    #                            ground_truth_input: validation_ground_truth})
    #             validation_writer.add_summary(validation_summary, i)
    #             tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
    #                             (datetime.now(), i, validation_accuracy * 100,
    #                              len(validation_bottlenecks)))
    #
    #         # Store intermediate results
    #         intermediate_frequency = FLAGS.intermediate_store_frequency
    #
    #         if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
    #             and i > 0):
    #             intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
    #                                       'intermediate_' + str(i) + '.pb')
    #             tf.logging.info('Save intermediate result to : ' +
    #                             intermediate_file_name)
    #             save_graph_to_file(sess, graph, intermediate_file_name)
    #
    #     # We've completed all our training, so run a final test evaluation on
    #     # some new images we haven't used before.
    #     test_bottlenecks, test_ground_truth, test_filenames = (
    #         get_random_cached_bottlenecks(
    #             sess, image_lists, FLAGS.test_batch_size, 'testing',
    #             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
    #             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
    #             FLAGS.architecture))
    #     test_accuracy, predictions = sess.run(
    #         [evaluation_step, prediction],
    #         feed_dict={bottleneck_input: test_bottlenecks,
    #                    ground_truth_input: test_ground_truth})
    #     tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
    #                     (test_accuracy * 100, len(test_bottlenecks)))
    #
    #     if FLAGS.print_misclassified_test_images:
    #         tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
    #         for i, test_filename in enumerate(test_filenames):
    #             if predictions[i] != test_ground_truth[i].argmax():
    #                 tf.logging.info('%70s  %s' %
    #                                 (test_filename,
    #                                  list(image_lists.keys())[predictions[i]]))
    #
    #     # Write out the trained graph and labels with the weights stored as
    #     # constants.
    #     save_graph_to_file(sess, graph, FLAGS.output_graph)
    #     with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
    #         f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )

    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )

    parser.add_argument(
        '--architecture',
        type=str,
        default='inception_v3',
        help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
