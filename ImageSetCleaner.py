import sys
from skimage.io import imread_collection
import cv2
from Gui_Image_Selector import MainWindow
import argparse
from PyQt5.QtWidgets import QApplication
from Predicting import *
from File_Processing import *

# Check :
# https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/
# http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#sphx-glr-auto-examples-covariance-plot-outlier-detection-py

# http://www.sciencedirect.com/science/article/pii/S0167947307002204
# https://www.researchgate.net/publication/224576812_Using_one-class_SVM_outliers_detection_for_verification_of_collaboratively_tagged_image_training_sets


def load_image(path, width, length):

    path = os.path.abspath(path) + '\\'

    col = imread_collection(path + '*.jpg')
    col = np.array(col)

    # Check for alpha
    for idx, elem in enumerate(col):
        if not elem.shape[2] == 3:
            col[idx] = cv2.cvtColor(elem, cv2.COLOR_RGBA2RGB)

    col = np.array([cv2.resize(im, (width, length)) for im in col])

    return col


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def flatten_set(image_set):
    """
    :param image_set as a 3 D numpy array : n_imagges x width x length
    :return: image_set as a 2 D numpy array : n_images x (width x length)
    """

    return np.reshape(image_set, [image_set.shape[0], -1])


def verify_input(_):
    """
    This method will check the values given by the user.
    :param _: Parser
    :return: Nothing
    """

    if not os.path.exists(FLAGS.image_dir):
        raise AssertionError('Image directory not found.')

    if FLAGS.pollution_percent <0 or FLAGS.pollution_percent > 40:
        raise AssertionError('Wrong value for pollution. Should be between 0 and 40.')

    if FLAGS.clustering_method not in CLUSTERING_METHODS:
        raise AssertionError('Wrong clustering method given.')

    if FLAGS.processing == 'move' and FLAGS.relocation_dir is None:
        raise AssertionError('You need to specify a relocation directory, if you want to move the detected images.')


def main(_):

    verify_input(FLAGS)

    image_set = get_bottlenecks_values(FLAGS.image_dir, FLAGS.architecture, FLAGS.model_dir)

    predictions = semi_supervised_detection(image_set, FLAGS.clustering_method, FLAGS.architecture,
                                            FLAGS.pollution_dir, float(FLAGS.pollution_percent) / 100)

    image_paths = get_image_paths(FLAGS.image_dir, predictions)

    if len(image_paths) == 0:
        raise AssertionError('No outlier detected in the directory.')

    if FLAGS.processing == 'gui':
        app = QApplication([])
        window = MainWindow(FLAGS.image_dir, image_set, image_paths, FLAGS.clustering_method, FLAGS.architecture, FLAGS.pollution_dir,
                            FLAGS.pollution_percent)
        sys.exit(app.exec_())
        # TODO : Wtf here ?

    elif FLAGS.processing == 'move':
        if FLAGS.relocation_dir:
            ensure_directory(FLAGS.relocation_dir)
            move_images(FLAGS.relocation_dir, image_paths)
        else:
            # TODO : NEED TO KILL IT I.e si il a mit quelque chose
            print('')

    elif FLAGS.processing == 'delete':
        delete_images(image_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help="""\
        Path to the folder of labeled images.\
        """
    )
    parser.add_argument(
        '--clustering_method',
        type=str,
        default='feature_agglo',
        help="""\
            Choose your method of clustering, between k-means, mean_shift, agglomerattive clustering, birch
            More info : http://scikit-learn.org/stable/modules/clustering.html\
            """
    )
    parser.add_argument(
        '--processing',
        type=str,
        default='gui',
        help="""\
            Select the method you will process the detected image :
            gui : Will open a window that will let you pick which image to delete and which to keep
            move : Will move the detected images, to your desired location, with the option --relocation_dir
            delete : Will delete all the detected images\
        """
    )
    parser.add_argument(
        '--relocation_dir',
        type=str,
        default=None,
        help="""\
            Directory where you want the detected images to be moved.
        """
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='MobileNet_1.0_224',
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
        '--pollution_dir',
        type=str,
        default='./Cached_pollution',
        help="""\
        Path to cached pollution bottlenecks.\
        """
    )
    parser.add_argument(
        '--pollution_percent',
        type=float,
        default=25,
        help="""\
            Give the percentage of pre-computed noisy / polluted bottlenecks, from random images to help the clustering
            algorithm get a good fit.\
        """
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
