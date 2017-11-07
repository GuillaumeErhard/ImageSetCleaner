import numpy as np
from PIL import Image
import os
from Saliency import get_saliency_ft, get_saliency_mbd
from skimage.io import imread_collection
import cv2
import time
from sklearn import cluster, datasets
from Bottleneck import get_bottlenecks_values
import argparse

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


def load_saliency(path, width, length):
    collection = load_image(path, width, length)

    return np.array([get_saliency_ft(img) for img in collection])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def flatten_set(image_set):
    """
    :param image_set as a 3 D numpy array : n_imagges x width x length
    :return: image_set as a 2 D numpy array : n_images x (width x length)
    """

    return np.reshape(image_set, [image_set.shape[0], -1])


def normalize_predictions(predictions):
    """
    We take the assumption that the data set contains less than 50 % of outlier.
    Given that the classifier, gives the label 0 and 1 for the same data
    randomly. We make sure that an inlier is described as a 0.

    :param predictions:  A 1 D numpy array with the predictions of our detector
    :return: predictions: A 1 D numpy array with the predictions of our detector, cleaned
    """
    if np.sum(predictions) > (len(predictions) / 2 - 1):
        predictions = 1 - predictions

    return predictions


def detection_with_agglomaritve_clustering(image_set):
    """
    Really good if the classes you are analyzing are close to what the network learned.

    :param image_set
    :return: predictions vector

     N.B : The detector breaks with a full black image.
    """

    # http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
    clf = cluster.AgglomerativeClustering(n_clusters=2)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_kmeans(image_set):
    """
    Fast, but might not be able to map great for nonlinear separation of classes.

    :param image_set
    :return: predictions vector
    """

    clf = cluster.KMeans(n_clusters=2, random_state=42)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_feature_agglo(image_set):
    """

    :param image_set:
    :return: predictions vector
    """

    clf = cluster.FeatureAgglomeration(n_clusters=2)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_birch(image_set):
    """

    :param image_set:
    :return: predictions vector
    """

    clf = cluster.Birch(n_clusters=2)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def grabbing_pollution(architecture, pollution_points, pollution_dir):
    # TODO Complete architecture
    """

    :param architecture:
    :param pollution_points: Number of points desired by the user to be added to the data values.
    :param pollution_dir: Location of the directory containing the precomputed values.
    :return: An int that contains how many pollution bottleneck we have and a numpy array containing, bottlencks of
            random images.
    """

    saved_values = os.listdir(pollution_dir)

    entry = 'Noise_' + architecture + '.npy'
    path = os.path.join(pollution_dir, entry)

    if entry not in saved_values:
        print('Pollution label not found')
        pollution_bottlenecks = np.array()
        nb_bottlenecks_to_return = 0
    else:
        print('Loading bottlenecks from :', path)
        pollution_bottlenecks = np.load(path)

        nb_bottlenecks = pollution_bottlenecks.shape[0]

        if nb_bottlenecks > pollution_points:
            nb_bottlenecks_to_return = pollution_points
        else:
            print('Problem, not enough polluted bottlenecks have been pre computed')
            nb_bottlenecks_to_return = nb_bottlenecks

        pollution_bottlenecks = pollution_bottlenecks[:nb_bottlenecks_to_return, :]

    return nb_bottlenecks_to_return, pollution_bottlenecks


def semi_supervised_detection(image_dir, clustering_method, architecture, model_dir, pollution_dir,
                              pollution_percent=0.20):
    # TODO : Compléter commentaire des deux fonctions
    # TODO : Enlever les print
    """

    :param image_dir:
    :param clustering_method: Wich algorithm is used to get a prediction on the data.
    :param architecture:
    :param model_dir:
    :param pollution_percent: Fraction of pollution added to our image values.
    :param pollution_dir:
    :return: A prediction vector, that is altered by a given amount of random data, to hopefuly get a better performance.
    """

    image_set = get_bottlenecks_values(image_dir, architecture, model_dir)
    pollution_points = int(image_set.shape[0] * pollution_percent)
    pollution_points, pollution_set = grabbing_pollution(architecture, pollution_points, pollution_dir)
    percent_of_pollution = pollution_points / image_set.shape[0]

    print('We use a pollution of :', percent_of_pollution * 100, '%')
    #print(image_set.shape, pollution_set.shape)
    synthetic_set = np.concatenate((image_set, pollution_set))
    #print(synthetic_set.shape)
    clustering_methods = ('kmeans', 'birch', 'feature_agglo', 'agglomerative')

    if clustering_method not in clustering_methods:
        # TODO : Add a shutdown du programme
        print('')

    if clustering_method == clustering_methods[0]:
        predictions = detection_with_kmeans(synthetic_set)
    elif clustering_method == clustering_methods[1]:
        predictions = detection_with_birch(synthetic_set)
    elif clustering_method == clustering_methods[2]:
        predictions = detection_with_feature_agglo(synthetic_set)
    elif clustering_method == clustering_methods[3]:
        predictions = detection_with_agglomaritve_clustering(synthetic_set)

    #print(predictions.shape)
    predictions = predictions[:-pollution_points]
    #print(predictions.shape)
    return predictions


def main(_):
    predictions = semi_supervised_detection(FLAGS.image_dir, FLAGS.clustering_method, FLAGS.architecture,
                                            FLAGS.model_dir, FLAGS.pollution_percent, FLAGS.pollution_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help="""\
        Path to folders of labeled images.\
        """
    )
    parser.add_argument(
        '--clustering_method',
        type=str,
        default='mean_shift',
        help="""\
            Choose your method of clustering, between k-means, mean_shift, agglomerattive clustering, birch
            More info : http://scikit-learn.org/stable/modules/clustering.html
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
        default=0.2,
        help="""\
            Give the percentage of pre-computed noisy / polluted bottlenecks, from random images to help the clustering
            algorithm get a good fit.\
        """
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
