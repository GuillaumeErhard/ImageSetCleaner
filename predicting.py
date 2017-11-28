import numpy as np
import os
from sklearn import cluster
from sklearn import mixture


CLUSTERING_METHODS = ('kmeans', 'birch', 'gaussian_mixture', 'agglomerative_clustering')


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


def detection_with_kmeans(image_set):
    """
    Fast, but might not be able to map great for nonlinear separation of classes.

    :param image_set: The bottleneck values of the relevant images.
    :return: Predictions vector
    """

    clf = cluster.KMeans(n_clusters=2)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_birch(image_set):
    """

    :param image_set: The bottleneck values of the relevant images.
    :return: Predictions vector
    """

    # The branching_factor, might be fine tune for better results
    clf = cluster.Birch(n_clusters=2)

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_gaussian_mixture(image_set):
    """

    :param image_set: The bottleneck values of the relevant images.
    :return: Predictions vector
    """

    # Might achieve, better results by initializing weights, or means, given we know when we introduce noisy labels
    clf = mixture.GaussianMixture(n_components=2)

    clf.fit(image_set)

    predictions = clf.predict(image_set)
    predictions = normalize_predictions(predictions)

    return predictions


def detection_with_agglomaritve_clustering(image_set):
    """
    Really good if the classes you are analyzing are close to what the network learned.

    :param image_set: The bottleneck values of the relevant images.
    :return: Predictions vector

     N.B : The detector breaks with a full black image.
    """

    # http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
    clf = cluster.AgglomerativeClustering(n_clusters=2, affinity="l2", linkage="complete")

    clf.fit(image_set)

    predictions = clf.labels_
    predictions = normalize_predictions(predictions)

    return predictions


def grabbing_pollution(architecture, pollution_dir, pollution_points):
    """
    This function that will see if the pollution directory exist, and try to look a .npy file following the right naming
    scheme.

    :param architecture: Which architecture to use to generate your bottlenecks.
    :param pollution_dir: Location of the directory containing precomputed values for random images.
    :param pollution_points: Number of points desired by the user to be added to the data values.
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
        pollution_bottlenecks = np.load(path)

        nb_bottlenecks = pollution_bottlenecks.shape[0]

        if nb_bottlenecks > pollution_points:
            nb_bottlenecks_to_return = pollution_points
        else:
            print('Problem, not enough polluted bottlenecks have been pre computed')
            nb_bottlenecks_to_return = nb_bottlenecks

        pollution_bottlenecks = pollution_bottlenecks[:nb_bottlenecks_to_return, :]

    return nb_bottlenecks_to_return, pollution_bottlenecks


def semi_supervised_detection(image_set, clustering_method, architecture, pollution_dir,
                              pollution_percent=0.20):
    """
    This function will assemble the values of the image directory, and from random images, to perform a clustering
    on those, and will return the predictions, only on image bottlenecks.

    :param image_set: The bottleneck values of the relevant images.
    :param clustering_method: Which algorithm is used to get a prediction on the data.
    :param architecture: Which architecture used to generate your bottlenecks.
    :param pollution_dir: Location of the directory containing precomputed values for random images.
    :param pollution_percent: Fraction of pollution added to our image values.
    :return: A prediction vector, that is altered by a given amount of random data, to hopefuly get a better performance.
    """

    pollution_points = int(image_set.shape[0] * pollution_percent)
    pollution_points, pollution_set = grabbing_pollution(architecture, pollution_dir, pollution_points)
    percent_of_pollution = pollution_points / image_set.shape[0]

    print('We use a pollution of :', percent_of_pollution * 100, '%')
    synthetic_set = np.concatenate((image_set, pollution_set))

    if clustering_method == CLUSTERING_METHODS[0]:
        predictions = detection_with_kmeans(synthetic_set)
    elif clustering_method == CLUSTERING_METHODS[1]:
        predictions = detection_with_birch(synthetic_set)
    elif clustering_method == CLUSTERING_METHODS[2]:
        predictions = detection_with_gaussian_mixture(synthetic_set)
    elif clustering_method == CLUSTERING_METHODS[3]:
        predictions = detection_with_agglomaritve_clustering(synthetic_set)

    if pollution_points > 0:
        predictions = predictions[:-pollution_points]

    return predictions
