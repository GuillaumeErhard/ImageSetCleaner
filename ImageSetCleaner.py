import numpy as np
from PIL import Image
import os
# from win32api import GetSystemMetrics
from Saliency import get_saliency_ft, get_saliency_mbd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from skimage.io import imread_collection
from sklearn import cluster
from sklearn import mixture
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import svm
import cv2
import time
from sklearn import cluster, datasets
from Bottleneck import get_bottlenecks_values


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


def stich_images(shape, images):
    width_screen = GetSystemMetrics(0)
    height_screen = GetSystemMetrics(1)

    nb_images = len(images)

    images_in_line_max = width_screen // shape[0]
    images_in_column_max = height_screen // shape[1]

    stitched_image = Image.new('RGB', (width_screen, height_screen))

    for idx_line in range(images_in_line_max):
        for idx_column in range(images_in_column_max):

            if idx_line * images_in_column_max + idx_column >= nb_images:
                break
            img_with_pil = Image.fromarray(images[idx_line * images_in_column_max + idx_column])
            stitched_image.paste(im=img_with_pil, box=(idx_line * shape[0], idx_column * shape[1]))

    stitched_image.show()


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


def get_scoring(ground_truth, predictions):
    """

    :param predictions: Vector of labels given by the classifier
    :param ground_truth: Vector of labels of the data
    :return: accuracy, precision, recall
    """
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = precision_score(ground_truth, predictions)

    return accuracy, precision, recall


def see_false_positive(image_set, predictions, ground_truth):
    """
    Construct and display images that were mislabeled by our classifier
    :param image_set: Entire set of images
    :param predictions:  Vector of labels given by the classifier
    :param ground_truth:  Vector of labels of the data
    """

    false_positives = [im.reshape([280, 180]) for idx, im in enumerate(image_set) if
                       predictions[idx] == 1 and ground_truth[idx] == 0]
    sent_images = 0
    image_sent_one_go = 30
    while sent_images + image_sent_one_go < len(false_positives):
        stich_images((280, 180), false_positives[sent_images:sent_images + image_sent_one_go])
        sent_images += image_sent_one_go

    stich_images((280, 180), false_positives[sent_images:])


def see_false_negative(image_set, predictions, ground_truth):
    """
    Construct and display images that were mislabeled by our classifier
    :param image_set: Entire set of images
    :param predictions:  Vector of labels given by the classifier
    :param ground_truth:  Vector of labels of the data
    """

    false_positives = [im.reshape([280, 180]) for idx, im in enumerate(image_set) if
                       predictions[idx] == 0 and ground_truth[idx] == 1]
    sent_images = 0
    image_sent_one_go = 30
    while sent_images + image_sent_one_go < len(false_positives):
        stich_images((280, 180), false_positives[sent_images:sent_images + image_sent_one_go])
        sent_images += image_sent_one_go

    stich_images((280, 180), false_positives[sent_images:])


def detection_with_agglomaritve_clustering(image_set):
    """
    Really good if the classes you are analyzing are close to what the network learned.

    :param image_set
    :return: predictions vector

     N.B : The detector breaks with a full black image.
    """

    # http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
    clf = cluster.AgglomerativeClustering(n_clusters=2, )
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


def detection_with_meanshift(image_set):
    """

    :param image_set:
    :return: predictions vector
    """

    clf = cluster.AgglomerativeClustering(n_clusters=2)

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


def get_nb_false_negative(ground_truth, predictions):
    nb_false_neg = 0

    for idx, pred in enumerate(predictions):
        if pred == 0 and ground_truth[idx] == 1:
            nb_false_neg += 1

    return nb_false_neg


def get_nb_false_positive(ground_truth, predictions):
    nb_false_pos = 0

    for idx, pred in enumerate(predictions):
        if pred == 1 and ground_truth[idx] == 0:
            nb_false_pos += 1

    return nb_false_pos


def get_nb_outlier(ground_truth):
    return np.sum(ground_truth)


def main():
    width = 280
    length = 180

    dir_location = ['./Test_cluster_no_outlier/', './Test_cluster_small/', './Test_cluster_1/', './Test_cluster_2/',
                    './Test_cat_vs_dog/']
    # 0 inlier, 1 outlier
    ground_true = [np.zeros(75), np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0]), np.concatenate([[1, 1, 1, 1], np.zeros(52)]),
                   np.concatenate([np.zeros(82), [1, 1, 1, 1, 1]]), np.concatenate([np.zeros(1050), np.ones(100)])]

    dir_location = dir_location[-2:]
    ground_true = ground_true[-2:]

    for idx, dir in enumerate(dir_location):
        # print("Currenty at :", dir)
        # image_set = load_image(dir, width, length)
        #
        # # Rgb2gray
        # image_set = np.array([rgb2gray(im_rgb) for im_rgb in image_set])
        # image_set = flatten_set(image_set)
        # #stich_images((width, length), image_set)
        #
        # predictions = detection_with_agglomaritve_clustering(image_set)
        #
        # accuracy, precision, recall = get_scoring(predictions, ground_true[idx])
        #
        # t0 = time.time()
        # image_set = load_saliency(dir, width, length)
        # image_set = flatten_set(image_set)
        # print('Time to get saliency :', time.time() - t0)
        #
        # predictions = detection_with_agglomaritve_clustering(image_set)
        # accuracy, precision, recall = get_scoring(predictions, ground_true[idx])
        #
        # print('Accuracy :', str(accuracy)[:4], 'Precision', str(precision)[:6], 'Recall', str(recall)[:6])

        t0 = time.time()
        # image_set = get_all_bottlenecks(dir, architecture='inception_v3', model_dir='./model/')
        image_set = get_bottlenecks_values(dir, architecture='MobileNet_1.0_224', model_dir='./model/')

        print('Time to get bottleneck :', time.time() - t0)

        predictions = detection_with_agglomaritve_clustering(image_set)
        accuracy, precision, recall = get_scoring(ground_true[idx], predictions)

        print('Aglomaritive_clustering', 'Accuracy :', str(accuracy)[:4], 'Precision', str(precision)[:6], 'Recall',
              str(recall)[:6])
        print('Nb false negative :', get_nb_false_negative(ground_true[idx], predictions), 'Nb false positive :',
              get_nb_false_positive(ground_true[idx], predictions), 'Out of :', get_nb_outlier(ground_true[idx]),
              'outliers')

        predictions = detection_with_kmeans(image_set)
        accuracy, precision, recall = get_scoring(ground_true[idx], predictions)

        print('K-means', 'Accuracy :', str(accuracy)[:4], 'Precision', str(precision)[:6], 'Recall', str(recall)[:6])
        print('Nb false negative :', get_nb_false_negative(ground_true[idx], predictions), 'Nb false positive :',
              get_nb_false_positive(ground_true[idx], predictions), 'Out of :', get_nb_outlier(ground_true[idx]),
              'outliers')


        # clf = mixture.BayesianGaussianMixture(n_components=2)
        # clf.fit(image_set)
        # print(clf.predict(image_set))

        #
        # outliers_fraction = 0.25
        #
        # classifiers = {
        #     "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
        #                                      kernel="rbf", gamma=0.1),
        #     "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        #     "Isolation Forest": IsolationForest(max_samples=n_samples,
        #                                         contamination=outliers_fraction,
        #                                         random_state=rng),
        #     "Local Outlier Factor": LocalOutlierFactor(
        #         n_neighbors=35,
        #         contamination=outliers_fraction)}
        #
        # for i, (clf_name, clf) in enumerate(classifiers.items()):
        #     # fit the data and tag outliers
        #     if clf_name == "Local Outlier Factor":
        #         y_pred = clf.fit_predict(X)
        #         scores_pred = clf.negative_outlier_factor_
        #     else:
        #         clf.fit(X)
        #         scores_pred = clf.decision_function(X)
        #         y_pred = clf.predict(X)
        #     threshold = stats.scoreatpercentile(scores_pred,
        #                                         100 * outliers_fraction)
        #     n_errors = (y_pred != ground_truth).sum()
        #     # plot the levels lines and the points
        #     if clf_name == "Local Outlier Factor":
        #         # decision_function is private for LOF
        #         Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        #     else:
        #         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #     Z = Z.reshape(xx.shape)


if __name__ == '__main__':
    main()
