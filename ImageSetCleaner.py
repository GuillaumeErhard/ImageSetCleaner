from random import randint
import numpy as np
from PIL import Image
import os
from win32api import GetSystemMetrics
from Saliency import get_saliency_ft
import scipy.misc
from sklearn.metrics import accuracy_score
from sklearn import cluster
from sklearn import mixture
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import svm
import cv2
import time
from sklearn import cluster, datasets


# Check :
# https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/
# http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#sphx-glr-auto-examples-covariance-plot-outlier-detection-py

# http://www.sciencedirect.com/science/article/pii/S0167947307002204
# https://www.researchgate.net/publication/224576812_Using_one-class_SVM_outliers_detection_for_verification_of_collaboratively_tagged_image_training_sets


# TODO : Change how image_list ist done and directly add vector of np array one after the other
def load_image(path, width, length):
    images_location = os.listdir(path)
    image_list = []

    for i in range(len(images_location)):
        im = Image.open(path + images_location[i])
        image_list.append(np.array(im.resize((width, length))))

    image_list = np.array(image_list)

    return image_list


def load_saliency(path, width, length):
    images_location = os.listdir(path)
    image_list = []

    for i in range(len(images_location)):
        im = get_saliency_ft(path + images_location[i])
        image_list.append(scipy.misc.imresize(im, (width, length)))

    image_list = np.array(image_list)

    return image_list


def stich_images(shape, images):
    width_screen = GetSystemMetrics(0)
    height_screen = GetSystemMetrics(1)

    nb_images = len(images)

    images_in_line_max = width_screen // shape[0]
    images_in_column_max = height_screen // shape[1]

    #images = [img.resize(shape, Image.ANTIALIAS) for img in images]

    stitched_image = Image.new('RGB', (width_screen, height_screen))

    for idx_line in range(images_in_line_max):
        for idx_column in range(images_in_column_max):

            if idx_line * images_in_column_max + idx_column >= nb_images:
                break

            img_with_pil = Image.fromarray(images[idx_line * images_in_column_max + idx_column])
            stitched_image.paste(im=img_with_pil, box=(idx_line * shape[0], idx_column * shape[1]))

    stitched_image.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def outlier_detection_check(image_set, ground_true):
    t0 = time.time()

    # Rgb2gray
    image_set = np.array([rgb2gray(im_rgb) for im_rgb in image_set])
    # Saliency
    # np.array([rgb2gray(im_rgb) for im_rgb in image_set])

    # Flatten image
    image_set = np.reshape(image_set, [image_set.shape[0], -1])


    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralBiclustering.html
    clf = cluster.SpectralBiclustering(n_clusters=2, n_components=11, n_best=11)
    clf.fit(image_set)

    predictions = clf.row_labels_
    print(predictions)
    sum_row_labels = np.sum(predictions)
    majority_class = int(np.sum(predictions) / (len(predictions) / 2 - 1))
    print(predictions)
    print(majority_class)

    if majority_class == 1:
        ground_true = 1 - ground_true

    print(ground_true)

    print('Accuracy :', accuracy_score(ground_true, clf.row_labels_))

    print('Time taken for classification :', time.time() - t0)


def main():

    width = 280
    length = 180

    dir_location = ['./Test_cluster_small/', './Test_cluster_few_to_many/']
    # 0 inlier, 1 outlier
    ground_true = [np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0]), np.array([np.zeros(154), 1])]

    for idx, dir in enumerate(dir_location):
        image_set = load_image(dir, width, length)

        outlier_detection_check(image_set, ground_true[idx])

        t0 = time.time()
        image_set = load_saliency(dir, width, length)
        print('Time to get saliency :', time.time() - t0)

        stich_images((width, length), image_set)

        outlier_detection_check(image_set, ground_true[idx])



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