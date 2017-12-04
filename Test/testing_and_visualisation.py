import sys
sys.path.insert(0, '../')
import argparse
import numpy as np
from bottleneck import get_bottlenecks_values
from predicting import *
import matplotlib.pyplot as plt
import time
import os
import warnings
import tkinter as tk
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score
from sklearn import decomposition
from sklearn import manifold


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


def get_nb_outliers_found(ground_truth, predictions):
    nb_outliers_found = 0

    for idx, pred in enumerate(predictions):
        if pred == 1 and ground_truth[idx] == 1:
            nb_outliers_found += 1

    return nb_outliers_found


def get_nb_outlier(ground_truth):
    return np.sum(ground_truth)


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


def benchmark_one_class_poluted(main_label_bottlenecks, polution_label_bottlenecks, suptitle=None):
    """

    :param main_label_bottlenecks: Numpy array containing all the bottleneck values of your main label.
    :param polution_label_bottlenecks: Numpy array containing all the bottleneck values of your polution label.
    :param suptitle: String to add as plot suptitles
    :return: Nothing. But will display graph
    """

    if len(polution_label_bottlenecks) > len(main_label_bottlenecks):
        warnings.warn('More polution label than true label, the array is truncated')
        polution_label_bottlenecks = polution_label_bottlenecks[len(main_label_bottlenecks)-1, :]

    # One class polluted
    true_label = np.zeros(len(main_label_bottlenecks))
    nb_point_calculation = 20
    steps_in_calculation = tuple(
        int(len(polution_label_bottlenecks) / nb_point_calculation * i) for i in range(1, nb_point_calculation + 1))

    k_means_fn_accumulator = np.zeros(nb_point_calculation)
    k_means_fp_accumulator = np.zeros(nb_point_calculation)

    birch_fn_accumulator = np.zeros(nb_point_calculation)
    birch_fp_accumulator = np.zeros(nb_point_calculation)

    gaussian_mixture_fn_accumulator = np.zeros(nb_point_calculation)
    gaussian_mixture_fp_accumulator = np.zeros(nb_point_calculation)

    agglomerative_fn_accumulator = np.zeros(nb_point_calculation)
    agglomerative_fp_accumulator = np.zeros(nb_point_calculation)

    nb_cat_bottlenecks = len(main_label_bottlenecks)
    x_axis = [x / (nb_cat_bottlenecks + len(polution_label_bottlenecks)) * 100 for x in
              steps_in_calculation]

    t0 = time.time()

    for idx, i in enumerate(steps_in_calculation):
        ground_true = np.concatenate([true_label, np.ones(i)])
        image_set = np.concatenate((main_label_bottlenecks, polution_label_bottlenecks[:i]))

        predictions = detection_with_kmeans(image_set)
        nb_points = len(predictions)
        k_means_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / nb_points * 100
        k_means_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = detection_with_birch(image_set)
        birch_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / nb_points * 100
        birch_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = detection_with_gaussian_mixture(image_set)
        gaussian_mixture_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / nb_points * 100
        gaussian_mixture_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = detection_with_agglomaritve_clustering(image_set)
        agglomerative_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / nb_points * 100
        agglomerative_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

    print("Finished to get predictions, generated in : ", time.time() - t0)

    plt.figure()
    plt.subplot(121)

    line_k_means, = plt.plot(x_axis, k_means_fn_accumulator, 'ro')
    line_birch, = plt.plot(x_axis, birch_fn_accumulator, 'k^')
    line_gaussian, = plt.plot(x_axis, gaussian_mixture_fn_accumulator, 'bs')
    line_agglomerative, = plt.plot(x_axis, agglomerative_fn_accumulator, 'gx')

    # y=x reference
    max_polution = steps_in_calculation[-1] / nb_points * 100
    plt.plot([0, max_polution], [0, max_polution])

    plt.xlabel('% of pollution')
    plt.ylabel('% of false negative')
    plt.legend([line_k_means, line_birch, line_gaussian, line_agglomerative],
               ['K-means', 'Birch', 'Gaussian Mixture', 'Agglomerative Clustering'], loc='best')

    plt.subplot(122)
    line_k_means, = plt.plot(x_axis, k_means_fp_accumulator, 'ro')
    line_agglomerative, = plt.plot(x_axis, agglomerative_fp_accumulator, 'gx')
    line_gaussian, = plt.plot(x_axis, gaussian_mixture_fp_accumulator, 'bs')
    line_birch, = plt.plot(x_axis, birch_fp_accumulator, 'k^')

    plt.xlabel('% of pollution')
    plt.ylabel('% of false positive')
    plt.legend([line_k_means, line_birch, line_gaussian, line_agglomerative],
               ['K-means', 'Birch', 'Gaussian Mixture', 'Agglomerative Clustering'], loc='best')

    if suptitle:
        plt.suptitle(suptitle)


def benchmark_semi_supervised(main_label_bottlenecks, polution_label_bottlenecks, architecture='mobilenet_1.0_224',
                              pollution_dir='../Cached_pollution', suptitle=None):
    """

    :param main_label_bottlenecks: Numpy array containing all the bottleneck values of your main label.
    :param polution_label_bottlenecks: Numpy array containing all the bottleneck values of your polution label.
    :param architecture: Which architecture used to generate your bottlenecks.
    :param pollution_dir: Location of the directory containing precomputed values for random images.
    :param suptitle: String to add as plot suptitles
    :return: Nothing. But will display graph
    """

    if len(polution_label_bottlenecks) > len(main_label_bottlenecks):
        warnings.warn('More polution label than true label, the array is truncated')
        polution_label_bottlenecks = polution_label_bottlenecks[len(main_label_bottlenecks)-1, :]

    nb_outliers = len(polution_label_bottlenecks)
    ground_true = np.concatenate((np.zeros(len(main_label_bottlenecks)), np.ones(len(polution_label_bottlenecks))))
    image_set = np.concatenate((main_label_bottlenecks, polution_label_bottlenecks))

    nb_point_calculation = 20
    steps_in_calculation = np.linspace(0, .40, nb_point_calculation)

    k_means_tp_accumulator = np.zeros(nb_point_calculation)
    k_means_fp_accumulator = np.zeros(nb_point_calculation)

    birch_tp_accumulator = np.zeros(nb_point_calculation)
    birch_fp_accumulator = np.zeros(nb_point_calculation)

    gaussian_mixture_tp_accumulator = np.zeros(nb_point_calculation)
    gaussian_mixture_fp_accumulator = np.zeros(nb_point_calculation)

    agglomerative_tp_accumulator = np.zeros(nb_point_calculation)
    agglomerative_fp_accumulator = np.zeros(nb_point_calculation)

    t0 = time.time()

    nb_points = len(image_set)

    # Could be simplify and faster with another implementation of semi_supervised_detection that wouldn't rely
    # on grabbing the .npy stored again and again
    for idx, i in enumerate(steps_in_calculation):

        predictions = semi_supervised_detection(image_set, CLUSTERING_METHODS[0], architecture, pollution_dir, i)
        k_means_tp_accumulator[idx] = get_nb_outliers_found(ground_true, predictions) / nb_outliers * 100
        k_means_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = semi_supervised_detection(image_set, CLUSTERING_METHODS[1], architecture, pollution_dir, i)
        birch_tp_accumulator[idx] = get_nb_outliers_found(ground_true, predictions) / nb_outliers * 100
        birch_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = semi_supervised_detection(image_set, CLUSTERING_METHODS[2], architecture, pollution_dir, i)
        gaussian_mixture_tp_accumulator[idx] = get_nb_outliers_found(ground_true, predictions) / nb_outliers * 100
        gaussian_mixture_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

        predictions = semi_supervised_detection(image_set, CLUSTERING_METHODS[3], architecture, pollution_dir, i)
        agglomerative_tp_accumulator[idx] = get_nb_outliers_found(ground_true, predictions) / nb_outliers * 100
        agglomerative_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / nb_points * 100

    print("Finished to get predictions, generated in : ", time.time() - t0)

    # To get back percent
    steps_in_calculation = steps_in_calculation * 100

    plt.figure()
    plt.subplot(121)

    line_k_means, = plt.plot(steps_in_calculation, k_means_tp_accumulator, 'ro')
    line_birch, = plt.plot(steps_in_calculation, birch_tp_accumulator, 'k^')
    line_gaussian, = plt.plot(steps_in_calculation, gaussian_mixture_tp_accumulator, 'bs')
    line_agglomerative, = plt.plot(steps_in_calculation, agglomerative_tp_accumulator, 'gx')

    plt.xlabel('% of added noise')
    plt.ylabel('% of outliers found')
    plt.legend([line_k_means, line_birch, line_gaussian, line_agglomerative],
               ['K-means', 'Birch', 'Gaussian Mixture', 'Agglomerative Clustering'], loc='best')

    plt.subplot(122)
    line_k_means, = plt.plot(steps_in_calculation, k_means_fp_accumulator, 'ro')
    line_agglomerative, = plt.plot(steps_in_calculation, agglomerative_fp_accumulator, 'gx')
    line_gaussian, = plt.plot(steps_in_calculation, gaussian_mixture_fp_accumulator, 'bs')
    line_birch, = plt.plot(steps_in_calculation, birch_fp_accumulator, 'k^')

    plt.xlabel('% of added noise')
    plt.ylabel('% of false positive')
    plt.legend([line_k_means, line_birch, line_gaussian, line_agglomerative],
               ['K-means', 'Birch', 'Gaussian Mixture', 'Agglomerative Clustering'], loc='best')

    if suptitle:
        plt.suptitle(suptitle)


def see_iso_map(bottlenecks, labels, suptitle=None):
    """

    :param bottlenecks:
    :param labels:
    :param suptitle: String to add as plot suptitles
    :return: Nothing, will just plot a scatter plot to show the distribution of our data after dimensionality reduction.
    """

    n_samples, n_features = bottlenecks.shape
    n_neighbors = 25
    n_components = 2
    start_index_outlier = np.where(labels == 1)[0][0]
    alpha_inlier = 0.25

    B_iso = manifold.Isomap(n_neighbors, n_components).fit_transform(bottlenecks)
    B_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(bottlenecks)
    B_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='standard').fit_transform(bottlenecks)
    B_spec = manifold.SpectralEmbedding(n_components=n_components, random_state=42,
                                        eigen_solver='arpack').fit_transform(bottlenecks)

    plt.figure()

    plt.subplot(221)
    plt.scatter(B_iso[:start_index_outlier, 0], B_iso[:start_index_outlier, 1], marker='o', c='b', alpha=alpha_inlier)
    plt.scatter(B_iso[start_index_outlier:, 0], B_iso[start_index_outlier:, 1], marker='^', c='k')
    plt.title("Isomap projection")

    plt.subplot(222)
    inlier_scatter = plt.scatter(B_lle[:start_index_outlier, 0], B_lle[:start_index_outlier, 1], marker='o', c='b',
                                 alpha=alpha_inlier)
    outlier_scatter = plt.scatter(B_lle[start_index_outlier:, 0], B_lle[start_index_outlier:, 1], marker='^', c='k')
    plt.legend([inlier_scatter, outlier_scatter], ['Inliers', 'Outliers'], loc='lower left')
    plt.title("Locally Linear Embedding")

    plt.subplot(223)
    plt.scatter(B_pca[:start_index_outlier, 0], B_pca[:start_index_outlier, 1], marker='o', c='b', alpha=alpha_inlier)
    plt.scatter(B_pca[start_index_outlier:, 0], B_pca[start_index_outlier:, 1], marker='^', c='k')
    plt.title("Principal Components projection")

    plt.subplot(224)
    plt.scatter(B_spec[:start_index_outlier, 0], B_spec[:start_index_outlier, 1], marker='o', c='b', alpha=alpha_inlier)
    plt.scatter(B_spec[start_index_outlier:, 0], B_spec[start_index_outlier:, 1], marker='^', c='k')
    plt.title("Spectral embedding")

    if suptitle:
        plt.suptitle(suptitle)


def load_bottleneck(image_dir, bottlenick_dir, architecture_chosen='MobileNet_1.0_224', model_location='../model'):
    """
    Function that will look if your label as already been transformed to bottleneck given a model, will register it or load,
    making it easy to tinker, but be carefull you will have to delete de file if you change the content of the image dir.

    :param image_dir: List of image dir location you want to test with
    :param bottlenick_dir: Where you want your bottleneck to be saved
    :param architecture_chosen: Which model architecture to use to generate your bottlenecks.
            Ranging from the incepetion to the MobileNet model
    :param model_location: Where the model will be downloaded
    :return: A dictionary, containing the bottleneck values, the key being the label of the image_dir.
    """
    bottleneck_values = {}

    saved_values = os.listdir(bottlenick_dir)

    for directory in image_dir:
        label = directory.strip('./')
        entry = label + '_' + architecture_chosen + '.npy'
        path = os.path.join(bottlenick_dir, entry)

        if entry not in saved_values:
            values = get_bottlenecks_values(directory, architecture_chosen,
                                            model_dir=model_location)
            np.save(path, values)
            print('Creating bottleneck for :', path)
        else:
            values = np.load(path)
            print('Loading bottlenecks from :', path)

        bottleneck_values[label] = values

    return bottleneck_values


def stich_images(shape, images):
    """
        Given a shape you want for your images, will create a mozaic, the size of your screen to visualize multiple
        image at once.

    :param shape: Tuple containing the shape of a single image.
    :param images: List containing all your images.
    :return: Nothing, will pop the created image, with your default image viewer.
    """

    root = tk.Tk()

    width_screen = root.winfo_screenwidth()
    height_screen = root.winfo_screenheight()

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


def see_false_positive(image_set, predictions, ground_truth):
    """
    Construct and display images that were mislabeled by our classifier
    :param image_set: Entire set of images
    :param predictions:  Vector of labels given by the classifier
    :param ground_truth:  Vector of labels of the data
    """

    width_images = 280
    height_images = 180

    false_positives = [im.reshape([width_images, height_images]) for idx, im in enumerate(image_set) if
                       predictions[idx] == 1 and ground_truth[idx] == 0]
    sent_images = 0
    image_sent_one_go = 30
    while sent_images + image_sent_one_go < len(false_positives):
        stich_images((width_images, height_images), false_positives[sent_images:sent_images + image_sent_one_go])
        sent_images += image_sent_one_go

    stich_images((width_images, height_images), false_positives[sent_images:])


def see_false_negative(image_set, predictions, ground_truth):
    """
    Construct and display images that were mislabeled by our classifier
    :param image_set: Entire set of images
    :param predictions:  Vector of labels given by the classifier
    :param ground_truth:  Vector of labels of the data
    """

    width_images = 280
    height_images = 180

    false_negatives = [im.reshape([width_images, height_images]) for idx, im in enumerate(image_set) if
                       predictions[idx] == 0 and ground_truth[idx] == 1]
    sent_images = 0
    image_sent_one_go = 30
    while sent_images + image_sent_one_go < len(false_negatives):
        stich_images((width_images, height_images), false_negatives[sent_images:sent_images + image_sent_one_go])
        sent_images += image_sent_one_go

    stich_images((width_images, height_images), false_negatives[sent_images:])


def main(_):

    # Just add your own labels, with distinct directory and update; the variables
    image_dir = ['./Cat', './Dog', './Flag', './Noise', './Graphic_card', './Not_graphic_card', './Knife', './Not_knife']

    bottlenecks = load_bottleneck(image_dir, './Saved_bottlenecks', architecture_chosen=FLAGS.architecture)

    if FLAGS.test == "benchmark_classifier":

        benchmark_one_class_poluted(bottlenecks['Cat'], bottlenecks['Noise'], 'Cat labels, polluted by random images')
        benchmark_one_class_poluted(bottlenecks['Flag'], bottlenecks['Noise'], 'Flag labels, polluted by random images')
        benchmark_one_class_poluted(bottlenecks['Cat'], bottlenecks['Dog'], 'Cat labels, polluted by dog images')

        plt.show()

    elif FLAGS.test == "benchmark_semi_supervised":
        benchmark_semi_supervised(bottlenecks['Graphic_card'], bottlenecks['Not_graphic_card'],
                                  suptitle='Results on google image querry \'graphic card\' (408 inliers, 174 outliers)')

        benchmark_semi_supervised(bottlenecks['Knife'], bottlenecks['Not_knife'],
                                  suptitle='Results on google image querry \'knife\' (532 inliers, 34 outliers)')


        benchmark_semi_supervised(bottlenecks['Flag'], bottlenecks['Noise'][: int(len(bottlenecks['Flag']) * 0.05), :],
                                  suptitle='Flag labels, with 5 % of random images')

        benchmark_semi_supervised(bottlenecks['Cat'], bottlenecks['Dog'][: int(len(bottlenecks['Cat']) * 0.1), :],
                                  suptitle='Cat labels, with 10 % of dog images.')

        benchmark_semi_supervised(bottlenecks['Cat'], bottlenecks['Noise'][: int(len(bottlenecks['Cat']) * 0.05), :],
                                  suptitle='Cat labels, with 5 % of random images.')

        plt.show()

    elif FLAGS.test == "isomap":
        # Close label test
        X = np.concatenate((bottlenecks['Cat'], bottlenecks['Dog'][: int(len(bottlenecks['Cat']) * 0.05), :]))
        Y = np.concatenate((np.zeros(len(bottlenecks['Cat'])), np.ones(int(len(bottlenecks['Dog']) * 0.05))))
        see_iso_map(X, Y, 'Cat labels, with 5 % of dog pictures.')

        # Label but noise to help
        X = np.concatenate((bottlenecks['Dog'][: int(len(bottlenecks['Dog']) * 0.05), :], bottlenecks['Noise']))
        X = np.concatenate((bottlenecks['Cat'], X))
        Y = np.concatenate((np.zeros(len(bottlenecks['Cat'])), np.ones(int(len(bottlenecks['Dog']) * 0.05) + int(len(bottlenecks['Noise'])))))
        see_iso_map(X, Y, 'Cat labels, with 5 % of Dog Noise and 15 % of Random Images')

        X = np.concatenate((bottlenecks['Cat'], bottlenecks['Noise']))
        Y = np.concatenate((np.zeros(len(bottlenecks['Cat'])), np.ones(len(bottlenecks['Noise']))))
        see_iso_map(X, Y, 'Cat labels, with 15 % of Random Images')

        plt.show()

    else:
        print('Wrong argument passed, with --test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help="""\
           Wich test you want to be executed.
           Either :
           - isomap
           - benchmark_classifier
           - benchmark_semi_supervised\
           """
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='MobileNet_1.0_224',
        help="""\
                  Which model architecture to use. 'inception_v3' is the most accurate, but
                  also the slowest, and you might it the dimensionality curse with small sample, given the bottleneck layer
                  is the biggest. For faster results, chose a MobileNet with the form :
                  'mobilenet_<parameter size>_<input_size>[_quantized]'.
                  For example, 'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
                  pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
                  less accurate, but smaller and faster network that's 920 KB on disk and
                  takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
                  for more information on Mobilenet.\
                  """)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)

