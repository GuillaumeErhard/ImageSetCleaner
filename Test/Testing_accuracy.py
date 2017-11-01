import numpy as np
from Bottleneck import get_bottlenecks_values
from ImageSetCleaner import detection_with_kmeans, detection_with_agglomaritve_clustering, detection_with_meanshift, \
    detection_with_birch
import matplotlib.pyplot as plt
import time
import os
import warnings
import tinker


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


def benchmark_one_class_poluted(main_label_bottlenecks, polution_label_bottlenecks):
    """

    :param main_label_bottlenecks: Numpy array containing all the bottleneck values of your main label.
    :param polution_label_bottlenecks: Numpy array containing all the bottleneck values of your polution label.
    :param architecture_chosen: Which model architecture to use. Ranging from the incepetion to the MobileNet model
    :param model_location: Where the model will be downloaded
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

    spectral_fn_accumulator = np.zeros(nb_point_calculation)
    spectral_fp_accumulator = np.zeros(nb_point_calculation)

    mean_shift_fn_accumulator = np.zeros(nb_point_calculation)
    mean_shift_fp_accumulator = np.zeros(nb_point_calculation)

    birch_fn_accumulator = np.zeros(nb_point_calculation)
    birch_fp_accumulator = np.zeros(nb_point_calculation)

    nb_cat_bottlenecks = len(main_label_bottlenecks)
    x_axis = [x / (nb_cat_bottlenecks + len(polution_label_bottlenecks)) * 100 for x in
              steps_in_calculation]

    t0 = time.time()

    for idx, i in enumerate(steps_in_calculation):
        ground_true = np.concatenate([true_label, np.ones(i + 1)])
        image_set = np.concatenate((main_label_bottlenecks, polution_label_bottlenecks[:i]))

        predictions = detection_with_kmeans(image_set)
        k_means_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100
        k_means_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100

        predictions = detection_with_agglomaritve_clustering(image_set)
        spectral_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100
        spectral_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100

        predictions = detection_with_meanshift(image_set)
        mean_shift_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / (
            nb_cat_bottlenecks + i) * 100
        mean_shift_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / (
            nb_cat_bottlenecks + i) * 100

        predictions = detection_with_birch(image_set)
        birch_fn_accumulator[idx] = get_nb_false_negative(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100
        birch_fp_accumulator[idx] = get_nb_false_positive(ground_true, predictions) / (nb_cat_bottlenecks + i) * 100

    print("Finished to get predictions, generated in : ", time.time() - t0)

    line_k_means, = plt.plot(x_axis, k_means_fn_accumulator, 'ro')
    line_spectral, = plt.plot(x_axis, spectral_fn_accumulator, 'gx')
    line_mean_shift, = plt.plot(x_axis, mean_shift_fn_accumulator, 'bs')
    line_birch, = plt.plot(x_axis, birch_fn_accumulator, 'k^')

    plt.xlabel('% of pollution')
    plt.ylabel('% of false positive')
    plt.legend([line_k_means, line_spectral, line_mean_shift, line_birch],
               ['k-means', 'Spectral Clustering', 'Mean Shift', 'Birch'], loc='best')

    plt.figure()

    line_k_means, = plt.plot(x_axis, k_means_fp_accumulator, 'ro')
    line_spectral, = plt.plot(x_axis, spectral_fp_accumulator, 'gx')
    line_mean_shift, = plt.plot(x_axis, mean_shift_fp_accumulator, 'bs')
    line_birch, = plt.plot(x_axis, birch_fp_accumulator, 'k^')

    plt.xlabel('% of pollution')
    plt.ylabel('% of false negative')
    plt.legend([line_k_means, line_spectral, line_mean_shift, line_birch],
               ['k-means', 'Spectral Clustering', 'Mean Shift', 'Birch'], loc='best')

    plt.show()


def benchmark_spectral(main_label, pollution_labels):
    """

        :param main_label_dir: Numpy array containing all the bottleneck values of your main label.
        :param polution_label_directory:  Numpy array containing all the bottleneck values of your polution label.
        :param architecture_chosen: Which model architecture to use. Ranging from the incepetion to the MobileNet model
        :param model_location: Where the model will be downloaded
        :return: Nothing. But will display graph, and info in the console
        """

    true_label = np.zeros(len(main_label_bottlenecks))

    nb_point_calculation = 10
    steps_in_calculation = tuple(
        int(len(polution_label_bottlenecks) / nb_point_calculation * i) for i in range(1, nb_point_calculation + 1))

    tuned_parameters = [{'n_clusters' : [2], }]


def semi_supervised(main_label, pollution_labels, synthetic_pollution):
    """
        This function is to test my hypothesis, given, the graph that we need a minimum of pollution, to have great result.
        So we will add fake pollution for our classifier, and take them out afterward.

    :param main_label: Numpy array containing all the bottleneck values of your main label.
    :param pollution_labels: Numpy array containing all the bottleneck values of your polution label.
    :param synthetic_pollution: Numpy array containing synthetic pollution, that will be as random as possible
    :return: Nothing. But will display graph, and info in the console
    """
    print('TODO')

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


def main():
    image_dir = ['./Cat', './Dog', './Flag', './Noise']

    bottlenecks = load_bottleneck(image_dir, './Saved_bottlenecks')

    benchmark_one_class_poluted(bottlenecks['Cat'], bottlenecks['Noise'])

    bottlenecks = load_bottleneck(image_dir, './Saved_bottlenecks', architecture_chosen = 'inception_v3')

    benchmark_one_class_poluted(bottlenecks['Cat'], bottlenecks['Noise'])

    see_false_negative()

if __name__ == '__main__':
    main()
