import os


def ensure_directory(path):

    if not os.path.exists(path):
        os.mkdir(path)


def get_all_images_path(image_dir):

    return [os.path.join(image_dir, path) for path in os.listdir(image_dir)]


def get_relevant_image_paths(all_paths, already_used, predictions):
    """
    A function that returns the paths of images detected, and not yet processed.
    :param all_paths: Paths of all original, images
    :param already_used: Paths of already processed images
    :param predictions: The vector of predictions
    :return: A list containing the paths of remaining detection
    """

    remaining_paths = []
    for idx, prediction in enumerate(predictions):
        if prediction and not all_paths[idx] in already_used:
            remaining_paths.append(all_paths[idx])

    return remaining_paths


def get_image_paths(image_dir, predictions):
    """
    A simple function that will find the paths of the detected images.
    :param image_dir: The location of the image directory.
    :param predictions: The vector of predictions
    :return: A list containing the paths to every images detected
    """

    images_names = get_all_images_path(image_dir)
    image_paths = []

    if len(images_names) > len(predictions):
        raise AssertionError('More prediction than files found. Probably your directory has subdirectories.')
    for idx, prediction in enumerate(predictions):
        if prediction:
            image_paths.append(images_names[idx])

    return image_paths


def move_images(relocation_dir, image_paths):
    """
    This function will move our detected images to the desired location.
    :param relocation_dir: The new location for our detected images.
    :param image_paths: A list containing the paths to every images detected
    :return: Nothing
    """

    ensure_directory(relocation_dir)
    for path in image_paths:
        os.rename(path, os.path.join(relocation_dir, os.path.basename(path)))


def delete_images(image_paths):
    """
    This function will delete our detected images to the desired location.
    :param image_paths: A list containing the paths to every images detected
    :return: Nothing
    """

    for path in image_paths:
        os.remove(path)
