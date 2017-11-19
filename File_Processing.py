import os


def ensure_directory(path):

    if not os.path.exists(path):
        os.mkdir(path)


def get_image_paths(image_dir, predictions):
    """
    A simple programm that will find the paths of the detected images.
    :param image_dir: The location of the image directory.
    :param predictions: The vector of predictions
    :return: A list containing the paths to every images detected
    """

    images_names = os.listdir(image_dir)
    image_paths = []

    if len(images_names) > len(predictions):
        raise AssertionError('More prediction than files found. Probably your directory has subdirectories.')
    for idx, prediction in enumerate(predictions):
        if prediction:
            image_paths.append(os.path.join(image_dir, images_names[idx]))

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
