from bottleneck import get_bottlenecks_values, ALL_ARCHITECTURES
from file_processing import ensure_directory
import os
import numpy as np
import argparse


def create_noisy_bottlenecks(image_dir, bottleneck_destination, architecture_chosen='MobileNet_1.0_224',
                             model_location='../model'):
    """
    Function that will compute your botlleneck values from your selection of noisy data.

    :param image_dir: List of image dir location you want to test with
    :param bottleneck_destination: Where you want your bottleneck to be saved
    :param architecture_chosen: Which architecture to use to generate your bottlenecks.
            Ranging from the inception to the MobileNet models
            Type 'all' if you want to cycle through all possibilities
    :param model_location: Where the model will be downloaded
    :return: Nothing
    """

    ensure_directory(bottleneck_destination)

    if architecture_chosen == 'all':
        architecture_cycle = ALL_ARCHITECTURES
    else:
        architecture_cycle = [architecture_chosen]

    saved_values = os.listdir(bottleneck_destination)

    for current_architecture in architecture_cycle:

        entry = 'Noise' + '_' + current_architecture + '.npy'

        if entry not in saved_values:
            bottleneck_values = get_bottlenecks_values(image_dir, current_architecture, model_location)
            path = os.path.join(bottleneck_destination, entry)
            np.save(path, bottleneck_values)
            print('Created chached pollution for :', current_architecture)
        else:
            print('Already found computed values for :', current_architecture,
                  ', delete the file, or change location if you want a new one.')


def verify_input(_):
    """
    This method will check the values given by the user.
    :param _: Parser
    :return: Nothing
    """

    if not os.path.exists(FLAGS.image_dir):
        raise AssertionError('Image directory not found.')

    if FLAGS.architecture not in ALL_ARCHITECTURES and not FLAGS.architecture == 'all':
        raise AssertionError('Wrong architecture given.')


def main(_):
    verify_input(_)
    create_noisy_bottlenecks(FLAGS.image_dir, FLAGS.bottleneck_destination, FLAGS.architecture, FLAGS.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help="""\
        Path to folders of random images.\
        """
    )
    parser.add_argument(
        '--bottleneck_destination',
        type=str,
        default='./Cached_pollution/',
        help="""\
            Directory where you want the computed noisy values to be stored.\
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
              for more information on Mobilenet.

              Type all if you want to cycle through all possibilities\
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

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
