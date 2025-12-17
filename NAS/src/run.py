import argparse
import tensorflow as tf
import numpy as np
from src.search_pipeline import architecture_search


def parse_args():
    parser = argparse.ArgumentParser(description='Search architectures for CIFAR-10')
    parser.add_argument(
        '--generations-number',
        type=int,
        default=40,
        help='Number of evolutionary search generations'
    )
    parser.add_argument(
        '--population-number',
        type=int,
        default=60,
        help='Number of architectures in population'
    )
    parser.add_argument(
        '--mutations-per-generation',
        type=int,
        default=15,
        help='Number of parent mutations per generation'
    )
    parser.add_argument(
        '--tournament-size',
        type=int,
        default=3,
        help='Evolutionary tournament size'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) \
        = tf.keras.datasets.cifar10.load_data()

    cifar_train_images = np.transpose(cifar_train_images, (0, 3, 1, 2))\
        .reshape((cifar_train_images.shape[0], 32 * 32 * 3)).astype('float32') / 255
    cifar_test_images = np.transpose(cifar_test_images, (0, 3, 1, 2))\
        .reshape((cifar_test_images.shape[0], 32 * 32 * 3)).astype('float32') / 255

    val_size = int(0.2 * cifar_train_images.shape[0])
    train_images = cifar_train_images[:-val_size]
    train_labels = cifar_train_labels[:-val_size]
    val_images = cifar_train_images[-val_size:]
    val_labels = cifar_train_labels[-val_size:]

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000)\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((cifar_test_images, cifar_test_labels))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    architecture_search(
        train_dataset,
        val_dataset,
        test_dataset,
        input_shape=(32, 32, 3),
        output_nodes=10,
        iterations_number=args.generations_number,
        mutations_per_iteration=args.mutations_per_generation,
        initial_population_number=args.population_number,
        tournament_size=args.tournament_size
    )
