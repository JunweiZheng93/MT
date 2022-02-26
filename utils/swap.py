import tensorflow as tf
import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-14]
sys.path.append(PROJ_ROOT)
from utils.cherry_pick import configure_gpu, CHERRY_CHAIRS
from utils.dataloader import CATEGORY_MAP
import importlib
import scipy.io
import numpy as np
from copy import deepcopy
from utils import visualization
import argparse


def swap(model_path,
         category='chair',
         which_part=1,
         mode='batch',
         batch_size=4,
         shape1=None,
         shape2=None,
         H=32,
         W=32,
         D=32,
         C=1,
         which_gpu=0,
         seed=0):

    def _get_pred_label(pred):
        code = 0
        for idx, each_part in enumerate(pred):
            code += each_part * 2 ** (idx + 1)
        pred_label = tf.math.floor(tf.experimental.numpy.log2(code + 1))
        pred_label = pred_label.numpy().astype('uint8')
        return pred_label

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    configure_gpu(which_gpu)

    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], 3,
                           hparam.hparam['use_attention'],
                           hparam.hparam['keep_channel'], hparam.hparam['use_extra_loss'], hparam.hparam['which_layer'],
                           hparam.hparam['num_blocks'], hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # get hash code of the shapes
    if mode == 'single':
        if shape1 is None or shape2 is None:
            raise ValueError('please enter 2 different shape hash code!')
        else:
            hash_codes = [[shape1, shape2]]
    elif mode == 'batch':
        np.random.seed(seed)
        if category == 'chair':
            hash_codes = list(np.random.choice(CHERRY_CHAIRS, size=2*batch_size, replace=False).reshape((batch_size, 2)))
    else:
        raise ValueError('mode should be one of single and batch!')

    # get unlabeled shapes and labeled shapes
    unlabeled_shape_list = list()
    labeled_shape_list = list()
    for code1, code2 in hash_codes:
        unlabeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_unlabeled.mat')
        unlabeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_unlabeled.mat')
        labeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_labeled.mat')
        labeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_labeled.mat')
        unlabeled_shape1 = scipy.io.loadmat(unlabeled_path1)['data'][..., np.newaxis]
        unlabeled_shape2 = scipy.io.loadmat(unlabeled_path2)['data'][..., np.newaxis]
        labeled_shape1 = scipy.io.loadmat(labeled_path1)['data']
        labeled_shape2 = scipy.io.loadmat(labeled_path2)['data']
        unlabeled_shape_list.append(tf.cast(tf.stack((unlabeled_shape1, unlabeled_shape2), axis=0), dtype=tf.float32))
        labeled_shape_list.append([labeled_shape1, labeled_shape2])

    for unlabeled_shape, labeled_shape, hash_code in zip(unlabeled_shape_list, labeled_shape_list, hash_codes):
        # get latent representation
        latent = my_model.decomposer(unlabeled_shape, training=False)

        # swap latent representation
        latent1, latent2 = tf.unstack(latent, axis=0)
        latent_array1 = latent1.numpy()
        latent_array2 = latent2.numpy()
        part_latent = deepcopy(latent_array1[which_part-1])
        latent_array1[which_part-1] = latent_array2[which_part-1]
        latent_array2[which_part-1] = part_latent
        swapped_latent = tf.cast(tf.stack([latent_array1, latent_array2], axis=0), dtype=tf.float32)

        # visualize gt
        for gt, code in zip(labeled_shape, hash_code):
            visualization.visualize(gt, title=code)

        # get reconstruction and swapped reconstruction
        fake_input = 0
        outputs = my_model(fake_input, training=False, decomposer_output=latent)
        swapped_outputs = my_model(fake_input, training=False, decomposer_output=swapped_latent)

        # visualize reconstruction
        for output, code in zip(outputs, hash_code):
            output = tf.squeeze(tf.where(output > 0.5, 1., 0.))
            output = _get_pred_label(output)
            visualization.visualize(output, title=code)

        # visualize swapped reconstruction
        for swapped_output, code in zip(swapped_outputs, hash_code):
            swapped_output = tf.squeeze(tf.where(swapped_output > 0.5, 1., 0.))
            swapped_output = _get_pred_label(swapped_output)
            visualization.visualize(swapped_output, title=code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-w', '--which_part', default=1, help='which part to be swapped. Default is 1')
    parser.add_argument('-m', '--mode', default='batch', help='swap a batch of full shape pair of just a single pair. Default is single')
    parser.add_argument('-b', '--batch_size', default='4', help='how many full shape pairs to be swapped. Only valid in batch mode. Default is 4')
    parser.add_argument('-s1', '--shape1', default=None, help='hash code of the first full shape. Only valid in single mode.')
    parser.add_argument('-s2', '--shape2', default=None, help='hash code of the second full shape. Only valid in single mode.')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    parser.add_argument('-s', '--seed', default=0, help='seed for choosing hash code randomly. Only valid in batch mode. Default is 0')
    args = parser.parse_args()

    swap(model_path=args.model_path,
         category=args.category,
         which_part=int(args.which_part),
         mode=args.mode,
         batch_size=int(args.batch_size),
         shape1=args.shape1,
         shape2=args.shape2,
         H=int(args.H),
         W=int(args.W),
         D=int(args.D),
         C=int(args.C),
         which_gpu=int(args.gpu),
         seed=int(args.seed))
