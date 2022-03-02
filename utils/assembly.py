import tensorflow as tf
import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-18]
sys.path.append(PROJ_ROOT)
from utils.cherry_pick import configure_gpu
from utils.dataloader import CATEGORY_MAP
import importlib
import scipy.io
import numpy as np
from utils import visualization
import argparse
from utils import stack_plot
from tensorflow.keras.utils import Progbar


def assembly(model_path,
             shape,
             seed=0,
             category='chair',
             visualize=False,
             H_crop_factor=0.2,
             W_crop_factor=0.55,
             H_shift=15,
             W_shift=40,
             H=32,
             W=32,
             D=32,
             C=1,
             which_gpu=0):

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
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'],
                           hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'], hparam.hparam['num_blocks'],
                           hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # get unlabeled shapes and labeled shapes
    unlabeled_shape_list = list()
    labeled_shape_list = list()
    for hash_code in shape:
        unlabeled_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], hash_code, 'object_unlabeled.mat')
        labeled_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], hash_code, 'object_labeled.mat')
        unlabeled_shape = scipy.io.loadmat(unlabeled_path)['data'][..., np.newaxis]
        labeled_shape = scipy.io.loadmat(labeled_path)['data']
        unlabeled_shape_list.append(unlabeled_shape)
        labeled_shape_list.append(labeled_shape)
    unlabeled_shape = tf.cast(tf.stack(unlabeled_shape_list, axis=0), dtype=tf.float32)

    base_dir = os.path.join(PROJ_ROOT, 'results', model_path.split('/')[-3], 'assembly')
    saved_dir = os.path.join(PROJ_ROOT, 'results', model_path.split('/')[-3], 'assembly')
    count = 0
    while True:
        if os.path.exists(saved_dir):
            saved_dir = f'{base_dir}_{count}'
            count += 1
            continue
        else:
            os.makedirs(saved_dir)
            break

    mixed_latent = assemble_latent(my_model, unlabeled_shape, seed)

    # mixed reconstruction
    fake_input = 0
    mixed_outputs = my_model(fake_input, training=False, decomposer_output=mixed_latent)

    if visualize:
        # visualize gt
        for gt, hash_code in zip(labeled_shape_list, shape):
            visualization.visualize(gt, title=hash_code)

        # visualize mixed reconstruction
        for mixed_output in mixed_outputs:
            mixed_output = tf.squeeze(tf.where(mixed_output > 0.5, 1., 0.))
            mixed_output = _get_pred_label(mixed_output)
            visualization.visualize(mixed_output)

    pb = Progbar(len(labeled_shape_list))
    print('Saving images, please wait...')
    for count, (gt, mixed_output, code) in enumerate(zip(labeled_shape_list, mixed_outputs, shape)):
        mixed_output = tf.squeeze(tf.where(mixed_output > 0.5, 1., 0.))
        mixed_output = _get_pred_label(mixed_output)
        visualization.save_visualized_img(gt, os.path.join(saved_dir, f'{code}_gt.png'))
        visualization.save_visualized_img(mixed_output, os.path.join(saved_dir, f'{code}_assembled.png'))
        pb.update(count+1)
    print('Stacking all images together, please wait...')
    stack_plot.stack_assembly_plot(saved_dir, H_crop_factor=H_crop_factor, W_crop_factor=W_crop_factor, H_shift=H_shift, W_shift=W_shift)
    print(f'Done! All images are saved in {saved_dir}')


def assemble_latent(my_model, unlabeled_shape, seed):

    # get latent representation
    latent = my_model.decomposer(unlabeled_shape, training=False)

    # mix latent representation
    mixed_order_list = list()
    np.random.seed(seed)
    for i in range(latent.shape[1]):
        mixed_order_list.append(np.random.choice(latent.shape[0], latent.shape[0], False))
    latent = tf.transpose(latent, (1, 0, 2))
    mixed_list = list()
    for each_latent, each_order in zip(latent, mixed_order_list):
        mixed_list.append(tf.gather(each_latent, each_order, axis=0))
    mixed_latent = tf.stack(mixed_list, axis=1)

    return mixed_latent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-s', '--shape', nargs='+', help='hash code of the shapes to be assembled.', default=['1bbe463ba96415aff1783a44a88d6274', '5893038d979ce1bb725c7e2164996f48', 'cd9702520ad57689bbc7a6acbd8f058b', '5042005e178d164481d0f12b8bf5c990'])
    parser.add_argument('--seed', default=6, help='seed for the random mixed order. Default is 6')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-v', '--visualize', action='store_true', help='whether visualize the result or not')
    parser.add_argument('--H_crop_factor', default=0.2, help='Percentage to crop empty spcae of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_crop_factor', default=0.55, help='Percentage to crop empty spcae of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('--H_shift', default=15, help='How many pixels to be shifted for the cropping of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_shift', default=40, help='How many pixels to be shifted for the cropping of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    assembly(model_path=args.model_path,
             shape=args.shape,
             seed=int(args.seed),
             category=args.category,
             visualize=args.visualize,
             H_crop_factor=float(args.H_crop_factor),
             W_crop_factor=float(args.W_crop_factor),
             H_shift=int(args.H_shift),
             W_shift=int(args.W_shift),
             H=int(args.H),
             W=int(args.W),
             D=int(args.D),
             C=int(args.C),
             which_gpu=int(args.gpu))
