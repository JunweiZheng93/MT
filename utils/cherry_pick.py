import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-21]
sys.path.append(PROJ_ROOT)
import tensorflow as tf
import importlib
import scipy.io
import numpy as np
import argparse
from tqdm import tqdm
from utils.dataloader import Dataset
from utils import visualization
import shutil


CHERRY_CHAIRS = ['54e2aa868107826f3dbc2ce6b9d89f11', '5042005e178d164481d0f12b8bf5c990', 'b8e4d2f12e740739b6c7647742d948e',
                 '9e145541bf7e278d19fb4103277a6b93', '2207db2fa2e4cc4579b3e1be3524f72f', '2a87cf850dca72a4a886e56ff3d54c4',
                 '9ab18a33335373b2659dda512294c744', '5b9ebc70e9a79b69c77d45d65dc3714', '1bbe463ba96415aff1783a44a88d6274',
                 'b2ded1854643f1451c1b3b2ed8d13bf8', 'dfc9e6a84553253ef91663a74ccd2338', '5893038d979ce1bb725c7e2164996f48',
                 '88aec853dcb10d526efa145e9f4a2693', '611f235819b7c26267d783b4714d4324', 'cd9702520ad57689bbc7a6acbd8f058b',
                 '2a56e3e2a6505ec492d9da2668ec34c', '5a643c0c638fc2c3ff3a3ae710d23d1e', '96929c12a4a6b15a492d9da2668ec34c',
                 '1b7ba5484399d36bc5e50b867ca2d0b9', '2fed64c67552aa689c1db271ad9472a7', '9d7d7607e1ba099bd98e59dfd5823115',
                 '2031dca3aaeff940f7628281ecb18112', '875925d42780159ffebad4f49b26ec52', '2025aa3a71f3c468d16ba2cb1292d98a',
                 '3c408a4ad6d57c3651bc6269fcd1b4c0']


def evaluate_model(model_path,
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
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], 3, hparam.hparam['use_attention'],
                           hparam.hparam['keep_channel'], hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'],
                           hparam.hparam['num_blocks'], hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    test_set, shape_paths = get_dataset(hparam.hparam['category'], 1, hparam.hparam['max_num_parts'])

    saved_dir = os.path.join(PROJ_ROOT, 'results', model_path.split('/')[-3], 'cherry_picks')
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)
    print('Generating images for cherry picks, please wait...')
    for (voxel_grid, label, trans), shape_path in tqdm(zip(test_set, shape_paths), total=len(shape_paths)):
        stacked_transformed_parts = my_model(voxel_grid, training=False)
        gt_label_path = os.path.join(shape_path, 'object_labeled.mat')
        gt_label = scipy.io.loadmat(gt_label_path)['data']
        shape_name = gt_label_path.split('/')[-2]
        visualization.save_visualized_img(gt_label, os.path.join(saved_dir, f'{shape_name}_gt.png'))
        if hparam.hparam['training_process'] == 1 or hparam.hparam['training_process'] == '1':
            pred = tf.squeeze(tf.where(my_model.stacked_decoded_parts > 0.5, 1., 0.))
            pred = pred.numpy().astype('uint8')
            for i, part in enumerate(pred):
                visualization.save_visualized_img(part, os.path.join(saved_dir, f'{shape_name}_part{i+1}_recon.png'))
        else:
            pred = tf.squeeze(tf.where(stacked_transformed_parts > 0.5, 1., 0.))
            pred_label = _get_pred_label(pred)
            visualization.save_visualized_img(pred_label, os.path.join(saved_dir, f'{shape_name}_shape_recon.png'))


def get_dataset(category, batch_size, max_num_parts):
    voxel_grid_fp, part_fp, trans_fp, shape_paths = get_fp(category)
    all_voxel_grid = list()
    all_part = list()
    all_trans = list()
    for v_fp, p_fp, t_fp in zip(voxel_grid_fp, part_fp, trans_fp):
        v = scipy.io.loadmat(v_fp)['data'][:, :, :, np.newaxis]
        all_voxel_grid.append(v)
        parts = list()
        transformations = list()
        member_list = [int(each[-5]) for each in p_fp]
        dir_name = os.path.dirname(v_fp)
        for i in range(1, max_num_parts + 1):
            if i not in member_list:
                part = np.zeros_like(v, dtype='uint8')
                parts.append(part)
                transformation = np.zeros((3, 4), dtype='float32')
                transformations.append(transformation)
            else:
                part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
                parts.append(part)
                transformations.append(
                    scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
        all_part.append(parts)
        all_trans.append(transformations)
    test_set = Dataset(all_voxel_grid, all_part, all_trans, batch_size)
    return test_set, shape_paths


def get_fp(category):
    if category == 'chair':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '03001627')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in CHERRY_CHAIRS]
    else:
        pass
    voxel_grid_fp = list()
    part_fp = list()
    trans_fp = list()
    for shape_path in shape_paths:
        voxel_grid = os.path.join(shape_path, 'object_unlabeled.mat')
        part_list = list()
        trans_list = list()
        all_files = sorted(os.listdir(shape_path))
        for file in all_files:
            if file.startswith('part') and file.endswith('.mat'):
                if file.startswith('part') and file.endswith('trans_matrix.mat'):
                    trans_list.append(os.path.join(shape_path, file))
                    continue
                part_list.append(os.path.join(shape_path, file))
        voxel_grid_fp.append(voxel_grid)
        part_fp.append(part_list)
        trans_fp.append(trans_list)
    return voxel_grid_fp, part_fp, trans_fp, shape_paths


def configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   H=int(args.H),
                   W=int(args.W),
                   D=int(args.D),
                   C=int(args.C),
                   which_gpu=int(args.gpu))
