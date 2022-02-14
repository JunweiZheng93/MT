import tensorflow as tf
import importlib
import os
import argparse
from utils import dataloader


PROJ_ROOT = os.path.abspath(__file__)[:-16]
CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def evaluate_model(model_path,
                   which_gpu=0,
                   H=32,
                   W=32,
                   D=32,
                   C=1):

    configure_gpu(which_gpu)

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['training_process'], hparam.hparam['use_attention'],
                           hparam.hparam['keep_channel'], hparam.hparam['use_extra_loss'], hparam.hparam['which_layer'],
                           hparam.hparam['num_blocks'], hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # get dataset
    training_set, test_set = dataloader.get_dataset(category=hparam.hparam['category'], batch_size=hparam.hparam['batch_size'],
                                                    split_ratio=hparam.hparam['split_ratio'], max_num_parts=hparam.hparam['max_num_parts'])

    part_mIoU_tracker = tf.keras.metrics.MeanIoU(2)
    shape_mIoU_tracker = tf.keras.metrics.MeanIoU(2)

    print('Your model is being evaluated, please wait...')

    if hparam.hparam['training_process'] == 1 or hparam.hparam['training_process'] == '1':
        for x, labels, trans in test_set:
            parts = my_model(x, training=False)
            parts = tf.transpose(tf.where(parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part in zip(labels, parts):
                part_mIoU_tracker.update_state(gt, part)
        print(f'part_mIoU: {part_mIoU_tracker.result()}')

    elif hparam.hparam['training_process'] == 2 or hparam.hparam['training_process'] == '2':
        for x, labels, trans in test_set:
            theta = my_model(x, training=False)
            shapes = model.Resampling()((my_model.stacked_decoded_parts, theta))
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            shape_mIoU_tracker.update_state(x, shapes)
        print(f'shape_mIoU: {shape_mIoU_tracker.result()}')

    else:
        for x, labels, trans in test_set:
            shapes = my_model(x, training=False)
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            parts = tf.transpose(tf.where(my_model.stacked_decoded_parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part in zip(labels, parts):
                part_mIoU_tracker.update_state(gt, part)
            shape_mIoU_tracker.update_state(x, shapes)
        print(f'part_mIoU: {part_mIoU_tracker.result()}')
        print(f'shape_mIoU: {shape_mIoU_tracker.result()}')


def configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   which_gpu=int(args.gpu),
                   H=int(args.H),
                   W=int(args.W),
                   D=int(args.D),
                   C=int(args.C))
