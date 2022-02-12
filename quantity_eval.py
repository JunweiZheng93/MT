import tensorflow as tf


def cal_precision(y_true, y_pred):

    y_pred = tf.where(y_pred > 0.5, 1., 0.)
    inverse_y_ture = tf.where(y_true > 0.5, 0., 1.)

    y_pred = tf.cast(y_pred, dtype=tf.bool)
    y_true = tf.cast(y_true, dtype=tf.bool)
    inverse_y_ture = tf.cast(inverse_y_ture, dtype=tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, y_true), dtype=tf.float32))
    num_fp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, inverse_y_ture), dtype=tf.float32))
    precision = intersection / (intersection + num_fp)

    return precision


def cal_accuracy(y_true, y_pred):

    y_pred = tf.where(y_pred > 0.5, 1., 0.)
    inverse_y_pred = tf.where(y_pred > 0.5, 0., 1.)
    inverse_y_ture = tf.where(y_true > 0.5, 0., 1.)

    y_pred = tf.cast(y_pred, dtype=tf.bool)
    y_true = tf.cast(y_true, dtype=tf.bool)
    inverse_y_ture = tf.cast(inverse_y_ture, dtype=tf.bool)
    inverse_y_pred = tf.cast(inverse_y_pred, dtype=tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, y_true), dtype=tf.float32)) # true positive
    inverse_intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(inverse_y_pred, inverse_y_ture), dtype=tf.float32)) # true negative
    num_fp = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, inverse_y_ture), dtype=tf.float32))
    num_fn = tf.reduce_sum(tf.cast(tf.math.logical_and(inverse_y_pred, y_true), dtype=tf.float32))
    accuracy = (intersection + inverse_intersection) / (intersection + inverse_intersection + num_fp + num_fn)

    return accuracy


def cal_mIoU(y_true, y_pred):

    y_pred = tf.where(y_pred > 0.5, 1., 0.)

    y_pred = tf.cast(y_pred, dtype=tf.bool)
    y_true = tf.cast(y_true, dtype=tf.bool)

    # TODO: NaN
    union = tf.reduce_sum(tf.cast(tf.math.logical_or(y_pred, y_true), dtype=tf.float32), axis=(-4, -3, -2, -1))
    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_pred, y_true), dtype=tf.float32), axis=(-4, -3, -2, -1))
    mIoU = tf.reduce_mean(intersection / union)

    return mIoU


if __name__ == '__main__':
    pass
