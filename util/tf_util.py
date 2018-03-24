import tensorflow as tf


def create_tf_scalar_summaries(var, scope_name='summaries'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Source: https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
    """

    with tf.name_scope(scope_name):

        with tf.name_scope('mean'):

            mean = tf.reduce_mean(var)

            tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):

            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)

        tf.summary.scalar('max', tf.reduce_max(var))

        tf.summary.scalar('min', tf.reduce_min(var))

        tf.summary.histogram('histogram', var)
