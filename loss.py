from tensorflow import keras
import tensorflow as tf


class WeightedLocLoss(keras.losses.Loss):
    def __init__(self, name="weighted_loc_loss", point_size=4):
        self.point_size = point_size
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, weights, loss_type="mse"):
        if self.point_size == 4:
            if loss_type == "mse":
                mse = tf.math.reduce_mean(tf.square(y_pred - y_true) * weights, 1)
                return mse
            if loss_type == "l1":
                l1 = tf.math.reduce_mean(tf.abs(y_pred - y_true) * weights, 1)
                return l1
            if loss_type == "log":
                log_loss = tf.math.reduce_mean(tf.math.log(1 + tf.abs(y_pred - y_true)) * weights, 1)
                return log_loss


class LineLoss(keras.losses.Loss):
    def __init__(self, name="line_loss"):
        super().__init__(name=name)

    def __call__(self, line):
        line_x = line[:, 0::2]
        line_y = line[:, 1::2]
        x_diff = line_x[:, 1:] - line_x[:, 0:-1]
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]
        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (
                    tf.sqrt(tf.square(x_diff_start) + tf.square(y_diff_start)) * tf.sqrt(
                tf.square(x_diff_end) + tf.square(y_diff_end)) + 0.0000000000001)
        slop_loss = tf.math.reduce_mean(1 - similarity, axis=1)
        x_diff_loss = tf.math.reduce_mean(tf.square(x_diff[:, 1:] - x_diff[:, 0:-1]), 1)
        y_diff_loss = tf.math.reduce_mean(tf.square(y_diff[:, 1:] - y_diff[:, 0:-1]), 1)
        return slop_loss, x_diff_loss + y_diff_loss


if __name__ == "__main__":
    import numpy as np
    loss = WeightedLocLoss()
    res = loss.__call__(y_true=np.ones((1, 8)), y_pred=np.zeros((1, 8)), weights=np.ones((1, 8)), loss_type="mse")
    res = loss.__call__(y_true=np.ones((1, 8)), y_pred=np.zeros((1, 8)), weights=np.ones((1, 8)), loss_type="l1")
    res = loss.__call__(y_true=np.ones((1, 8)), y_pred=np.zeros((1, 8)), weights=np.ones((1, 8)), loss_type="log")
    print("done")
