import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAvgPool2D


class LDRNet(Model):
    def __init__(self, input_shapes=[224, 224, 3], points_size=100, classification_list=[1], alpha=0.35):
        super(LDRNet, self).__init__()
        self.classification_list = classification_list
        self.points_size = points_size
        self.input_shapes = input_shapes
        self.base_model = \
            tf.keras.applications.MobileNetV2(input_shape=self.input_shapes,
                                              alpha=alpha, include_top=False)
        if len(classification_list) > 0:
            class_size = sum(self.classification_list)
        else:
            class_size = 0
        self.global_pool = GlobalAvgPool2D()
        self.corner = OutputBranch(8, "output_corner")
        self.border = OutputBranch((points_size - 4) * 2, "output_border")
        self.cls = OutputBranch(class_size + len(self.classification_list), "output_class")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_pool(x)
        corner_output = self.corner(x)
        border_output = self.border(x)
        cls_output = self.cls(x)
        return corner_output, border_output, cls_output


class OutputBranch(Model):
    def __init__(self, size, name=None):
        super(OutputBranch, self).__init__()
        self.dense = Dense(size, name=name)

    def call(self, inputs):
        output = self.dense(inputs)
        return output


if __name__ == "__main__":
    import numpy as np

    xx = np.zeros((1, 224, 224, 3))
    model = LDRNet()
    y = model(xx)
    print(y)
