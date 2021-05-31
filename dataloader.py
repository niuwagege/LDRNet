import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import tensorflow_addons as tfa
import numpy as np

def rotate_point(point, radians, image_height, image_width):
    image_height, image_width = (
        tf.cast(image_height, dtype=dtypes.float32), tf.cast(image_width, dtype=dtypes.float32))
    # y = -tf.cast(image_height * (point[1] - 0.5),dtype=dtypes.int32)
    # x = tf.cast(image_width * (point[0] - 0.5),dtype=dtypes.int32)
    y = -image_height * (point[:, 1] - 0.5)
    x = image_width * (point[:, 0] - 0.5)
    # coordinates = tf.stack()
    coordinates = tf.stack([y, x], axis=1)
    rotation_matrix = tf.stack(
        [[tf.cos(radians), tf.sin(radians)],
         [-tf.sin(radians), tf.cos(radians)]])
    new_coords = tf.cast(
        tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
    x = -(tf.cast(new_coords[0, :], dtype=dtypes.float32) / image_height - 0.5)
    y = tf.cast(new_coords[1, :], dtype=dtypes.float32) / image_width + 0.5
    return tf.stack([y, x], axis=1)


def random_padding(img, label, max_ratio=0.5):
    ratio = tf.random.uniform(shape=[4], minval=0., maxval=max_ratio)
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    size_change = tf.round(ratio * tf.cast(tf.concat([img_shape[0:2], img_shape[0:2]], axis=0), dtype=dtypes.float32))
    new_height = img_shape[0] + size_change[0] + size_change[2]
    new_width = img_shape[1] + size_change[1] + size_change[3]
    img = tf.image.pad_to_bounding_box(img, tf.cast(size_change[0], dtype=tf.int32),
                                       tf.cast(size_change[1], dtype=tf.int32), tf.cast(new_height, dtype=tf.int32),
                                       tf.cast(new_width, dtype=tf.int32))
    coords = label[0:8] * [img_shape[1], img_shape[0], img_shape[1], img_shape[0], img_shape[1], img_shape[0],
                           img_shape[1], img_shape[0]]
    coords = coords + [size_change[1], size_change[0], size_change[1], size_change[0], size_change[1], size_change[0],
                       size_change[1], size_change[0]]
    coords = coords / [new_width, new_height, new_width, new_height, new_width, new_height, new_width, new_height]
    return img, tf.concat([tf.reshape(coords, [-1]), label[8:]], axis=0)


def random_crop(img, label, max_ratio=0.1):
    # label：normalized
    ratio = tf.random.uniform(shape=[4], minval=0., maxval=max_ratio)
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    size_change = tf.round(ratio * tf.cast(tf.concat([img_shape[0:2], img_shape[0:2]], axis=0), dtype=dtypes.float32))
    new_height = img_shape[0] - size_change[0] - size_change[2]
    new_width = img_shape[1] - size_change[1] - size_change[3]
    # tfa.image.
    img = tf.image.crop_to_bounding_box(img, tf.cast(size_change[0], dtype=tf.int32),
                                        tf.cast(size_change[1], dtype=tf.int32), tf.cast(new_height, dtype=tf.int32),
                                        tf.cast(new_width, dtype=tf.int32))
    coords = label[0:8] * [img_shape[1], img_shape[0], img_shape[1], img_shape[0], img_shape[1], img_shape[0],
                           img_shape[1], img_shape[0]]
    coords = coords - [size_change[1], size_change[0], size_change[1], size_change[0], size_change[1], size_change[0],
                       size_change[1], size_change[0]]
    coords = coords / [new_width, new_height, new_width, new_height, new_width, new_height, new_width, new_height]
    return img, tf.concat([tf.reshape(coords, [-1]), label[8:]], axis=0)


class CardDataset:
    def __init__(self, txt_path, coord_size=8, img_folder="", val_ratio=0.2, batch_size=1, class_sizes=[],
                 weighted=False):
        self.txt_path = txt_path
        self.data_size = 0
        self.val_size = 0
        self.val_ratio = val_ratio
        self.weighted = weighted
        self.batch_size = batch_size
        self.class_size = len(class_sizes)
        self.img_folder = img_folder
        self.coord_size = coord_size
        self.img_size = 224

    @staticmethod
    def decode_img(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img_shape = tf.shape(img)
        # resize the image to the desired size
        img = tf.image.resize(img, [224, 224])
        print(tf.shape(img))
        return img / 255.0, img_shape

    @staticmethod
    def augment(img, label):
        img = img * 255.0

        img_height = tf.shape(img)[0]
        img_width = tf.shape(img)[1]

        # to gray
        if tf.random.uniform([]) < 0.2:
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.grayscale_to_rgb(img)
            # img = tf.stack([img, img, img], axis=2)

        # brightness
        if tf.random.uniform([]) < 0.5:
            img = tf.image.random_brightness(img, 0.2)

        # hue
        if tf.random.uniform([]) < 0.1:
            img = tf.image.random_hue(img, 0.1)
        # constrast
        if tf.random.uniform([]) < 0.1:
            img = tf.image.random_contrast(img, 0.2, 0.5)
        # if tf.random.uniform([]) < 0.2:
        #     img = tf.image.random_jpeg_quality(img, 70, 95)

        # saturation
        if tf.random.uniform([]) < 0.01:
            img = tf.image.random_saturation(img, 5, 10)

        # gaussian
        if tf.random.uniform([]) < 0.5:
            img = tfa.image.gaussian_filter2d(img)

        # random location cutout
        if tf.random.uniform([]) < 0.9:
            scale = np.random.uniform(0.05, 0.2)
            size = tf.cast(tf.cast(img_height, dtype=dtypes.float32) * scale, dtype=dtypes.int32)
            size = size // 2 * 2
            img = tf.expand_dims(img, axis=0)
            img = tfa.image.random_cutout(img, (size, size))
            img = img[0]

        # rotate 90 180 270
        rotate_p = tf.random.uniform([])
        rad = 0.0
        if rotate_p < 0.25:
            rad = 3.141592653 / 2
        elif rotate_p < 0.5:
            rad = 3.141592653
        elif rotate_p < 0.75:
            rad = 3.141592653 / 2 * 3
        else:
            rad = 0.0

        # rotate random degress
        if tf.random.uniform([]) < 0.3:
            rad += np.random.uniform(0, 3.141592653 * 2)
        img = tfa.image.rotate(img, rad)
        coords = tf.reshape(label[0:8], [-1, 2])
        new_coord = rotate_point(coords, rad, img_height, img_width)
        new_coord = tf.reshape(new_coord, [1, -1])
        label = tf.concat([tf.reshape(new_coord, [-1]), label[8:]], axis=0)

        # random crop
        if tf.random.uniform([]) < 0.5:
            img, label = random_crop(img, label)

        # random padding
        if tf.random.uniform([]) < 0.5:
            img, label = random_padding(img, label)

        # must_resize after padding or crop
        img = tf.image.resize(img, [224, 224])

        return img / 255, label

    @staticmethod
    def augment_tfa(img, label):
        img = img * 255.0
        img_height = tf.shape(img)[0]
        img_width = tf.shape(img)[1]
        img = tfa.image.gaussian_filter2d(img)

        scale = np.random.uniform(0.05, 0.2)
        size = int(tf.shape(img)[0].numpy() * scale)
        size = size // 2 * 2
        img = tf.expand_dims(img, axis=0)
        img = tfa.image.random_cutout(img, (size, size))
        img = img[0]

        rad = np.random.uniform(0, 3.14)
        img = tfa.image.rotate(img, rad)
        coords = tf.reshape(label[0:8], [-1, 2])
        new_coord = rotate_point(coords, rad, img_height, img_width)
        new_coord = tf.reshape(new_coord, [1, -1])
        label = tf.concat([tf.reshape(new_coord, [-1]), label[8:]], axis=0)
        return img / 255, label

    def decode_label(self, label, height=1, width=1):
        labels = label  # tf.strings.split(label, "-")
        number_labels = tf.strings.to_number(labels)
        normalizer = [width, height, width, height, width, height, width, height] + [1] * (
                self.class_size + self.coord_size)
        # print(normalizer)
        return number_labels / normalizer

    def analysis_line(self, line):
        # print("1")
        parts = tf.strings.split(line, ",")
        # if self.weighted:
        img, shape = CardDataset.decode_img(self.img_folder + parts[0])
        return img, self.decode_label(parts[1:], shape[0], shape[1])
        # else:
        #     # not completed
        #     return

    def analysis_line_with_weighted(self, line):
        parts = tf.strings.split(line, ",")
        # if self.weighted:
        return CardDataset.decode_img(parts[0]), CardDataset.decode_label(parts[1])
        # else:
        #     # not completed
        #     return

    def configure_for_performance(self, ds, aug=False):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        if aug:
            ds = ds.map(CardDataset.augment)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def get_data(self):
        with open(self.txt_path, "r") as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
            self.data_size = len(lines)
        np.random.shuffle(lines)
        file_lines = ops.convert_to_tensor(
            lines, dtype=dtypes.string, name="file_lines")
        dataset = tf.data.Dataset.from_tensor_slices(file_lines)
        self.val_size = int(self.data_size * self.val_ratio)
        dataset.shuffle(self.data_size, reshuffle_each_iteration=False)
        dataset_train = dataset.skip(self.val_size)
        dataset_val = dataset.take(self.val_size)

        if not self.weighted:
            train_ds = dataset_train.map(self.analysis_line)
            val_ds = dataset_val.map(self.analysis_line)
        else:
            train_ds = dataset_train.map(self.analysis_line_with_weighted)
            val_ds = dataset_val.map(self.analysis_line_with_weighted)
        return self.configure_for_performance(train_ds, aug=True), self.configure_for_performance(val_ds, aug=True)


if __name__ == "__main__":
    import numpy as np
    import cv2
    import PIL

    fr = open("/Users/horanqian/workspace/data/video_new_imgs2/label_new2.txt", "r")
    lines = fr.readlines()
    count = 0
    while 1:
        dataset = CardDataset(img_folder="/Users/horanqian/workspace/data/video_new_imgs2/card_detector_images2/",
                              txt_path="/Users/horanqian/workspace/data/video_new_imgs2/label_new2.txt",
                              class_sizes=[2])
        img, label = dataset.analysis_line(lines[count])
        # img = tfa.image.rotate(img,3.14)
        # img, _ = CardDataset.decode_img(
        #     "/Users/horanqian/workspace/data/video_new_imgs2/card_detector_images2/03_app_正常_1600394645061_152.png")
        # img = tf.convert_to_tensor(img_in.astype(np.float32))
        # img, label = random_padding(img, label)
        img_show1 = img.numpy() * 255
        img_show1 = img_show1.astype(np.uint8)
        img_show1 = cv2.cvtColor(img_show1, cv2.COLOR_RGB2BGR)
        img, label = CardDataset.augment(img, label)
        print(label)
        img_show = img.numpy() * 255
        img_show = img_show.astype(np.uint8)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        coord = label[0:8].numpy()
        # coord = [int(x * 224) for x in coord]
        h, w = img_show.shape[0:2]
        coord[0::2] = coord[0::2] * w
        coord[1::2] = coord[1::2] * h
        coord = [int(x) for x in coord]
        cv2.circle(img_show, (coord[0], coord[1]), 3, (255, 0, 0))
        cv2.circle(img_show, (coord[2], coord[3]), 3, (0, 255, 0))
        cv2.circle(img_show, (coord[4], coord[5]), 3, (0, 0, 255))
        cv2.circle(img_show, (coord[6], coord[7]), 3, (255, 0, 255))

        cv2.imshow("asd", img_show)

        cv2.imshow("asd1", img_show1)
        cv2.waitKey(0)
        count += 1
    # card_dataset = CardDataset(txt_path="./test_data.txt", img_folder="./", class_sizes=[2])
    # train_ds, val_ds = card_dataset.get_data()
    # image_batch, label_batch = next(iter(train_ds))
    # print("asd")
