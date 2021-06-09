import tensorflow as tf
from LDRNet import LDRNet
from loss import WeightedLocLoss, LineLoss
from dataloader import CardDataset
import os
import numpy as np
import cv2
import sys
import argparse
import yaml


# using weights 可以把那些分类为0的 loc loss 和line loss都设置为0，通过label.txt

def init_args():
    parser = argparse.ArgumentParser(description='LDRNet')
    parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def grad(config, model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, loss_list, y_ = loss(config, model, inputs, targets, training=True,
                                         coord_size=config["points_size"] * 2,
                                         class_list=config["class_list"])
        return loss_value, loss_list, y_, tape.gradient(loss_value, model.trainable_variables)


def loss(config, model, x, y, training, coord_size=8, class_list=[1], use_line_loss=True):
    weighted_loc_loss = WeightedLocLoss()
    line_loss = LineLoss()
    default_coord_size = 8
    if coord_size > 8:
        assert coord_size % 4 == 0, "coord size wrong"
        size_per_line = int((coord_size - 8) / 4 / 2)

        coord_start = y[:, 0:8]
        coord_end = tf.concat([y[:, 2:8], y[:, 0:2]], axis=1)
        coord_increment = (coord_end - coord_start) / (size_per_line + 1)
        new_coord = coord_start + coord_increment
        for index in range(1, size_per_line):
            new_coord = tf.concat([new_coord, coord_start + (index + 1) * coord_increment], axis=1)
        if config["loss"]["using_weights"]:
            weights_start = y[:, 8:16]
            weights_end = tf.concat([y[:, 10:16], y[:, 8:10]], axis=1)
            weights_increment = (weights_end - weights_start) / (size_per_line + 1)
            new_weights = weights_start + weights_increment
            for index in range(1, size_per_line):
                new_weights = tf.concat([new_weights, weights_start + (index + 1) * weights_increment], axis=1)
            y = tf.concat([coord_start, new_coord, weights_start, new_weights, tf.expand_dims(y[:, 8 * 2], axis=1)],
                          axis=1)
        else:
            y = tf.concat([new_coord, y[:, 8]], axis=1)
    corner_y_, border_y_, class_y_ = model(x, training=training)
    coord_y_ = tf.concat([corner_y_, border_y_], axis=1)
    coord_y = y[:, 0:coord_size]
    if config["loss"]["using_weights"]:
        weights = y[:, coord_size:coord_size * 2]
        y_end = coord_size * 2
    else:
        weights = tf.ones(weighted_loc_loss.point_size * 2) / (weighted_loc_loss.point_size * 2)
        y_end = coord_size
    y__end = coord_size
    losses = []
    total_loss = 0
    for class_size in class_list:
        class_y = y[:, y_end]
        y_end += 1
        y__end += class_size + 1
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(class_y, class_y_, from_logits=True)
        losses.append(class_loss)
        total_loss += class_loss
    # tf.keras.losses.sparse_categorical_crossentropy()
    loc_loss = config["loss"]["loss_ratio"] * weighted_loc_loss(coord_y, coord_y_, weights=weights, loss_type="mse")
    total_loss += loc_loss * config["loss"]["class_loss_ratio"]
    losses.append(loc_loss * config["loss"]["class_loss_ratio"])

    # cal slope and distance

    if coord_size > 8:
        total_slop_loss = 0
        total_diff_loss = 0
        for index in range(4):
            line = coord_y_[:, index * 2:index * 2 + 2]
            for coord_index in range(size_per_line):
                line = tf.concat(
                    [line, coord_y_[:, 8 + coord_index * 8 + index * 2:8 + coord_index * 8 + index * 2 + 2]], axis=1)
                # liner = tf.concat([liner,coord_y[:,8+coord_index*8+index*2:8+coord_index*8+index*2+2]],axis=1)
            line = tf.concat([line, coord_y_[:, (index * 2 + 2) % 8:(index * 2 + 2 + 2) % 8]], axis=1)
            cur_slop_loss, cur_diff_loss = line_loss(line)
            if config["loss"]["using_weights"]:
                total_slop_loss += cur_slop_loss * tf.math.reduce_mean(weights, axis=1)
                total_diff_loss += cur_diff_loss * tf.math.reduce_mean(weights, axis=1)
        if use_line_loss:
            losses.append(total_slop_loss * config["loss"]["slop_loss_ratio"])
            losses.append(total_diff_loss * config["loss"]["diff_loss_ratio"])
            total_loss += total_slop_loss * config["loss"]["slop_loss_ratio"]
            total_loss += total_diff_loss * config["loss"]["diff_loss_ratio"]
        else:
            losses.append(total_slop_loss - total_slop_loss)  # total_slop_loss * slop_loss_ratio)
            losses.append(total_diff_loss - total_diff_loss)  # total_diff_loss * diff_loss_ratio)
        total_loss += 0  # total_slop_loss * slop_loss_ratio
        total_loss += 0  # total_diff_loss * diff_loss_ratio
    return total_loss, losses, [coord_y_, class_y_]


def train(config):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    fwt = open("./train_icdar15_angle_loss_36p_{}bs_L2loss_{}points_{}backboneAlpha.log".format(config["batch_size"],
                                                                                                config["points_size"],
                                                                                                config["backbone_alpha"]
                                                                                                ), "a")
    out_path = "./output/train_icdar_15_angle_loss_36p_{}bs_L2loss_{}points_{}backboneAlpha/".format(
        config["batch_size"],
        config["points_size"],
        config["backbone_alpha"],
    )
    dataset = CardDataset(config["label_path"],
                          img_folder=config["img_folder_path"],
                          class_sizes=config["class_list"], batch_size=config["batch_size"])
    train_dataset, val_dataset = dataset.get_data()

    LDRModel = LDRNet(classification_list=config["class_list"], points_size=config["points_size"],
                      alpha=config["backbone_alpha"])
    epoch_size = len(train_dataset)
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [s * epoch_size for s in config["optimizer"]["bounds"]], config["optimizer"]["rates"])
    optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
    train_loss_results = []
    loc_loss_result = []
    class_loss_results = []
    for i in range(len(config["class_list"])):
        class_loss_results.append([])
    step = -1
    for epoch in range(config["num_epochs"]):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_loc_loss_avg = tf.keras.metrics.Mean()
        epoch_diff_loss_avg = tf.keras.metrics.Mean()
        epoch_slop_loss_avg = tf.keras.metrics.Mean()
        epoch_class_losses_avg = []
        for i in range(len(config["class_list"])):
            epoch_class_losses_avg.append(tf.keras.metrics.Mean())
        for x, y in train_dataset:
            # img = x[0].numpy()*255
            # img = img.astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(test_folder,str(step)+".png"),img)
            step += 1
            loss_value, loss_list, y_, grads = grad(config, LDRModel, x, y)
            optimizer.apply_gradients(zip(grads, LDRModel.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            for i in range(len(loss_list) - 3):
                epoch_class_losses_avg[i].update_state(loss_list[i])
            epoch_loc_loss_avg.update_state(loss_list[-3])
            epoch_slop_loss_avg.update_state(loss_list[-2])
            epoch_diff_loss_avg.update_state(loss_list[-1])
            msg_in = "Step {:06d}: Loss: {:.6f}, lr: {:.6f}, Loc_loss: {:.6f}, Slop_loss: {:.6f}, Diff_loss: {:.6f}".format(
                step, loss_value.numpy().mean(),
                optimizer._decayed_lr(
                    tf.float32).numpy(),
                loss_list[-3].numpy().mean(), loss_list[-2].numpy().mean(), loss_list[-1].numpy().mean())
            for i in range(len(config["class_list"])):
                # class_loss_results[i].append(epoch_class_losses_avg[i].result())
                msg_in += ",class_loss_{}:{:.6f}".format(i, loss_list[i].numpy().mean())
            print(msg_in)
            # print(optimizer._decayed_lr(tf.float32).numpy())
            fwt.write(msg_in + "\n")
            fwt.flush()
        train_loss_results.append(epoch_loss_avg.result())
        loc_loss_result.append(epoch_loc_loss_avg.result())
        msg = "Epoch {:06d}: Loss: {:.6f}, lr: {:.6f}, Loc_loss: {:.6f}".format(epoch,
                                                                                epoch_loss_avg.result(),
                                                                                optimizer._decayed_lr(
                                                                                    tf.float32).numpy(),
                                                                                epoch_loc_loss_avg.result())
        for i in range(len(config["class_list"])):
            class_loss_results[i].append(epoch_class_losses_avg[i].result())
            msg += ",class_loss_{}:{:.6f}".format(i, epoch_class_losses_avg[i].result())
        fwt.write(msg + "\n")
        fwt.flush()
        print(msg)

        if epoch % 5 == 0:
            epoch5_loss_avg = tf.keras.metrics.Mean()
            epoch5_loc_loss_avg = tf.keras.metrics.Mean()
            epoch5_slop_loss_avg = tf.keras.metrics.Mean()
            epoch5_diff_loss_avg = tf.keras.metrics.Mean()
            for x, y in val_dataset:
                loss_value, loss_list, y_ = loss(config, LDRModel, x, y, training=False,
                                                 coord_size=config["points_size"] * 2,
                                                 class_list=config["class_list"])
                epoch5_loc_loss_avg.update_state(loss_list[-3])
                epoch5_slop_loss_avg.update_state(loss_list[-2])
                epoch5_diff_loss_avg.update_state(loss_list[-1])
                epoch5_loss_avg.update_state(loss_value)
            msg_eval = "Epoch {:06d}: Test Loss: {:.6f}, Test Loc Loss:{:.6f}, Test Slop Loss:{:.6f}, Test Diff Loss:{:.6f}".format(
                epoch, epoch5_loss_avg.result(),
                epoch5_loc_loss_avg.result(), epoch5_slop_loss_avg.result(), epoch5_diff_loss_avg.result())
            fwt.write(msg_eval + "\n")
            fwt.flush()
            print(msg_eval)
            LDRModel.save(
                out_path + "{:06d}_{:.6f}_{:.6f}_{:.6f}".format(epoch, epoch_loss_avg.result(),
                                                                epoch5_loss_avg.result(),
                                                                optimizer._decayed_lr(tf.float32).numpy()))


if __name__ == "__main__":
    args = init_args()
    assert os.path.exists(args.config_file)
    config = yaml.load(open(args.config_file, 'rb'))
    train(config)
