import tensorflow as tf
import os
import cv2
import numpy as np
import Polygon


def sort_gt(gt):
    '''
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt:
    :return: sorted gt
    '''
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    return np.asarray((tl, tr, br, bl))


def intersection_with_correction_smart_doc_implementation(gt, prediction, img):
    # Reference : https://github.com/jchazalon/smartdoc15-ch1-eval

    gt = sort_gt(gt)
    prediction = sort_gt(prediction)
    img1 = np.zeros_like(img)
    cv2.fillConvexPoly(img1, gt, (255, 0, 0))

    target_width = 2100
    target_height = 2970
    # Referential: (0,0) at TL, x > 0 toward right and y > 0 toward bottom
    # Corner order: TL, BL, BR, TR
    # object_coord_target = np.float32([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]])
    object_coord_target = np.array(
        np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]]))
    # print (gt, object_coord_target)
    H = cv2.getPerspectiveTransform(gt.astype(np.float32).reshape(-1, 1, 2), object_coord_target.reshape(-1, 1, 2))

    # 2/ Apply to test result to project in target referential
    test_coords = cv2.perspectiveTransform(prediction.astype(np.float32).reshape(-1, 1, 2), H)

    # 3/ Compute intersection between target region and test result region
    # poly = Polygon.Polygon([(0,0),(1,0),(0,1)])
    poly_target = Polygon.Polygon(object_coord_target.reshape(-1, 2))
    poly_test = Polygon.Polygon(test_coords.reshape(-1, 2))
    poly_inter = poly_target & poly_test

    area_target = poly_target.area()
    area_test = poly_test.area()
    area_inter = poly_inter.area()

    area_union = area_test + area_target - area_inter
    # Little hack to cope with float precision issues when dealing with polygons:
    #   If intersection area is close enough to target area or GT area, but slighlty >,
    #   then fix it, assuming it is due to rounding issues.
    area_min = min(area_target, area_test)
    if area_min < area_inter and area_min * 1.0000000001 > area_inter:
        area_inter = area_min
        print("Capping area_inter.")

    jaccard_index = area_inter / area_union
    return jaccard_index


def cal_mJI(label_path, img_folder, net):
    fr = open(label_path, "r")
    files_gt = fr.readlines()
    imgs = [f.split(",")[0] for f in files_gt]
    print("done")
    files = os.listdir(img_folder)
    sum_ji = 0
    p_count = 0
    result_iou = {}
    for i in range(len(files)):

        f = files[-i]
        if f not in imgs:
            continue
        index = imgs.index(f)
        coord_gt = files_gt[index].split(",")[1:9]
        if f.endswith("png"):
            img = cv2.imread(os.path.join(img_folder, f))
            h, w = img.shape[0:2]
            img = img.astype(np.float32)
            img = cv2.resize(img, (224, 224))
            img2 = img
            img2 = img2 / 255.0

            '''x = net.base_model(np.expand_dims(img2,0))
            x[4] = net.avgpool4(x[4])
            final_shape = x[4].shape[-1]
            x = [net.flatten(i) for i in x]
            for i in range(4):
                test = tf.expand_dims(x[i],-1)
                test = tf.expand_dims(test,-1)
                test = tf.image.resize(test,[final_shape,1])
                x[i] = tf.squeeze(test)
            x = [x[i]*net.final_weights[i] for i in range(5)]
            x = tf.reduce_sum(x,axis=0)
            out = net.dense(x)
            result = out[0]'''
            result = net([img2,img2])#.numpy()[0]
            pred_coord = np.copy(result[0].numpy()[0])
            pred_coord[0::2] *= w
            pred_coord[1::2] *= h
            pred_coord = [round(float(x)) for x in pred_coord]
            coord_gt = [round(float(x)) for x in coord_gt]
            gt = np.array(coord_gt[0:8]).reshape((4, 2))
            pred = np.array(pred_coord[0:8]).reshape((4, 2))
            ji = intersection_with_correction_smart_doc_implementation(gt, pred, img)
            #print("------------------------ji is:", ji)
            result_iou[f] = ji
            sum_ji += ji
            p_count += 1
    return sum_ji / p_count


if __name__ == "__main__":
    import os
    import time
    import sys
    argvs = sys.argv
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    net_tag = sys.argv[1]
    net_folder = "./output/"+net_tag
    png_folder = "" #path to your img foler
    txt_path = "" #path to your test data label
    record_path = "./{}.txt".format(net_tag) #test result
    if os.path.exists(record_path):
        fr = open(record_path,"r")

        lines = fr.readlines()
        nets = [line.split(",")[0] for line in lines]
        fr.close()
    else:
        nets = []
    fw = open(record_path,"a+")
    
    total_result = []
    for f in os.listdir(net_folder):
        net_path = os.path.join(net_folder, f)
        if not os.path.isdir(net_path):
            continue
        if f in nets:
            continue
        print(net_path)
        net = tf.keras.models.load_model(net_path)
        start_t = time.time()
        result = cal_mJI(txt_path,png_folder,net)
        time_consumed = time.time() - start_t
        print(time_consumed)
        
        cur_result = "{},{}\n".format(f,result)
        total_result.append(cur_result)
        fw.writelines(cur_result)
        fw.flush()
        print(cur_result)
    
