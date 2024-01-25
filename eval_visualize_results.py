import sys
import os
from trimesh import PointCloud
sys.path.append(os.getcwd())
from glob import glob
#import gen_utils as gu
import new_gu as gu
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--mesh_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj", type=str)
parser.add_argument('--gt_json_path', default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances/013FHA7K/013FHA7K_lower.json" ,type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
args = parser.parse_args()
def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:      # label的数量
        #instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels==ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)        # 获取与pred对应index的groundtruth的label以及其数量，取最大者
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 += 2*(precision*recall) / (precision + recall)
        IOU += TP / (FP+TP+FN)
        IOU_arr.append(TP / (FP+TP+FN))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:     # 分割类别为8还是17
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1
        #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr

allIoU = 0.0
allF1 = 0.0
allAcc = 0.0
cnt = 0
for root, dirs, files in os.walk(args.pred_json_path):
    for file in files:
        if file.endswith('.json'):
            folder_name = file[:-11]
            file_path = os.path.join(root, file)
            pred_loaded_json = gu.load_json(file_path)

            pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)
            if pred_loaded_json['jaw'] == 'lower':
                pred_labels -= 20
            pred_labels[
                pred_labels // 10 == 1] %= 10  # [1,....,8] => [11,....,18] or [31,....,38] , [9,....,16]  => [21,....,28] or [41,....,48]
            pred_labels[pred_labels // 10 == 2] = (pred_labels[pred_labels // 10 == 2] % 10) + 8

            gt_path = os.path.join(args.gt_json_path, folder_name,file)
            print(gt_path)
            gt_loaded_json = gu.load_json(gt_path)
            gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)
            if gt_loaded_json['jaw'] == 'lower':
                gt_labels -= 20
            gt_labels[
                gt_labels // 10 == 1] %= 10  # [1,....,8] => [11,....,18] or [31,....,38] , [9,....,16]  => [21,....,28] or [41,....,48]
            gt_labels[gt_labels // 10 == 2] = (gt_labels[gt_labels // 10 == 2] % 10) + 8
            IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
            print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)
            allIoU += IoU
            allF1 += F1
            allAcc += Acc
            cnt += 1
            mesh_path = os.path.join(args.mesh_path, folder_name, file[:-5]+".obj")
            _, mesh = gu.read_txt_obj_ls(mesh_path, ret_mesh=True, use_tri_mesh=True)
            # gu.print_3d(gu.get_colored_mesh(mesh, gt_labels)) # color is random
            gu.print_3d(gu.get_colored_mesh(mesh, pred_labels)) # color is random

print("allIoU", allIoU/len(files), "allF1", allF1/len(files), "allAcc", allAcc/len(files))