import gen_utils as gu
import numpy as np
import torch
from sklearn.neighbors import KDTree
import os
import open3d as o3d
import time

class InferencePipeLine:
    def __init__(self, model):
        self.model = model

        self.scaler = 1.8
        self.shifter = 0.8

    def __call__(self, stl_path):
        DEBUG=False
        start = time.time()
        _, mesh = gu.read_txt_obj_ls(stl_path, ret_mesh=True, use_tri_mesh=True) #TODO slow processing speed
        print("read time: ", time.time()-start)
        vertices = np.array(mesh.vertices)
        n_vertices = vertices.shape[0]
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        org_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)      # 细分
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        start = time.time()
        sampled_feats = gu.resample_pcd([vertices.copy()], 24000, "fps")[0] #TODO slow processing speed
        print("resample time: ", time.time()-start)
        with torch.no_grad():
            start = time.time()
            #torch.cuda.empty_cache()
            input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)
            cls_pred = self.model([input_cuda_feats])['cls_pred']
            print("inference time: ", time.time()-start)
        cls_pred = cls_pred.argmax(axis=1)
        cls_pred = gu.torch_to_numpy(cls_pred)
        cls_pred[cls_pred>=9] += 2
        cls_pred[cls_pred>0] += 10
        start = time.time()
        tree = KDTree(sampled_feats[:,:3], leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)       # near_points大小为n，值为0-23999
        result_ins_labels = cls_pred.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)         # knn映射
        print("knn time: ", time.time()-start)
        if False:
            gu.print_3d(
                    gu.np_to_pcd_with_label(org_feats[:,:3], result_ins_labels), 
                )
            gu.print_3d(
                gu.np_to_pcd_with_label(org_feats[:,:3], result_sem_labels)
            )

        return {
            "sem":result_ins_labels.reshape(-1),
            "ins":result_ins_labels.reshape(-1),
        }       
