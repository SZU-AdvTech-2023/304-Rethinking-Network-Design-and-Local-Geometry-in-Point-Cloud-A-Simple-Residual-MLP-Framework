import torch
import numpy as np
import ops_utils as ou
import gen_utils as gu
from .tsg_centroid_module import get_model as get_centroid_module
from .tsg_seg_module import get_model as get_seg_module
from sklearn.cluster import DBSCAN
from external_libs.pointnet2_utils.pointnet2_utils import square_distance

class TSegNetModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.cent_module = get_centroid_module()
        self.seg_module = get_seg_module()
        
        
        if self.config["run_tooth_segmentation_module"]:
            self.run_seg_module = True
        else:
            self.run_seg_module = False
        
    def get_ddf(self, cropped_coord, center_points):
        B, N, C  = cropped_coord.shape
        
        center_points = torch.from_numpy(center_points).cuda()
        ddf = square_distance(cropped_coord, center_points.permute(1,0,2))
        ddf = torch.sqrt(ddf)
        ddf *= (-4)
        ddf = torch.exp(ddf)
        ddf = ddf.permute(0,2,1)
        return ddf

    def forward(self, inputs):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        # 对输入点云的坐标进行归一化
        # x_min = inputs[0].min(dim=2)
        # x_max = inputs[0].max(dim=2)
        # print("min: ", x_min)
        # print("max: ", x_max)
        # y_min = inputs[:,[1],:].min(dim=2)
        # y_max = inputs[:,[1],:].max(dim=2)
        # print("y_min: ", y_min)
        # print("y_max: ", y_max)
        # z_min = inputs[:,[2],:].min(dim=2)
        # z_max = inputs[:,[2],:].max(dim=2)
        # print("z_min: ", z_min)
        # print("z_max: ", z_max)


        outputs = {}

        l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result = self.cent_module(inputs[0])
        outputs.update({
            "l0_points": l0_points, 
            "l3_points":l3_points, 
            "l0_xyz": l0_xyz, 
            "l3_xyz": l3_xyz, 
            "offset_result":offset_result, 
            "dist_result":dist_result
        })

        if not self.run_seg_module: return outputs

        moved_points = gu.torch_to_numpy(l3_xyz + offset_result).T.reshape(-1,3)
        moved_points = moved_points[gu.torch_to_numpy(dist_result).reshape(-1)<0.3,:]
        print("moved_points.shape: ", moved_points.shape)
        #dbscan_results = DBSCAN(eps=0.05, min_samples=3).fit(moved_points, 3)
        dbscan_results = DBSCAN(eps=0.05, min_samples=3).fit(moved_points,3)
        #gu.print_3d(gu.np_to_pcd_with_label(moved_points, dbscan_results.labels_))
        #print("dbscan_results_count: ", np.unique(dbscan_results.labels_))
        center_points = []
        for label in np.unique(dbscan_results.labels_):
            if label == -1:
                #center_points.append(np.array([-10.0,-10.0,-10.0],dtype=np.float32))
                continue            # -1说明是噪声点
            center_points.append(moved_points[dbscan_results.labels_==label].mean(axis=0))      # 相同类的取平均值就是预测的中心
        center_points = np.array(center_points)
        #print("center_points: ", center_points)

        if len(center_points) == 0:
            center_points = np.array([[-10.0,-10.0,-10.0]],dtype=np.float32)
        #print("center_points.shape: ", center_points.shape)
        center_points = center_points[None,:,:]
        #gu.print_3d(gu.np_to_pcd(moved_points,color=[0,1,0]), center_points[0])
        
        rand_indexes = np.random.permutation(center_points.shape[1])[:8]                    #  随机取8个聚类中心
        center_points = center_points.transpose(0,2,1)[:,:,rand_indexes].transpose(0,2,1)   # 聚类中心的index是随机的，继续打乱
        
        nn_crop_indexes = ou.get_nearest_neighbor_idx(gu.torch_to_numpy(l0_xyz.permute(0,2,1)), center_points, 3072)
        #print("nn_crop_indexes shape : ", len(nn_crop_indexes),len(nn_crop_indexes[0]))
        cropped_input_ls = ou.get_indexed_features(inputs[0], nn_crop_indexes)
        cropped_feature_ls = ou.get_indexed_features(l0_points, nn_crop_indexes)
        ddf = self.get_ddf(cropped_input_ls[:,:3,:].permute(0,2,1), center_points)

        cluster_gt_seg_label = ou.get_indexed_features(inputs[1], nn_crop_indexes)  # output: [B*cluster_num, 1, 3072]

        cropped_feature_ls = torch.cat([cropped_input_ls[:,:3,:], cropped_feature_ls, ddf], axis=1)     # 3 + 32 + 1， 作为分割模块的输入
        # 注意，经过上面处理后，batchsize发生了变化，变为了原来的cluster_num倍
        pd_1, weight_1, pd_2, id_pred = self.seg_module(cropped_feature_ls)
        outputs.update({
            "pd_1":pd_1, "weight_1":weight_1, "pd_2":pd_2, "id_pred":id_pred, "center_points": center_points, "cluster_gt_seg_label": cluster_gt_seg_label,
            "cropped_feature_ls": cropped_feature_ls
        })
        return outputs