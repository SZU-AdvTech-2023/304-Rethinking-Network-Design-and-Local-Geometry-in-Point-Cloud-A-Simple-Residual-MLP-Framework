import torch
from . import tgn_loss
from models.base_model import BaseModel
from loss_meter import LossMap


class PointMlpFirstModel(BaseModel):  # init看BaseModel类
    def get_loss(self, gt_seg_label_1, sem_1):
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, 1),
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)  # 设置train或者test

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:, :3, :].cuda()

        # centroids = batch_item[1].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()

        inputs = [points, seg_label]

        if phase == "train":
            output = self.module(inputs)
            # net = torch.nn.DataParallel(self.module,device_ids=[0, 1])
            # output = net(inputs)
        else:
            with torch.no_grad():
                # net = torch.nn.DataParallel(self.module)
                # output = net(inputs)
                output = self.module(inputs)
        loss_meter = LossMap()

        loss_meter.add_loss_by_dict(self.get_loss(
            seg_label,
            output["cls_pred"],
        )
        )

        if phase == "train":
            loss_sum = loss_meter.get_sum()  # 可以实现有权重的损失
            self.optimizer.zero_grad()  # 清空之前的梯度
            loss_sum.backward()
            self.optimizer.step()

        #return loss_meter, seg_label, output["cls_pred"]
        return loss_meter

    def infer(self, batch_idx, batch_item, **options):
        pass