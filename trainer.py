import torch
import wandb
from loss_meter import LossMeter
from math import inf
import numpy as np
import time
class Trainer:
    def __init__(self, config = None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config
        self.model = model
        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        if config["wandb"]["wandb_on"]:
            wandb.init(
            entity=self.config["wandb"]["entity"],
            project=self.config["wandb"]["project"],
            notes=self.config["wandb"]["notes"],
            tags=self.config["wandb"]["tags"],
            name=self.config["wandb"]["name"],
            config=self.config,
            )
        self.best_val_loss = inf

    def train(self, epoch, data_loader):
        total_loss_meter = LossMeter()
        step_loss_meter =  LossMeter()
        pre_step = self.step_count
        log_path = self.config["log_path"]
        #print("log_path:", log_path)
        #with open(log_path, "w") as file:
        loss_sum = {}
        for batch_idx, batch_item in enumerate(data_loader):
            #loss,gt, pred = self.model.step(batch_idx, batch_item, "train")          # 包括了参数更新
            loss = self.model.step(batch_idx, batch_item, "train")
            #print("batch_item", batch_item['feat'].shape)
            #print("train: ","epoch: ", epoch, "/200 ", "batch_idx: ", batch_idx, "/",len(data_loader))
            #file.write("train: "+"epoch: "+str(epoch)+"/200 "+"batch_idx: "+str(batch_idx)+"/"+str(len(data_loader))+"\n")
            #pred.permute(0,2,1)
            # pred = pred.detach().cpu()
            # print("pred: ",pred.shape)
            # #print("pred[0][0]: ",pred[0][0])
            # #print("pred:", pred.transpose(2,1)[0][0])
            # #pred1 = torch.max(pred, axis=1,keepdim=True)[0].view(-1)
            # pred = torch.argmax(pred.transpose(2,1), axis=2).view(-1)
            # #print("pred1:",pred1[0:17])
            # gt = gt.detach().cpu()
            # gt = gt.view(-1)
            # gt = gt + 1     # -1 - 15 => 0 - 16
            # IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt, pred, pred)
            # print("IoU", IoU, "F1(TSA)", F1, "Acc", Acc,"SEM_ACC(TIR)", SEM_ACC)
            # file.write("IoU"+str(IoU)+"F1(TSA)"+str(F1)+"Acc"+str(Acc)+"SEM_ACC(TIR)"+str(SEM_ACC)+"\n")
            #torch.cuda.empty_cache()            # 去掉之后出现 RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
            #print(loss.get_loss_dict_for_print("step"))
            for loss_name in loss.get_loss_dict_for_print("").keys():
                if loss_name not in loss_sum.keys():
                    loss_sum[loss_name] = 0.0

                loss_sum[loss_name] += loss.get_loss_dict_for_print("")[loss_name]
            #file.write(str(loss.get_loss_dict_for_print("step")))
            #file.write("\n")
            if ((batch_idx+1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or (self.step_count == pre_step and batch_idx == len(data_loader)-1):
                if self.config["wandb"]["wandb_on"]:
                    wandb.log(step_loss_meter.get_avg_results(), step=self.step_count)
                    wandb.log({"step_lr": self.model.scheduler.get_last_lr()[0]}, step = self.step_count)
                self.step_count +=1
                self.model.scheduler.step(self.step_count)          # 更新lr
                step_loss_meter.init()
        #file.close()
        file = open(log_path, "a")
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("time: ", time_now)
        print("train: ", "epoch: ", epoch, "/200")
        file.write("train: " + "epoch: " + str(epoch) + "/200" + "\n")
        file.write("time: " + time_now + "\n")
        for loss_name in loss_sum.keys():
            loss_sum[loss_name] /= len(data_loader)
            print(loss_name, ": ", loss_sum[loss_name],end=' ')
            file.write(loss_name + ": " + str(loss_sum[loss_name]) + " ")
        print()
        file.write("\n")
        file.close()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(total_loss_meter.get_avg_results(), step = self.step_count)
            self.train_count += 1
        self.model.save("train")

    def test(self, epoch, data_loader, save_best_model):
        total_loss_meter = LossMeter()
        log_path = self.config["val_log_path"]
        loss_sum = {}

        for batch_idx, batch_item in enumerate(data_loader):
            #loss,gt, pred = self.model.step(batch_idx, batch_item, "test")
            loss= self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))
            for loss_name in loss.get_loss_dict_for_print("").keys():
                if loss_name not in loss_sum.keys():
                    loss_sum[loss_name] = 0.0

                loss_sum[loss_name] += loss.get_loss_dict_for_print("")[loss_name]
            #print("val: ","epoch: ", epoch, "/200 ", "batch_idx: ", batch_idx, "/",len(data_loader))
            #print("train: ", "epoch: ", epoch, "/200 ", "batch_idx: ", batch_idx, "/", len(data_loader))
            #file.write("train: " + "epoch: " + str(epoch) + "/200 " + "batch_idx: " + str(batch_idx) + "/" + str(
            #   len(data_loader)) + "\n")

            # pred = pred.detach().cpu()
            # pred = torch.argmax(pred.transpose(2, 1), axis=2).view(-1)
            # gt = gt.detach().cpu()
            # gt = gt.view(-1)
            # gt = gt + 1  # -1 - 15 => 0 - 16
            # IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt, pred, pred)
            # print("IoU", IoU, "F1(TSA)", F1, "Acc", Acc, "SEM_ACC(TIR)", SEM_ACC)
            # file.write("IoU" + str(IoU) + "F1(TSA)" + str(F1) + "Acc" + str(Acc) + "SEM_ACC(TIR)" + str(SEM_ACC) + "\n")

            #print(loss.get_loss_dict_for_print("step"))
            #file.write(str(loss.get_loss_dict_for_print("step")))
            #file.write("\n")
        avg_total_loss = total_loss_meter.get_avg_results()
        file = open(log_path, "a")
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("time: ", time_now)
        print("val: ", "epoch: ", epoch, "/200")
        file.write("time: " + time_now + "\n")
        file.write("val: " + "epoch: " + str(epoch) + "/200" + "\n")

        for loss_name in loss_sum.keys():
            loss_sum[loss_name] /= len(data_loader)
            print(loss_name, ": ", loss_sum[loss_name],end=' ')
            file.write(loss_name + ": " + str(loss_sum[loss_name]) + " ")
        print()
        file.write("\n")
        file.close()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step = self.step_count)
            self.val_count+=1

        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.model.save("val")

    # def train_depr(self):
    #     total_loss = 0
    #     step_loss = 0
    #     for batch_idx, batch_item in enumerate(self.train_loader):
    #         loss = self.model.step(batch_idx, batch_item, "train")
    #         total_loss += loss
    #         step_loss += loss
    #         if (batch_idx+1) % self.config["tr_set"]["schedueler_step"] == 0:
    #             self.model.scheduler.step()
    #             step_loss /= self.config["tr_set"]["schedueler_step"]
    #             if self.config["wandb"]["wandb_on"]:
    #                 wandb.log({"step_train_loss":step_loss})
    #             step_loss = 0
    #     total_loss /= len(self.train_loader)
    #     if self.config["wandb"]["wandb_on"]:
    #         wandb.log({"train_loss": total_loss})
    #     self.model.save("train")
    #
    # def test_depr(self):
    #     total_loss = 0
    #     for batch_idx, batch_item in enumerate(self.val_loader):
    #         loss = self.model.step(batch_idx, batch_item, "test")
    #         total_loss += loss
    #     total_loss /= len(self.val_loader)
    #     if self.config["wandb"]["wandb_on"]:
    #         wandb.log({"val_loss": total_loss})
    #
    #     if self.best_val_loss > total_loss:
    #         self.best_val_loss = total_loss
    #         self.model.save("val")
    
    def run(self):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        for epoch in range(200):

            self.train(epoch, train_data_loader)
            self.test(epoch, val_data_loader, True)