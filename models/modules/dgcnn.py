from torch import nn
import torch
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k)
            #idx = knn(x[:, 6:], k=k)
            #idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


# class Transform_Net(nn.Module):
#     def __init__(self, num_features):
#         super(Transform_Net, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(num_features * 2, 128, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(128),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(256),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
#                                    nn.BatchNorm1d(1024),
#                                    nn.LeakyReLU(negative_slope=0.2))
#
#         self.linear1 = nn.Linear(1024, 512, bias=True)
#         #self.bn3 = nn.BatchNorm1d(512)
#         self.linear2 = nn.Linear(512, 256, bias=True)
#         #self.bn4 = nn.BatchNorm1d(256)
#
#         self.transform = nn.Linear(256, num_features * num_features, bias=True)
#         nn.init.constant_(self.transform.weight, 0)
#         nn.init.eye_(self.transform.bias.view(num_features, num_features))
#         self.num_features = num_features
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         x = self.conv1(x)                       # (batch_size, in_channel, num_points, k) -> (batch_size, 128, num_points, k)
#         x = self.conv2(x)                       # (batch_size, 128, num_points, k) -> (batch_size, 256, num_points, k)
#         x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
#
#         x = self.conv3(x)                       # (batch_size, 256, num_points) -> (batch_size, 1024, num_points)
#         x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)
#
#         x = F.leaky_relu(self.linear1(x),negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
#         x = F.leaky_relu(self.linear2(x), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
#
#         x = self.transform(x)                   # (batch_size, 256) -> (batch_size, num_features^2)
#         x = x.view(batch_size, self.num_features, self.num_features)            # (batch_size, num_features) -> (batch_size, num_features, num_features)
#
#         return x


class get_model(nn.Module):
    def __init__(self, config):
        super(get_model, self).__init__()
        drop_out_ratio = 0.5
        emb_dims = 1024
        self.k = 20
        input_dim = 6
        
        self.scale = 1
        # self.transform_net = Transform_Net(input_dim)
        self.bn1 = nn.BatchNorm2d(64*self.scale)
        self.bn2 = nn.BatchNorm2d(64*self.scale)
        self.bn3 = nn.BatchNorm2d(64*self.scale)
        self.bn4 = nn.BatchNorm2d(64*self.scale)
        self.bn5 = nn.BatchNorm2d(64*self.scale)
        self.bn6 = nn.BatchNorm1d(emb_dims*self.scale)
        self.bn7 = nn.BatchNorm1d(512*self.scale)
        #self.bn7 = nn.BatchNorm1d(256 * self.scale)
        self.bn8 = nn.BatchNorm1d(256 * self.scale)
        #self.bn9 = nn.BatchNorm1d(128 * self.scale)

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2*self.scale, 64*self.scale, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192*self.scale, emb_dims*self.scale, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216*self.scale, 512*self.scale, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(1216*self.scale, 256*self.scale, kernel_size=1, bias=False),
        #                             self.bn7,
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512*self.scale, 256*self.scale, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(256*self.scale, 256*self.scale, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv9 = nn.Sequential(nn.Conv1d(256*self.scale, 128*self.scale, kernel_size=1, bias=False),
        #                            self.bn9,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=drop_out_ratio)
        #self.dp2 = nn.Dropout(p=drop_out_ratio)
        self.cls_conv = nn.Conv1d(256, 17, kernel_size=1, bias=False)
        #self.cls_conv = nn.Conv1d(128, 17, kernel_size=1, bias=False)
        #self.offset_conv = nn.Conv1d(256, 3, kernel_size=1, bias=False)
        #self.dist_conv = nn.Conv1d(256, 1, kernel_size=1, bias=False)

        #nn.init.zeros_(self.offset_conv.weight)
        #nn.init.zeros_(self.dist_conv.weight)

    def forward(self, x_in):
        x = x_in[0]
        #l0_xyz = x[:,:3,:]
        batch_size = x.size(0)
        num_points = x.size(2)

        # idx_xyz = knn(x[:, :3, :], self.k)
        # x0 = get_graph_feature(x, k=self.k, idx=idx_xyz) # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)              # (batch_size, 3, 3)
        # x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)


        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)

        x = self.conv8(x)
        x = self.dp1(x)
        #x = self.dp2(x)
        #x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 128, num_points)

        cls_result = self.cls_conv(x)                       # (batch_size, 128, num_points) -> (batch_size, 17, num_points)
        #offset_result = self.offset_conv(x)
        #dist_result = self.dist_conv(x)
        return cls_result

        # return {
        #     "cls_pred" : cls_result
        # }


class DgcnnFirstModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_sem_model = get_model(self.config)

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        cls_pred = self.first_sem_model(inputs)
        outputs = {
            "cls_pred": cls_pred
        }
        return outputs