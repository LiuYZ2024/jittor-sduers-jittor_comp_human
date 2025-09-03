import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class Point_Transformer2(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer2, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = Point_Transformer_Last()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        batch_size, _, _ = x.size()
        
        # Conv1d + BN + ReLU
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, N]

        x = x.permute(0, 2, 1)  # [B, N, 64], points feature

        # Layer 1
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        # RuntimeError: ('new_xyz shape:', [16,512,3,], 'new_feature shape:', [16,512,32,128,])        
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512]
        
        # Layer 2
        feature = feature_0.permute(0, 2, 1)  # [B, 512, 128]
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]

        # Point transformer & fusion
        x = self.pt_last(feature_1, new_xyz)  # [B, 1024, 256]
        x = concat([x, feature_1], dim=1)  # [B, 1280, 256]
        x = self.conv_fuse(x)  # [B, 1024, 256]

        # ✅ 保存每点特征（transpose成 [B, N, D]，即 [B, 256, 1024] → [B, 256, 1024]）
        self._inter_feat = x.transpose(1, 2).stop_grad()  # [B, 256, 1024]，避免显存泄露
        
        # Global max pooling
        x = jt.max(x, dim=2)  # [B, 1024]
        x = x.view(batch_size, -1)

        # Fully connected head
        x = self.relu(self.bn6(self.linear1(x)))  # [B, 512]
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))  # [B, 256]
        x = self.dp2(x)
        x = self.linear3(x)  # [B, output_channels]

        return x

    def get_intermediate_features(self):
        return self._inter_feat  # [B, N=256, D=1024]
    
class Point_Transformer_pro(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_pro, self).__init__()
        self.conv1 = nn.Conv1d(3+22, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = Point_Transformer_Last()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x, vertices):
        """ 
            x is expected to be [B, 3+22, N]
        """
        xyz = vertices.permute(0, 2, 1)  # [B, N, 3]
        batch_size, _, _ = x.size()
        
        # Conv1d + BN + ReLU
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, N]

        x = x.permute(0, 2, 1)  # [B, N, 64], points feature

        # Layer 1
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512]
        
        # Layer 2
        feature = feature_0.permute(0, 2, 1)  # [B, 512, 128]
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]

        # Point transformer & fusion
        x = self.pt_last(feature_1, new_xyz)  # [B, 1024, 256]
        x = concat([x, feature_1], dim=1)  # [B, 1280, 256]
        x = self.conv_fuse(x)  # [B, 1024, 256]

        # ✅ 保存每点特征（transpose成 [B, N, D]，即 [B, 256, 1024] → [B, 256, 1024]）
        self._inter_feat = x.transpose(1, 2).stop_grad()  # [B, 256, 1024]，避免显存泄露
        
        # Global max pooling
        x = jt.max(x, dim=2)  # [B, 1024]
        x = x.view(batch_size, -1)

        # Fully connected head
        x = self.relu(self.bn6(self.linear1(x)))  # [B, 512]
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))  # [B, 256]
        x = self.dp2(x)
        x = self.linear3(x)  # [B, output_channels]

        return x

    def get_intermediate_features(self):
        return self._inter_feat  # [B, N=256, D=1024]


from models.MIA.models_ae import AutoEncoder

class Point_Transformer_ShapeEncoder(nn.Module):
    def __init__(self, output_channels=256):
        super(Point_Transformer_ShapeEncoder, self).__init__()
        # self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        
        self.encoder = AutoEncoder(depth=6, dim=256, num_inputs=2048, num_latents=512)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=512, out_channels=512)
        self.pt_last = Point_Transformer_Last_512()

        # self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(2560, 2048, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(2048),
                                   nn.LeakyReLU(scale=0.2))

        # self.linear1 = nn.Linear(1024, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        batch_size, _, _ = x.size()
        
        # # Conv1d + BN + ReLU
        # x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        # x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, N]
        
        # Shape Encoder from MIA
        x_t = x.permute(0, 2, 1)  # [B, N, 3]
        sample_feat, sample_pc = self.encoder.encode_sample(x_t) # [B, 512, 256], points feature, points_num = 512 now
 
        # # Layer 1
        # new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=sample_pc, points=sample_feat)
        # feature_0 = self.gather_local_0(new_feature)  # [B, 256, 512]
        
        # Layer 2
        # feature = feature_0.permute(0, 2, 1)  # [B, 512, 256]
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=sample_pc, points=sample_feat) 
        # raise RuntimeError("new_xyz shape:", new_xyz.shape, "new_feature shape:", new_feature.shape)
        feature_1 = self.gather_local_1(new_feature)  # [B, 512, 256] B, D, M

        # Point transformer & fusion
        x = self.pt_last(feature_1, new_xyz)  # [B, 4*512, 256]
        x = concat([x, feature_1], dim=1)  # [B, 5*512, 256]
        x = self.conv_fuse(x)  # [B, 4*512, 256]

        # # ✅ 保存每点特征（transpose成 [B, N, D]，即 [B, 256, 1024] → [B, 256, 1024]）
        # self._inter_feat = x.transpose(1, 2).stop_grad()  # [B, 256, 1024]，避免显存泄露
        
        # Global max pooling
        x_max = jt.max(x, dim=2) # [B, 2048]
        x_avg = jt.mean(x, dim=2) # [B, 2048]
        x_max_feature = x_max.view(batch_size, -1)
        x_avg_feature = x_avg.view(batch_size, -1)
        global_feature = concat((x_max_feature, x_avg_feature), 1) # [B, 4096]
        # global_feature = x_max_feature  # [B, 2048]

        # # Fully connected head
        # x = self.relu(self.bn6(self.linear1(x)))  # [B, 512]
        # x = self.dp1(x)
        # x = self.relu(self.bn7(self.linear2(x)))  # [B, 256]
        # x = self.dp2(x)
        # x = self.linear3(x)  # [B, output_channels]

        return global_feature

    # def get_intermediate_features(self):
    #     return self._inter_feat  # [B, N=256, D=1024]


class Point_Transformer_ShapeEncoder_4096(nn.Module):
    def __init__(self, output_channels=256):
        super(Point_Transformer_ShapeEncoder_4096, self).__init__()
        # self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)

        self.encoder = AutoEncoder(depth=6, dim=256, num_inputs=4096, num_latents=512)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=512, out_channels=512)
        self.pt_last = Point_Transformer_Last_512()

        # self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(2560, 2048, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(2048),
                                   nn.LeakyReLU(scale=0.2))

        # self.linear1 = nn.Linear(1024, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        batch_size, _, _ = x.size()
        
        # # Conv1d + BN + ReLU
        # x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        # x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, N]
        
        # Shape Encoder from MIA
        x_t = x.permute(0, 2, 1)  # [B, N, 3]
        sample_feat, sample_pc = self.encoder.encode_sample(x_t) # [B, 512, 256], points feature, points_num = 512 now
 
        # # Layer 1
        # new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=sample_pc, points=sample_feat)
        # feature_0 = self.gather_local_0(new_feature)  # [B, 256, 512]
        
        # Layer 2
        # feature = feature_0.permute(0, 2, 1)  # [B, 512, 256]
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=sample_pc, points=sample_feat) 
        # raise RuntimeError("new_xyz shape:", new_xyz.shape, "new_feature shape:", new_feature.shape)
        feature_1 = self.gather_local_1(new_feature)  # [B, 512, 256] B, D, M

        # Point transformer & fusion
        x = self.pt_last(feature_1, new_xyz)  # [B, 4*512, 256]
        x = concat([x, feature_1], dim=1)  # [B, 5*512, 256]
        x = self.conv_fuse(x)  # [B, 4*512, 256]

        # ✅ 保存每点特征（transpose成 [B, N, D]，即 [B, 256, 1024] → [B, 256, 1024]）
        self._inter_feat = x.transpose(1, 2).stop_grad()  # [B, 256, 1024]，避免显存泄露
        
        # Global max pooling
        x_max = jt.max(x, dim=2) # [B, 2048]
        x_avg = jt.mean(x, dim=2) # [B, 2048]
        x_max_feature = x_max.view(batch_size, -1)
        x_avg_feature = x_avg.view(batch_size, -1)
        global_feature = concat((x_max_feature, x_avg_feature), 1) # [B, 4096]
        # global_feature = x_max_feature  # [B, 2048]

        # # Fully connected head
        # x = self.relu(self.bn6(self.linear1(x)))  # [B, 512]
        # x = self.dp1(x)
        # x = self.relu(self.bn7(self.linear2(x)))  # [B, 256]
        # x = self.dp2(x)
        # x = self.linear3(x)  # [B, output_channels]

        return global_feature

    # def get_intermediate_features(self):
    #     return self._inter_feat  # [B, N=256, D=1024]
    
    
class Point_Transformer_ShapeEncoder_Dropout(nn.Module):
    def __init__(self, output_channels=256):
        super(Point_Transformer_ShapeEncoder_Dropout, self).__init__()

        # Shape Encoder from MIA
        self.encoder = AutoEncoder(depth=6, dim=128, num_inputs=1024, num_latents=512)

        # 局部特征聚合
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        # 最后一层 Point Transformer
        self.pt_last = Point_Transformer_Last_256()

        # 融合层：输入维度 5*512=2560
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )

        # 新增：对 global_feature 进行降维 + dropout 防止过拟合
        self.linear_proj = nn.Linear(2048, 1024)
        self.dropout_proj = nn.Dropout(0.3)

        # 中间特征缓存
        self._inter_feat = None

    def execute(self, x):
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        batch_size, _, _ = x.size()

        # AutoEncoder 编码得到点特征与采样点
        x_t = x.permute(0, 2, 1)  # [B, N, 3]
        sample_feat, sample_pc = self.encoder.encode_sample(x_t)  # [B, 512, 128], [B, 512, 3]

        # 局部区域采样和特征聚合
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=sample_pc, points=sample_feat)
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]

        # Transformer 特征处理
        x = self.pt_last(feature_1, new_xyz)  # [B, 4*256, 256]
        x = concat([x, feature_1], dim=1)  # [B, 5*256, 256]
        x = self.conv_fuse(x)  # [B, 1024, 256]

        # 中间特征保存（用于可视化或附加任务）
        self._inter_feat = x.transpose(1, 2).stop_grad()  # [B, 256, 1024]

        # Global Max + Avg Pooling
        x_max = jt.max(x, dim=2)  # [B, 1024]
        x_avg = jt.mean(x, dim=2)  # [B, 1024]
        global_feature = concat([x_max, x_avg], dim=1)  # [B, 2048]

        # 添加降维和 Dropout 防止过拟合
        # global_feature = self.linear_proj(global_feature)  # [B, 1024]
        # global_feature = self.dropout_proj(global_feature)

        return global_feature  # 输出全局特征

    def get_intermediate_features(self):
        return self._inter_feat  # [B, 256, 2048]



class Point_Transformer_Last_512(nn.Module):
    def __init__(self, channels=512):
        super(Point_Transformer_Last_512, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # raise RuntimeError()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        # xyz = self.pos_xyz(xyz)
        # xyz_embed = self.conv_pos(xyz)
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = x + xyz_embed

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x



class Point_Transformer_Last_256(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last_256, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # raise RuntimeError()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        # xyz = self.pos_xyz(xyz)
        # xyz_embed = self.conv_pos(xyz)
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = x + xyz_embed

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # raise RuntimeError()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        # xyz = self.pos_xyz(xyz)
        # xyz_embed = self.conv_pos(xyz)
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = x + xyz_embed

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x
    
    
class Point_Transformer(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2) # MA-Pool
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x



class SA_Layer(nn.Module): # 实际上是OA
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
      # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # Add a projection for xyz coordinates
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # Project xyz to the same channel dimension as x
        xyz_feat = self.xyz_proj(xyz)
        
        # Now we can safely add them
        x = x + xyz_feat
        
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # OA
        x = x + x_r
        return x

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = Point_Transformer()
    out_logits = network(input_points)
    print (out_logits.shape)

