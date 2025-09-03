import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from math import sqrt

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class SimpleSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res
    
###################################### pct2 #########################################
class SimpleSkinModel2(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer2(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res
#####################################################################################


######################################unirig#########################################
import math

class FrequencyPositionalEmbedding(nn.Module):
    
    def __init__(self, num_freqs=6, logspace=True, input_dim=3, include_input=True, include_pi=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        if logspace:
            freqs = 2.0 ** jt.arange(num_freqs).float32()
        else:
            freqs = jt.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)

        if include_pi:
            freqs *= math.pi

        self.frequencies = freqs
        self.out_dim = self._get_dims(input_dim)

    def _get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def execute(self, x):
        if self.num_freqs > 0:
            x_exp = x.unsqueeze(-1) * self.frequencies
            embed = x_exp.reshape(*x.shape[:-1], -1)

            sin = jt.sin(embed)
            cos = jt.cos(embed)

            if self.include_input:
                return jt.concat([x, sin, cos], dim=-1)
            else:
                return jt.concat([sin, cos], dim=-1)
        else:
            return x

from jittor import attention

class ResidualCrossAttn(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        # help(attention.MultiheadAttention)

        self.attention = attention.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            
        )

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim),
        )
        
        # self.last_attn_weights = None  # ⬅️ 新增，用于保存注意力图

    
    def execute(self, q, kv):
        residual = q

        # 默认 MultiheadAttention 内部已经做了缩放，无需手动 attn_scale
        attn_output, attn_weights = self.attention(
            query=q,
            key=kv,
            value=kv,
            need_weights=True  # 默认就返回 attn map，保险加一下
        )

        # self.last_attn_weights = attn_weights  # [B, num_heads, Q, K]
        # print(self.last_attn_weights[0][0])
        x = self.norm1(residual + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x


class BoneEncoder(nn.Module):
    def __init__(
        self,
        feat_bone_dim: int,
        feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_attn: int,
    ):
        super().__init__()
        self.feat_bone_dim = feat_bone_dim
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_attn = num_attn


        self.position_embed = FrequencyPositionalEmbedding(input_dim=self.feat_bone_dim)

        self.bone_encoder = nn.Sequential(
            self.position_embed,
            nn.Linear(self.position_embed.out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

        self.attn = nn.ModuleList()
        for _ in range(self.num_attn):
            self.attn.append(ResidualCrossAttn(feat_dim, self.num_heads))

    def execute(
        self,
        base_bone,        # (B, J, C)
        min_coord,        # (B, 3)
        global_latents,   # (B, T, D)
        num_bones=None,        # (B,)
        parents=None,          # (B, J)
    ):
        B, J, C = base_bone.shape

        # Normalize base_bone by subtracting min_coord
        x_input = base_bone - min_coord.reshape(-1, 1, 3)


        # Flatten and pass through bone_encoder MLP
        x = self.bone_encoder(x_input.reshape(-1, C)).reshape(B, J, -1)  # (B, J, D)

        # Concatenate bone features with global latent tokens
        latents = jt.concat([x, global_latents], dim=1)  # (B, J+T, D)

        for attn in self.attn:
            x = attn(x, latents)  # Cross-attn between bone and global tokens

        return x  # (B, J, D)        
    

class SkinweightPred(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),

            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),

            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),

            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),

            nn.Linear(mlp_dim, 1),
        )

    def execute(self, x):
        return self.net(x)


class UniSkinModel(nn.Module):
    def __init__(self, feat_dim, num_joints):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        self.pct = Point_Transformer2(output_channels=feat_dim)

        self.pos_embed = FrequencyPositionalEmbedding(input_dim=3, include_input=True)
        self.vertex_encoder = MLP(self.pos_embed.out_dim + feat_dim, feat_dim)
        self.bone_encoder = BoneEncoder(3, feat_dim, embed_dim=128, num_heads=4, num_attn=2)
        self.cross_attn = ResidualCrossAttn(feat_dim, num_heads=4)
        self.skin_pred = SkinweightPred(in_dim=feat_dim, mlp_dim=256)

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        _, J, _ = joints.shape

        # frequency embed
        v_pos = self.pos_embed(vertices)
        j_pos = self.pos_embed(joints)

        # global latent from PointTransformer
        shape_latent = self.pct(vertices.permute(0, 2, 1))  # (B, feat_dim)

        # vertex and joint initial embedding
        vertex_input = jt.concat([v_pos, shape_latent.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        vertex_feat = self.vertex_encoder(vertex_input)

        # parents = jt.array([None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,])
        # bone encoder
        min_coord = jt.min(vertices, dim=1)[0]  # (B, 3)

        bone_feat = self.bone_encoder(
            joints,
            min_coord,
            global_latents=shape_latent.unsqueeze(1))

        # Residual cross-attn
        vertex_feat = self.cross_attn(vertex_feat, kv=bone_feat)

        # predict per (vertex, joint)
        vertex_feat_expand = vertex_feat.unsqueeze(2).repeat(1, 1, J, 1)
        bone_feat_expand = bone_feat.unsqueeze(1).repeat(1, N, 1, 1)
        pred_input = vertex_feat_expand * bone_feat_expand  # 或 concat
        weights = self.skin_pred(pred_input).squeeze(-1)  # (B, N, J)
        weights = nn.softmax(weights, dim=-1)

        return weights


######################################################################################


######################################uni_pct_seg#####################################

from PCT.networks.seg.pct_partseg_uniseg import Point_Transformer_PartSeg

class MLP_pctseg(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.point_transformer_partseg = Point_Transformer_PartSeg(part_num=22)
    
    def execute(self, x):
        B = x.shape[0]
        x = x.reshape(-1, self.input_dim)
        res = self.encoder(x)
        res = res.reshape(B, -1, self.output_dim)
        return res # [B, N, output_dim]


class UniSkinModel_seg(nn.Module):
    def __init__(self, feat_dim, num_joints):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        self.pct = Point_Transformer2(output_channels=feat_dim)

        self.pos_embed = FrequencyPositionalEmbedding(input_dim=3, include_input=True)
        self.vertex_encoder = MLP(52 + feat_dim, feat_dim)
        self.point_transformer_partseg = Point_Transformer_PartSeg(joint_num=52)
        self.bone_encoder = BoneEncoder(3, feat_dim, embed_dim=128, num_heads=4, num_attn=2)
        self.cross_attn = ResidualCrossAttn(feat_dim, num_heads=4)
        self.skin_pred = SkinweightPred(in_dim=feat_dim, mlp_dim=256)

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        _, J, _ = joints.shape

        # frequency embed
        # v_pos = self.pos_embed(vertices) # [B,N,39,]
        # j_pos = self.pos_embed(joints)

        # global latent from PointTransformer
        shape_latent = self.pct(vertices.permute(0, 2, 1))  # (B, feat_dim)
        # pct segmentation
        vertex_seg = self.point_transformer_partseg(vertices.permute(0,2,1)) # [B, 22, N]
        # vertex and joint initial embedding
        vertex_input = jt.concat([vertex_seg.permute(0,2,1), shape_latent.unsqueeze(1).repeat(1, N, 1)], dim=-1) # vertex_input:  [B,N,278,] 278=256+22
        
        vertex_feat = self.vertex_encoder(vertex_input)
        # ('vertex_feat: ', [B,N,256,])

        # parents = jt.array([None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,])
        # bone encoder
        min_coord = jt.min(vertices, dim=1)[0]  # (B, 3)

        bone_feat = self.bone_encoder(
            joints,
            min_coord,
            global_latents=shape_latent.unsqueeze(1))

        # Residual cross-attn
        vertex_feat = self.cross_attn(vertex_feat, kv=bone_feat)

        # predict per (vertex, joint)
        vertex_feat_expand = vertex_feat.unsqueeze(2).repeat(1, 1, J, 1)
        bone_feat_expand = bone_feat.unsqueeze(1).repeat(1, N, 1, 1)
        pred_input = vertex_feat_expand * bone_feat_expand  # 或 concat
        weights = self.skin_pred(pred_input).squeeze(-1)  # (B, N, J)
        weights = nn.softmax(weights, dim=-1)

        return weights


######################################################################################


# Factory function to create models
def create_model(model_name='pct', feat_dim=256, **kwargs):
    if model_name == "pct":
        return SimpleSkinModel(feat_dim=feat_dim, num_joints=22)
    if model_name == "pct2":
        return SimpleSkinModel2(feat_dim=feat_dim, num_joints=22)
    if model_name == "unirig":
        return UniSkinModel(feat_dim=feat_dim,num_joints=22)
    if model_name == "uniseg":
        return UniSkinModel_seg(feat_dim=feat_dim,num_joints=52)
    raise NotImplementedError()