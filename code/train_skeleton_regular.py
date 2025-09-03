import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from dataset.format import num_joints, parents
from models.skeleton import create_model

from models.metrics import J2J

# Set Jittor flags
jt.flags.use_cuda = 1

# 手部关节索引
HAND_JOINTS = list(range(22, 52))
# 大骨架关节索引
BODY_JOINTS = [i for i in range(0, 22)]

def compute_bone_length_loss(pred_joints, parents_list):
    """
    骨长约束 loss
    pred_joints: [B, num_joints, 3]
    """
    loss = 0.0
    for i, p in enumerate(parents_list):
        if p is None:
            continue
        # 只对身体骨架和手部分别加
        if i in BODY_JOINTS or i in HAND_JOINTS:
            bone_vec = pred_joints[:, i] - pred_joints[:, p]
            bone_len = jt.norm(bone_vec, dim=-1)
            # 期望骨长使用batch均值作为动态参考
            mean_len = jt.mean(bone_len, dim=0)
            loss += jt.mean((bone_len - mean_len)**2)
    return loss / len(parents_list)

def compute_hand_direction_loss(pred_joints):
    """
    手部方向约束 loss
    手指关节在同一平面内，限制关节之间的叉乘方向
    """
    loss = 0.0
    B = pred_joints.shape[0]
    # 左手
    left_hand_chains = [
        [22, 23, 24],  # thumb
        [25, 26, 27],  # index
        [28, 29, 30],  # middle
        [31, 32, 33],  # ring
        [34, 35, 36],  # pinky
    ]
    # 右手
    right_hand_chains = [
        [37, 38, 39],  # thumb
        [40, 41, 42],  # index
        [43, 44, 45],  # middle
        [46, 47, 48],  # ring
        [49, 50, 51],  # pinky
    ]
    all_chains = left_hand_chains + right_hand_chains

    for chain in all_chains:
        vec1 = pred_joints[:, chain[1]] - pred_joints[:, chain[0]]
        vec2 = pred_joints[:, chain[2]] - pred_joints[:, chain[1]]
        cross = jt.cross(vec1, vec2, dim=-1)  # [B,3]
        # 希望 cross 接近 0 向量 → 平面弯曲
        loss += jt.mean(jt.norm(cross, dim=-1))
    return loss / len(all_chains)

def train(args):
    """
    Main training function
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"Starting training with parameters: {args}")
    
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type,
        output_channels=num_joints*3,
    )

    sampler = SamplerMix(num_samples=args.num_sample, vertex_samples=args.vertices_sample,
                         use_hand_bias=False)

    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    criterion = nn.MSELoss()
    
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
        random_pose=args.random_pose,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
            random_pose=True,
        )
    else:
        val_loader = None
    
    best_loss = 99999999
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices, joints = data['vertices'], data['joints']
            vertices = vertices.permute(0, 2, 1)

            outputs = model(vertices)
            pred_joints = outputs.reshape(outputs.shape[0], num_joints, 3)
            gt_joints = joints.reshape(outputs.shape[0], num_joints, 3)

            # 原始 MSE loss
            loss = criterion(pred_joints, gt_joints)

            # 添加骨长约束
            bone_loss = compute_bone_length_loss(pred_joints, parents)
            # 添加手部方向约束
            hand_dir_loss = compute_hand_direction_loss(pred_joints)

            # 总 loss
            total_loss = loss + 0.1 * bone_loss + 0.1 * hand_dir_loss  # 权重可调整

            optimizer.zero_grad()
            optimizer.backward(total_loss)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {total_loss.item():.4f}")
        
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        log_message(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} "
                   f"Time: {epoch_time:.2f}s LR: {optimizer.lr:.6f}")

        # 验证逻辑保持不变
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                vertices, joints = data['vertices'], data['joints']
                
                joints = joints.reshape(joints.shape[0], -1)
                if vertices.ndim == 3:
                    vertices = vertices.permute(0, 2, 1)
                outputs = model(vertices)
                
                loss = criterion(outputs, joints)
                
                if batch_idx == show_id:
                    exporter = Exporter()
                    exporter._render_skeleton(path=f"tmp_{args.output_dir}/skeleton/epoch_{epoch}_/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp_{args.output_dir}/skeleton/epoch_{epoch}_/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"tmp_{args.output_dir}/skeleton/epoch_{epoch}_/vertices.png", vertices=vertices[0].permute(1, 0).numpy())
                
                val_loss += loss.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")
            
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, f'best_model_{epoch+1}_regular.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
    
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    return model, best_loss

def main():
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    parser.add_argument('--train_data_list', type=str, required=True)
    parser.add_argument('--val_data_list', type=str, default='')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='pct', choices=['pct', 'pct2', 'custom_pct', 'skeleton'])
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'enhanced'])
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--random_pose', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output/skeleton')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=2048)
    parser.add_argument('--vertices_sample', type=int, default=1024)
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()
