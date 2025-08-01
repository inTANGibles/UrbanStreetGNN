"""
边嵌入模型训练脚本
基于GNN建模策略的完整实现
"""

import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
from edge_embedding import EdgeEmbeddingModel, ContrastiveLearningLoss, GraphAutoEncoderLoss
import os
from datetime import datetime


def encode_highway_type(highway_type):
    """编码道路类型"""
    type_mapping = {
        'footway': 0, 'secondary': 1, 'tertiary': 2, 'primary': 3,
    }
    return type_mapping.get(highway_type, 0)


def extract_node_centrality(nx_graph):
    """提取节点中心性特征"""
    # 计算各种中心性指标
    degree_centrality = nx.degree_centrality(nx_graph)
    betweenness_centrality = nx.betweenness_centrality(nx_graph)
    closeness_centrality = nx.closeness_centrality(nx_graph)
    
    centrality_features = {}
    for node in nx_graph.nodes():
        centrality_features[node] = [
            degree_centrality.get(node, 0),
            betweenness_centrality.get(node, 0),
            closeness_centrality.get(node, 0)
        ]
    
    return centrality_features


def convert_ego_graphs_to_pytorch(pkl_path):
    """将ego-graphs转换为PyTorch Geometric格式"""
    print("加载pkl文件...")
    with open(pkl_path, 'rb') as f:
        ego_graphs = pickle.load(f)
    
    print("转换图数据...")
    pytorch_graphs = []
    
    for i, graph_data in enumerate(ego_graphs):
        nx_graph = graph_data['graph']
        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        
        # 提取节点中心性特征
        centrality_features = extract_node_centrality(nx_graph)
        
        # 节点特征：度数 + 中心性
        node_features = []
        for node in nx_graph.nodes():
            # 基础特征
            degree = nx_graph.nodes[node]['degree']
            centrality = nx_graph.nodes[node]['centrality']
            
            # 组合特征
            features = [
                degree,  # 度数
                centrality,  # 原始中心性
            ]
            node_features.append(features)
                
        # 边特征和索引
        edge_features = []
        edge_indices = []
        for u, v, data in nx_graph.edges(data=True):
            # 安全地提取边特征，处理可能的字符串类型
            try:
                highway_encoded = encode_highway_type(data.get('highway', 'unknown'))
                width = float(data.get('width', 0.0))
                length = float(data.get('length', 0.0))
                
                features = [
                    highway_encoded,
                    width,
                    length
                ]
                edge_features.append(features)
                edge_indices.append([node_mapping[u], node_mapping[v]])
            except (ValueError, TypeError) as e:
                print(f"警告：跳过边 ({u}, {v})，数据转换错误: {e}")
                print(f"边数据: {data}")
                continue
        
        # 创建Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            graph_id=graph_data.get('id', i)
        )
        pytorch_graphs.append(data)
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(ego_graphs)} 个图")
    
    print(f"转换完成！共处理 {len(pytorch_graphs)} 个图")
    return pytorch_graphs


def create_model_config():
    """根据建模策略创建模型配置"""
    config = {
        # 输入特征：节点中心性 + 边属性
        'node_features': 2,      # 度数、中心性 
        'edge_features': 3,      # 道路类型、宽度、长度
        
        # 编码层：Linear/MLP(2层)
        'hidden_dim': 256,       # 隐藏层维度
        'embedding_dim': 128,    # 嵌入维度
        
        # GNN层：GCNConv + ReLU + Dropout
        'num_layers': 3,         # GNN层数
        'dropout': 0.2,          # Dropout率
        'conv_type': 'gcn',      # 卷积类型
        
        # # 消息传播方式：明确分离N2N和N2E
        # 'message_passing': {
        #     'n2n_layers': 3,     # Node-to-Node层数
        #     'n2e_method': 'concatenate',  # Node-to-Edge聚合方法
        #     'conv_type': 'gcn'   # 卷积类型
        # },
        
        # # 输出方式：直接作为embedding
        # 'output_method': 'direct_embedding'
    }
    return config


def train_model(graphs, config, num_epochs=500, learning_rate=1e-3, device='cpu'):
    """训练模型"""
    print("创建模型...")
    model = EdgeEmbeddingModel(**config)
    model = model.to(device)
    
    # 损失函数：图自编码器损失
    gae_loss = GraphAutoEncoderLoss(alpha=0.7, beta=0.3)
    
    # Optimizer：Adam + LR 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2正则化
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 数据加载器
    dataloader = DataLoader(graphs, batch_size=1, shuffle=True)
    
    print("开始训练...")
    model.train()
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            # 前向传播
            center_edge_embedding, structural_score = model(batch)
            
            # 计算图自编码器损失
            loss = gae_loss(center_edge_embedding, batch.edge_index, structural_score)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        training_history.append(avg_loss)
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_edge_embedding_model.pt')
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    
    # 保存训练历史
    np.save('training_history.npy', np.array(training_history))
    
    return model, training_history


def evaluate_model_performance(model, graphs, device='cpu'):
    """评估模型性能"""
    model.eval()
    
    all_center_edge_embeddings = []
    all_structural_scores = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            center_edge_embedding, structural_score = model(graph)
            
            all_center_edge_embeddings.append(center_edge_embedding.cpu())
            all_structural_scores.append(structural_score.cpu())
    
    # 合并结果 - 现在每个子图输出一个中心边的embedding
    all_center_edge_embeddings = torch.cat(all_center_edge_embeddings, dim=0)  # [num_graphs, embedding_dim]
    all_structural_scores = torch.cat(all_structural_scores, dim=0)  # [num_graphs]
    
    # 计算统计信息
    stats = {
        'total_graphs': len(all_center_edge_embeddings),
        'embedding_dim': all_center_edge_embeddings.shape[1],
        'score_range': (all_structural_scores.min().item(), all_structural_scores.max().item()),
        'score_mean': all_structural_scores.mean().item(),
        'score_std': all_structural_scores.std().item(),
        'score_median': all_structural_scores.median().item()
    }
    
    print("=== 模型性能评估 ===")
    print(f"总子图数: {stats['total_graphs']}")
    print(f"嵌入维度: {stats['embedding_dim']}")
    print(f"输出形状: {all_center_edge_embeddings.shape}")  # 应该是 [100, 128]
    print(f"结构得分范围: [{stats['score_range'][0]:.4f}, {stats['score_range'][1]:.4f}]")
    print(f"结构得分均值: {stats['score_mean']:.4f}")
    print(f"结构得分标准差: {stats['score_std']:.4f}")
    print(f"结构得分中位数: {stats['score_median']:.4f}")
    
    return all_center_edge_embeddings, all_structural_scores, stats


def save_results(model, center_edge_embeddings, structural_scores, stats, config):
    """保存结果"""
    # 保存模型
    torch.save(model.state_dict(), 'edge_embedding_model.pt')
    
    # 保存结果
    torch.save({
        'center_edge_embeddings': center_edge_embeddings,
        'structural_scores': structural_scores,
        'stats': stats,
        'config': config
    }, 'edge_embedding_results.pt')
    
    # 保存配置
    import json
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("结果已保存！")
    print(f"中心边嵌入形状: {center_edge_embeddings.shape}")  # 应该是 [100, 128]


def main():
    """主函数"""
    print("🚀 开始边嵌入模型训练")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数据路径
    pkl_path = "ego_graphs_20250731_194528/ego_graphs.pkl"
    
    # 检查数据文件
    if not os.path.exists(pkl_path):
        print(f"❌ 数据文件不存在: {pkl_path}")
        return
    
    # 1. 数据转换
    print("\n📊 步骤1: 数据转换")
    graphs = convert_ego_graphs_to_pytorch(pkl_path)
    
    # 2. 创建模型配置
    print("\n⚙️ 步骤2: 创建模型配置")
    config = create_model_config()
    print("模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 3. 训练模型
    print("\n🎯 步骤3: 训练模型")
    model, training_history = train_model(
        graphs=graphs,
        config=config,
        num_epochs=500,
        learning_rate=1e-3,
        device=device
    )
    
    # 4. 评估模型
    print("\n📈 步骤4: 评估模型")
    center_edge_embeddings, structural_scores, stats = evaluate_model_performance(
        model, graphs, device
    )
    
    # 5. 保存结果
    print("\n💾 步骤5: 保存结果")
    save_results(model, center_edge_embeddings, structural_scores, stats, config)
    
    print(f"\n🎉 训练完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n生成的文件:")
    print("- edge_embedding_model.pt (最终模型)")
    print("- best_edge_embedding_model.pt (最佳模型)")
    print("- edge_embedding_results.pt (推理结果)")
    print("- model_config.json (模型配置)")
    print("- training_history.npy (训练历史)")


if __name__ == "__main__":
    main() 