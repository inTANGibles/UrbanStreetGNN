"""
模型评估脚本
加载训练好的边嵌入模型并进行推理
"""

import torch
from torch_geometric.data import DataLoader
from edge_embedding import EdgeEmbeddingModel
from train_edge_embedding import convert_ego_graphs_to_pytorch
import numpy as np


def evaluate_model(model_path, pkl_path, device='cpu'):
    """评估模型"""
    # 加载模型
    config = {
        'node_features': 2,
        'edge_features': 3,
        'hidden_dim': 256,
        'embedding_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'conv_type': 'gcn'
    }
    
    model = EdgeEmbeddingModel(**config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 加载数据
    graphs = convert_ego_graphs_to_pytorch(pkl_path)
    
    # 推理
    all_edge_embeddings = []
    all_structural_scores = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            edge_embeddings, structural_scores = model(graph)
            
            all_edge_embeddings.append(edge_embeddings.cpu())
            all_structural_scores.append(structural_scores.cpu())
    
    # 合并结果
    all_edge_embeddings = torch.cat(all_edge_embeddings, dim=0)
    all_structural_scores = torch.cat(all_structural_scores, dim=0)
    
    print(f"评估完成！")
    print(f"总边数: {all_edge_embeddings.shape[0]}")
    print(f"嵌入维度: {all_edge_embeddings.shape[1]}")
    print(f"结构得分范围: [{all_structural_scores.min():.4f}, {all_structural_scores.max():.4f}]")
    print(f"结构得分均值: {all_structural_scores.mean():.4f}")
    print(f"结构得分标准差: {all_structural_scores.std():.4f}")
    
    # 保存结果
    torch.save({
        'edge_embeddings': all_edge_embeddings,
        'structural_scores': all_structural_scores
    }, 'edge_embedding_results.pt')
    
    print("结果已保存到: edge_embedding_results.pt")
    
    return all_edge_embeddings, all_structural_scores


def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model_path = "edge_embedding_model.pt"
    pkl_path = "ego_graphs_20250731_194528/ego_graphs.pkl"
    
    edge_embeddings, structural_scores = evaluate_model(model_path, pkl_path, device)


if __name__ == "__main__":
    main() 