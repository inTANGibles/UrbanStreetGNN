"""
边嵌入模型 - 城市街道网络结构评分核心模型
基于图神经网络的无监督学习方法，以街道"边"为基本分析单元
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional


class EdgeEmbeddingModel(nn.Module):
    """
    边嵌入模型 - 核心架构
    
    该模型将街道网络建模为空间图结构，提取多维语义特征，
    构建以边为中心的局部子图，进行嵌入学习并生成结构得分。
    """
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 conv_type: str = "gcn"):
        """
        初始化边嵌入模型
        
        Args:
            node_features: 节点特征维度
            edge_features: 边特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 嵌入维度
            num_layers: GNN层数
            dropout: Dropout率
            conv_type: 卷积类型 ("gcn", "gat", "sage")
        """
        super(EdgeEmbeddingModel, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            self.conv_layers.append(conv)
        
        # 边嵌入投影层
        self.edge_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 源节点 + 目标节点
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 结构评分层
        self.structural_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的结构重要性得分
        )
        
        # 边特征融合层
        self.edge_fusion = nn.Sequential(
            nn.Linear(embedding_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 为每个子图输出中心边的embedding
        
        Args:
            data: PyTorch Geometric Data对象，包含节点特征、边特征、边索引等
            
        Returns:
            center_edge_embedding: 中心边的嵌入向量 [1, embedding_dim]
            structural_score: 中心边的结构重要性得分 [1]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 第一步：节点特征编码
        node_embeddings = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # 第二步：GNN节点层消息传递 (N2N - Node-to-Node)
        h_nodes = node_embeddings
        for conv in self.conv_layers:
            h_nodes = conv(h_nodes, edge_index)  # GCN/GAT消息传递
            h_nodes = F.relu(h_nodes)
            h_nodes = F.dropout(h_nodes, p=self.dropout, training=self.training)
        # h_nodes.shape = [num_nodes, hidden_dim]
        
        # 第三步：N2E聚合（构建所有边的embedding）
        row, col = edge_index  # 获取边的源节点和目标节点索引
        h_edges = torch.cat([
            h_nodes[row],  # 边的source节点表示
            h_nodes[col]   # 边的target节点表示
        ], dim=1)  # h_edges.shape = [num_edges, 2*hidden_dim]
        
        # 第四步：边嵌入投影
        edge_embeddings = self.edge_projection(h_edges)  # [num_edges, embedding_dim]
        
        # 第五步：融合原始边特征
        edge_embeddings = torch.cat([edge_embeddings, edge_attr], dim=1)
        edge_embeddings = self.edge_fusion(edge_embeddings)  # [num_edges, embedding_dim]
        
        # 第六步：识别中心边（假设中心边是第一个边，或者通过某种方式识别）
        # 这里我们假设中心边是第一个边，您可以根据实际需求修改这个逻辑
        center_edge_embedding = edge_embeddings[0:1]  # [1, embedding_dim]
        
        # 第七步：计算中心边的结构重要性得分
        structural_score = self.structural_scorer(center_edge_embedding).squeeze(-1)  # [1]
        
        return center_edge_embedding, structural_score
    
    def get_edge_embeddings(self, data: Data) -> torch.Tensor:
        """获取边嵌入（用于下游任务）"""
        with torch.no_grad():
            edge_embeddings, _ = self.forward(data)
        return edge_embeddings
    
    def get_structural_scores(self, data: Data) -> torch.Tensor:
        """获取结构重要性得分"""
        with torch.no_grad():
            _, structural_scores = self.forward(data)
        return structural_scores


class ContrastiveLearningLoss(nn.Module):
    """
    对比学习损失函数 - 用于无监督训练
    
    通过对比学习让相似的边在嵌入空间中更接近，
    不相似的边在嵌入空间中更远离。
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        """
        初始化对比学习损失
        
        Args:
            temperature: 温度参数，控制相似度计算的敏感度
            margin: 负样本对的边界距离
        """
        super(ContrastiveLearningLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, edge_embeddings: torch.Tensor, 
                edge_index: torch.Tensor,
                structural_scores: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            edge_embeddings: 边嵌入向量 [num_edges, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            structural_scores: 结构重要性得分 [num_edges]
            
        Returns:
            loss: 对比学习损失
        """
        # 计算边嵌入之间的相似度矩阵
        similarity_matrix = torch.mm(edge_embeddings, edge_embeddings.t()) / self.temperature
        
        # 基于结构得分构建正样本对（结构得分相似的边）
        score_diff = torch.abs(structural_scores.unsqueeze(1) - structural_scores.unsqueeze(0))
        positive_mask = (score_diff < 0.1).float()  # 结构得分差异小于0.1的边对为正样本
        
        # 基于拓扑关系构建正样本对（共享节点的边）
        row, col = edge_index
        shared_node_mask = torch.zeros_like(similarity_matrix)
        for i in range(edge_index.size(1)):
            for j in range(edge_index.size(1)):
                if i != j and (row[i] == row[j] or row[i] == col[j] or 
                              col[i] == row[j] or col[i] == col[j]):
                    shared_node_mask[i, j] = 1.0
        
        # 合并正样本掩码
        positive_mask = torch.max(positive_mask, shared_node_mask)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = (positive_mask * similarity_matrix).sum(dim=1)
        negative_sim = (exp_sim * (1 - positive_mask)).sum(dim=1)
        
        # InfoNCE损失
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        
        return loss.mean()


class GraphAutoEncoderLoss(nn.Module):
    """
    图自编码器损失函数 - 用于无监督训练
    
    通过重建图结构来学习边嵌入，包括：
    1. 链接预测损失（重建边）
    2. 结构一致性损失（保持图结构特征）
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        """
        初始化图自编码器损失
        
        Args:
            alpha: 链接预测损失权重
            beta: 结构一致性损失权重
        """
        super(GraphAutoEncoderLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, center_edge_embedding: torch.Tensor, 
                edge_index: torch.Tensor,
                structural_score: torch.Tensor) -> torch.Tensor:
        """
        计算图自编码器损失 - 针对中心边
        
        Args:
            center_edge_embedding: 中心边嵌入向量 [1, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            structural_score: 中心边结构重要性得分 [1]
            
        Returns:
            loss: 图自编码器损失
        """
        # 1. 结构一致性损失
        # 确保结构得分与嵌入向量的模长相关
        embedding_magnitude = torch.norm(center_edge_embedding, dim=1)
        structural_consistency_loss = F.mse_loss(
            structural_score, embedding_magnitude
        )
        
        # 2. 嵌入正则化损失（防止过拟合）
        embedding_regularization_loss = torch.norm(center_edge_embedding, dim=1).mean()
        
        # 3. 总损失
        total_loss = self.alpha * structural_consistency_loss + self.beta * embedding_regularization_loss
        
        return total_loss


class StructuralConsistencyLoss(nn.Module):
    """
    结构一致性损失 - 确保嵌入结果符合空间句法等传统指标趋势
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化结构一致性损失
        
        Args:
            alpha: 权重参数
        """
        super(StructuralConsistencyLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, structural_scores: torch.Tensor,
                centrality_scores: torch.Tensor,
                connectivity_scores: torch.Tensor) -> torch.Tensor:
        """
        计算结构一致性损失
        
        Args:
            structural_scores: 模型预测的结构得分
            centrality_scores: 中心性指标得分
            connectivity_scores: 连通性指标得分
            
        Returns:
            loss: 结构一致性损失
        """
        # 确保结构得分与中心性指标正相关
        centrality_loss = F.mse_loss(structural_scores, centrality_scores)
        
        # 确保结构得分与连通性指标正相关
        connectivity_loss = F.mse_loss(structural_scores, connectivity_scores)
        
        # 总损失
        total_loss = centrality_loss + self.alpha * connectivity_loss
        
        return total_loss


class SuperblockBoundaryLoss(nn.Module):
    """
    超级街区边界损失 - 鼓励模型识别出与超级街区边界高度一致的边缘结构
    """
    
    def __init__(self, boundary_weight: float = 1.0):
        """
        初始化超级街区边界损失
        
        Args:
            boundary_weight: 边界权重
        """
        super(SuperblockBoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, structural_scores: torch.Tensor,
                boundary_labels: torch.Tensor) -> torch.Tensor:
        """
        计算超级街区边界损失
        
        Args:
            structural_scores: 结构重要性得分
            boundary_labels: 边界标签（1表示边界，0表示内部）
            
        Returns:
            loss: 边界损失
        """
        # 边界边应该有更高的结构得分
        boundary_loss = F.binary_cross_entropy_with_logits(
            structural_scores, boundary_labels.float()
        )
        
        return self.boundary_weight * boundary_loss


def create_edge_embedding_model(config: Dict) -> EdgeEmbeddingModel:
    """
    根据配置创建边嵌入模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: 边嵌入模型实例
    """
    model = EdgeEmbeddingModel(
        node_features=config.get('node_features', 64),
        edge_features=config.get('edge_features', 32),
        hidden_dim=config.get('hidden_dim', 256),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        conv_type=config.get('conv_type', 'gcn')
    )
    
    return model


def get_model_summary(model: EdgeEmbeddingModel) -> str:
    """
    获取模型结构摘要
    
    Args:
        model: 边嵌入模型
        
    Returns:
        summary: 模型结构摘要字符串
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
    边嵌入模型结构摘要:
    ====================
    节点特征维度: {model.node_features}
    边特征维度: {model.edge_features}
    隐藏层维度: {model.hidden_dim}
    嵌入维度: {model.embedding_dim}
    GNN层数: {model.num_layers}
    卷积类型: {model.conv_type}
    Dropout率: {model.dropout}
    
    参数统计:
    - 总参数数量: {total_params:,}
    - 可训练参数: {trainable_params:,}
    """
    
    return summary 