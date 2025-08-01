# 边嵌入模型 - 城市街道网络结构评分

基于图神经网络的无监督学习方法，以街道"边"为基本分析单元，捕捉其在整体网络中的结构贡献，为街区尺度的更新提供量化依据。

## 项目概述

本研究提出一种基于图神经网络（GNN）的无监督学习方法，专门用于分析城市街道网络的结构特征。通过边嵌入学习，模型能够：

- 提取街道网络的多维语义特征
- 生成结构重要性得分
- 识别超级街区边界
- 支持城市更新决策

## 文件结构

```
models/
├── edge_embedding.py          # 核心边嵌入模型
├── train_edge_embedding.py    # 模型训练脚本（基于GNN建模策略）
├── evaluate_model.py          # 模型评估脚本
├── analyze_results.py         # 结果分析脚本
├── training_monitor.py        # 训练监控和分析脚本
├── visualize_architecture.py  # 架构可视化脚本
├── run_experiment.py          # 完整实验运行脚本
├── README.md                  # 项目说明文档
└── ego_graphs_20250731_194528/
    ├── ego_graphs.pkl         # 原始数据文件
    └── ego_graph_*.png        # 图可视化文件
```

## 核心模型

### EdgeEmbeddingModel

边嵌入模型基于GNN建模策略设计，核心架构包含：

#### 输入特征
- **节点中心性 + 边属性**：节点坐标、度数、中心性（度中心性、介数中心性、接近中心性）+ 道路类型、宽度、长度

#### 编码层
- **Linear/MLP(2层)**：建立统一维度，将输入特征转换为隐藏表示

#### 消息传播架构
- **第一步：N2N (Node-to-Node)**：使用GCN/GAT进行节点层消息传递
- **第二步：N2E (Node-to-Edge)**：拼接源节点和目标节点表示构建边嵌入
- **架构**：`h_nodes = GCN(x_nodes, edge_index)` → `h_edges = torch.cat([h_nodes[source], h_nodes[target]])`

#### 输出方式
- **直接作为embedding**：后续可用于强化学习等下游任务

#### 损失函数
- **ContrastiveLearningLoss**：无监督场景友好的对比学习损失

#### 优化器
- **Adam + LR 1e-3**：默认即可的优化策略

#### 正则策略
- **Dropout + L2**：防止过拟合的正则化技术

### 损失函数

目前使用 **ContrastiveLearningLoss**（对比学习损失）：

- 让相似的边在嵌入空间中更接近
- 让不相似的边在嵌入空间中更远离
- 基于结构得分和拓扑关系构建正负样本对

## 数据格式

### 输入数据（pkl文件）

```python
{
    'graph': NetworkX图对象,
    'id': 图ID
}

# 节点特征
节点属性: {
    'pos': [经度, 纬度],
    'degree': 度数,
    'centrality': 中心性
}

# 边特征
边属性: {
    'highway': 道路类型,
    'width': 宽度,
    'length': 长度,
    'geometry': 几何信息
}
```

### 输出数据

- **边嵌入向量**：128维的边表示
- **结构重要性得分**：0-1之间的重要性评分

## 使用方法

### 1. 环境准备

安装必要的依赖包：

```bash
pip install torch torch-geometric networkx numpy matplotlib scikit-learn seaborn
```

### 2. 快速开始

运行完整实验（推荐）：

```bash
python run_experiment.py
```

这个脚本会自动执行：
1. 环境检查
2. 模型训练
3. 模型评估
4. 结果分析
5. 生成报告

### 3. 分步执行

#### 步骤1：训练模型

```bash
python train_edge_embedding.py
```

#### 步骤2：评估模型

```bash
python evaluate_model.py
```

#### 步骤3：分析结果

```bash
python analyze_results.py
```

#### 步骤4：架构可视化

```bash
python visualize_architecture.py
```

#### 步骤5：训练监控

```bash
python training_monitor.py
```

## 模型配置

默认配置参数：

```python
config = {
    'node_features': 4,      # 节点特征维度
    'edge_features': 3,      # 边特征维度
    'hidden_dim': 256,       # 隐藏层维度
    'embedding_dim': 128,    # 嵌入维度
    'num_layers': 3,         # GNN层数
    'dropout': 0.2,          # Dropout率
    'conv_type': 'gcn'       # 卷积类型
}
```

训练参数：

```python
num_epochs = 100             # 训练轮数
learning_rate = 1e-3         # 学习率
temperature = 0.1            # 对比学习温度参数
margin = 1.0                 # 对比学习边界距离
```

## 输出文件

实验完成后会生成以下文件：

- `edge_embedding_model.pt`：训练好的模型权重
- `best_edge_embedding_model.pt`：最佳模型权重
- `edge_embedding_results.pt`：边嵌入和结构得分
- `model_config.json`：模型配置
- `training_history.npy`：训练历史
- `embeddings_visualization.png`：t-SNE可视化
- `structural_scores_analysis.png`：结构得分分析
- `cluster_analysis.png`：聚类分析结果
- `message_passing_architecture.png`：消息传播架构图
- `model_architecture.png`：模型架构图
- `embedding_quality_analysis.png`：嵌入质量分析
- `training_report.txt`：训练报告
- `experiment_report.txt`：实验报告

## 结果分析

### 1. 边嵌入可视化

使用t-SNE将128维边嵌入降维到2维进行可视化：
- 按结构得分着色
- 聚类分析（K-means，5个聚类）

### 2. 结构得分分析

- 分布直方图
- 箱线图
- 分位数分析
- 累积分布

### 3. 聚类分析

- 各聚类的结构得分分布
- 聚类大小对比
- 聚类均值对比
- 聚类标准差对比

## 实验数据

实验数据覆盖7个代表性城市：
- 北京、上海、深圳、重庆、成都、西安、香港

每个城市抽取典型地块中的5条街道边段，构建2-hop ego-graph子图，最终形成包含100个边中心子图的无监督训练数据集。

## 技术特点

1. **无监督学习**：不需要人工标注，自动发现结构模式
2. **边中心分析**：以街道边为基本分析单元
3. **多维特征**：融合拓扑、几何、语义特征
4. **可解释性**：生成结构重要性得分
5. **泛化能力**：可应用于不同城市

## 研究意义

本研究创新性地将GNN引入街区尺度的道路嵌入建模与更新评估体系：

- 避免了对人为分类与监督标签的依赖
- 提出了以边嵌入为核心的城市形态"结构评分机制"
- 为中国情境下的城市更新实践提供理论与技术支持

## 注意事项

1. **计算资源**：建议使用GPU加速训练
2. **内存需求**：至少8GB内存
3. **数据格式**：确保pkl文件格式正确
4. **依赖版本**：建议使用最新版本的PyTorch和PyTorch Geometric

## 联系方式

如有问题或建议，请联系项目维护者。

---

*本项目基于图神经网络技术，专注于城市街道网络的结构分析，为城市规划和更新提供量化依据。* 