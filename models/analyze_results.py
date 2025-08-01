"""
简化版结果分析脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_path):
    """加载结果"""
    results = torch.load(results_path, map_location='cpu')
    edge_embeddings = results['edge_embeddings'].numpy()
    structural_scores = results['structural_scores'].numpy()
    return edge_embeddings, structural_scores

def plot_training_curve():
    """绘制训练曲线"""
    try:
        # 加载训练历史数据
        training_history = np.load('training_history.npy')
        
        plt.figure(figsize=(10, 6))
        plt.plot(training_history, 'b-', linewidth=2)
        plt.title('训练损失曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练轮次: {len(training_history)}")
        print(f"最终损失: {training_history[-1]:.6f}")
        print(f"最佳损失: {training_history.min():.6f}")
        
    except FileNotFoundError:
        print("未找到训练历史文件 training_history.npy")

def simple_analysis(edge_embeddings, structural_scores):
    """简化分析"""
    print("=== 简化分析 ===")
    
    # 基础统计
    print(f"嵌入形状: {edge_embeddings.shape}")
    print(f"得分形状: {structural_scores.shape}")
    print(f"嵌入均值: {edge_embeddings.mean():.6f}")
    print(f"嵌入标准差: {edge_embeddings.std():.6f}")
    print(f"得分均值: {structural_scores.mean():.6f}")
    print(f"得分标准差: {structural_scores.std():.6f}")
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 结构得分分布
    ax1.hist(structural_scores, bins=30, alpha=0.7, color='skyblue')
    ax1.set_title('结构得分分布')
    ax1.set_xlabel('结构得分')
    ax1.set_ylabel('频次')
    
    # 2. t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(edge_embeddings)
    
    scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=structural_scores, cmap='viridis', alpha=0.6)
    ax2.set_title('t-SNE空间分布')
    plt.colorbar(scatter, ax=ax2, label='结构得分')
    
    # 3. 聚类分析
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(edge_embeddings)
    
    scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=clusters, cmap='tab10', alpha=0.7)
    ax3.set_title('聚类结果')
    plt.colorbar(scatter, ax=ax3, label='聚类标签')
    
    # 4. 高分vs低分对比
    high_score = structural_scores > np.percentile(structural_scores, 75)
    low_score = structural_scores < np.percentile(structural_scores, 25)
    
    ax4.scatter(embeddings_2d[~high_score & ~low_score, 0], 
               embeddings_2d[~high_score & ~low_score, 1], 
               c='gray', alpha=0.5, label='中等得分')
    ax4.scatter(embeddings_2d[high_score, 0], embeddings_2d[high_score, 1], 
               c='red', alpha=0.7, label=f'高分 ({np.sum(high_score)}个)')
    ax4.scatter(embeddings_2d[low_score, 0], embeddings_2d[low_score, 1], 
               c='blue', alpha=0.7, label=f'低分 ({np.sum(low_score)}个)')
    ax4.set_title('高分vs低分边段')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('simple_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"高分边段: {np.sum(high_score)} 个")
    print(f"低分边段: {np.sum(low_score)} 个")
    print("分析完成！")

def main():
    """主函数"""
    results_path = "edge_embedding_results.pt"
    edge_embeddings, structural_scores = load_results(results_path)
    simple_analysis(edge_embeddings, structural_scores)
    
    # 添加训练曲线可视化
    plot_training_curve()

if __name__ == "__main__":
    main()