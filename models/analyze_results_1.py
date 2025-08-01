"""
结果分析脚本 - 针对城市更新目标
可视化边嵌入和结构得分，分析通达性、边界封闭性和功能断裂问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_path):
    """加载结果"""
    results = torch.load(results_path, map_location='cpu')
    edge_embeddings = results['edge_embeddings'].numpy()
    structural_scores = results['structural_scores'].numpy()
    return edge_embeddings, structural_scores


def analyze_accessibility(edge_embeddings, structural_scores, save_path='accessibility_analysis.png'):
    """
    目标1：提升通达性分析
    与空间句法integration值对比，t-SNE空间中检测"边缘异类点"
    """
    print("=== 分析目标1：提升通达性 ===")
    
    # 计算通达潜力分数（基于结构得分）
    accessibility_scores = structural_scores / np.max(structural_scores)
    
    # 检测边缘异类点（使用IQR方法）
    Q1 = np.percentile(accessibility_scores, 25)
    Q3 = np.percentile(accessibility_scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (accessibility_scores < lower_bound) | (accessibility_scores > upper_bound)
    
    # t-SNE降维用于可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(edge_embeddings)
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 通达潜力分数分布
    ax1.hist(accessibility_scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(np.mean(accessibility_scores), color='red', linestyle='--', 
                label=f'均值: {np.mean(accessibility_scores):.3f}')
    ax1.axvline(lower_bound, color='orange', linestyle='--', label=f'异常值下界: {lower_bound:.3f}')
    ax1.axvline(upper_bound, color='orange', linestyle='--', label=f'异常值上界: {upper_bound:.3f}')
    ax1.set_title('通达潜力分数分布')
    ax1.set_xlabel('通达潜力分数')
    ax1.set_ylabel('频次')
    ax1.legend()
    
    # 2. t-SNE空间中的边缘异类点检测
    scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=accessibility_scores, cmap='viridis', alpha=0.6)
    # 标记异常值
    outlier_points = embeddings_2d[outliers]
    ax2.scatter(outlier_points[:, 0], outlier_points[:, 1], 
               c='red', marker='x', s=100, label=f'边缘异类点 ({np.sum(outliers)}个)')
    ax2.set_title('t-SNE空间中的通达性分布\n(红色X为边缘异类点)')
    ax2.set_xlabel('t-SNE维度1')
    ax2.set_ylabel('t-SNE维度2')
    plt.colorbar(scatter, ax=ax2, label='通达潜力分数')
    ax2.legend()
    
    # 3. 高embedding值 vs 低connectivity分析
    # 假设结构得分高的边具有更好的通达性
    high_accessibility = accessibility_scores > np.percentile(accessibility_scores, 75)
    low_accessibility = accessibility_scores < np.percentile(accessibility_scores, 25)
    
    ax3.scatter(embeddings_2d[high_accessibility, 0], embeddings_2d[high_accessibility, 1], 
               c='green', alpha=0.7, label=f'高通达性 ({np.sum(high_accessibility)}个)')
    ax3.scatter(embeddings_2d[low_accessibility, 0], embeddings_2d[low_accessibility, 1], 
               c='red', alpha=0.7, label=f'低通达性 ({np.sum(low_accessibility)}个)')
    ax3.set_title('高vs低通达性边段分布')
    ax3.set_xlabel('t-SNE维度1')
    ax3.set_ylabel('t-SNE维度2')
    ax3.legend()
    
    # 4. 通达性分数统计
    stats_data = {
        '指标': ['样本总数', '高通达性边段', '低通达性边段', '边缘异类点', '平均通达分数'],
        '数值': [len(accessibility_scores), np.sum(high_accessibility), 
                np.sum(low_accessibility), np.sum(outliers), f"{np.mean(accessibility_scores):.3f}"]
    }
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=pd.DataFrame(stats_data).values, 
                     colLabels=stats_data['指标'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('通达性分析统计')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"通达性分析已保存到: {save_path}")
    print(f"检测到 {np.sum(outliers)} 个边缘异类点")
    print(f"高通达性边段: {np.sum(high_accessibility)} 个")
    print(f"低通达性边段: {np.sum(low_accessibility)} 个")
    
    return accessibility_scores, outliers


def analyze_boundary_enclosure(edge_embeddings, structural_scores, save_path='boundary_enclosure_analysis.png'):
    """
    目标2：缓解边界封闭分析
    聚类类别与街区边缘分布重叠度分析，与dead-end分布对比
    """
    print("=== 分析目标2：缓解边界封闭 ===")
    
    # 进行聚类分析
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(edge_embeddings)
    
    # 计算边界感识别指数（基于聚类的分离度）
    silhouette_avg = silhouette_score(edge_embeddings, clusters)
    
    # 计算弱连接度判别（基于结构得分的低值区域）
    weak_connectivity_threshold = np.percentile(structural_scores, 30)
    weak_connections = structural_scores < weak_connectivity_threshold
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 聚类结果可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(edge_embeddings)
    
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=clusters, cmap='tab10', alpha=0.7)
    ax1.set_title(f'边段聚类结果 (Silhouette Score: {silhouette_avg:.3f})')
    ax1.set_xlabel('t-SNE维度1')
    ax1.set_ylabel('t-SNE维度2')
    plt.colorbar(scatter, ax=ax1, label='聚类标签')
    
    # 2. 各聚类的结构得分分布
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        cluster_scores = structural_scores[clusters == i]
        ax2.hist(cluster_scores, alpha=0.6, color=cluster_colors[i], 
                label=f'聚类 {i} ({len(cluster_scores)}个)', bins=20)
    
    ax2.axvline(weak_connectivity_threshold, color='red', linestyle='--', 
                label=f'弱连接阈值: {weak_connectivity_threshold:.3f}')
    ax2.set_title('各聚类的结构得分分布')
    ax2.set_xlabel('结构得分')
    ax2.set_ylabel('频次')
    ax2.legend()
    
    # 3. 弱连接度判别
    ax3.scatter(embeddings_2d[~weak_connections, 0], embeddings_2d[~weak_connections, 1], 
               c='green', alpha=0.6, label=f'强连接 ({np.sum(~weak_connections)}个)')
    ax3.scatter(embeddings_2d[weak_connections, 0], embeddings_2d[weak_connections, 1], 
               c='red', alpha=0.8, s=50, label=f'弱连接 ({np.sum(weak_connections)}个)')
    ax3.set_title('弱连接度判别')
    ax3.set_xlabel('t-SNE维度1')
    ax3.set_ylabel('t-SNE维度2')
    ax3.legend()
    
    # 4. 边界感识别指数分析
    cluster_stats = []
    for i in range(n_clusters):
        cluster_scores = structural_scores[clusters == i]
        cluster_weak = np.sum(weak_connections[clusters == i])
        cluster_stats.append({
            '聚类': i,
            '样本数': len(cluster_scores),
            '平均得分': np.mean(cluster_scores),
            '弱连接数': cluster_weak,
            '弱连接比例': cluster_weak / len(cluster_scores)
        })
    
    # 绘制弱连接比例
    weak_ratios = [stat['弱连接比例'] for stat in cluster_stats]
    bars = ax4.bar(range(n_clusters), weak_ratios, color='orange', alpha=0.7)
    ax4.set_title('各聚类弱连接比例')
    ax4.set_xlabel('聚类标签')
    ax4.set_ylabel('弱连接比例')
    ax4.set_xticks(range(n_clusters))
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"边界封闭分析已保存到: {save_path}")
    print(f"聚类Silhouette Score: {silhouette_avg:.3f}")
    print(f"弱连接边段: {np.sum(weak_connections)} 个 ({np.sum(weak_connections)/len(structural_scores)*100:.1f}%)")
    
    return clusters, weak_connections, silhouette_avg


def analyze_functional_discontinuity(edge_embeddings, structural_scores, save_path='functional_discontinuity_analysis.png'):
    """
    目标3：衔接功能断裂分析
    与land use分区、POI功能标签对比，是否跨越异构区域
    """
    print("=== 分析目标3：衔接功能断裂 ===")
    
    # 计算功能耦合指数（基于嵌入的多样性）
    # 使用嵌入向量的标准差作为多样性指标
    embedding_diversity = np.std(edge_embeddings, axis=1)
    
    # 计算熵（基于结构得分的分布）
    def calculate_entropy(scores, bins=10):
        hist, _ = np.histogram(scores, bins=bins, density=True)
        hist = hist[hist > 0]  # 避免log(0)
        return -np.sum(hist * np.log(hist))
    
    # 计算局部熵（使用滑动窗口）
    window_size = min(50, len(structural_scores) // 10)
    local_entropy = []
    for i in range(len(structural_scores)):
        start = max(0, i - window_size // 2)
        end = min(len(structural_scores), i + window_size // 2)
        local_scores = structural_scores[start:end]
        local_entropy.append(calculate_entropy(local_scores))
    
    local_entropy = np.array(local_entropy)
    
    # 功能耦合指数 = 多样性 * 熵
    functional_coupling_index = embedding_diversity * local_entropy
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 功能耦合指数分布
    ax1.hist(functional_coupling_index, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.axvline(np.mean(functional_coupling_index), color='red', linestyle='--', 
                label=f'均值: {np.mean(functional_coupling_index):.3f}')
    ax1.set_title('功能耦合指数分布')
    ax1.set_xlabel('功能耦合指数')
    ax1.set_ylabel('频次')
    ax1.legend()
    
    # 2. 嵌入多样性 vs 局部熵
    scatter = ax2.scatter(embedding_diversity, local_entropy, 
                         c=functional_coupling_index, cmap='plasma', alpha=0.6)
    ax2.set_title('嵌入多样性 vs 局部熵')
    ax2.set_xlabel('嵌入多样性')
    ax2.set_ylabel('局部熵')
    plt.colorbar(scatter, ax=ax2, label='功能耦合指数')
    
    # 3. t-SNE空间中的功能耦合分布
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(edge_embeddings)
    
    scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=functional_coupling_index, cmap='plasma', alpha=0.6)
    ax3.set_title('t-SNE空间中的功能耦合分布')
    ax3.set_xlabel('t-SNE维度1')
    ax3.set_ylabel('t-SNE维度2')
    plt.colorbar(scatter, ax=ax3, label='功能耦合指数')
    
    # 4. 功能边界识别
    # 识别高功能耦合的边段（可能跨越功能边界）
    high_coupling_threshold = np.percentile(functional_coupling_index, 80)
    high_coupling_edges = functional_coupling_index > high_coupling_threshold
    
    ax4.scatter(embeddings_2d[~high_coupling_edges, 0], embeddings_2d[~high_coupling_edges, 1], 
               c='blue', alpha=0.6, label=f'低功能耦合 ({np.sum(~high_coupling_edges)}个)')
    ax4.scatter(embeddings_2d[high_coupling_edges, 0], embeddings_2d[high_coupling_edges, 1], 
               c='red', alpha=0.8, s=60, label=f'高功能耦合 ({np.sum(high_coupling_edges)}个)')
    ax4.set_title('功能边界识别\n(红色为可能跨越功能边界的边段)')
    ax4.set_xlabel('t-SNE维度1')
    ax4.set_ylabel('t-SNE维度2')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"功能断裂分析已保存到: {save_path}")
    print(f"高功能耦合边段: {np.sum(high_coupling_edges)} 个")
    print(f"平均功能耦合指数: {np.mean(functional_coupling_index):.3f}")
    print(f"嵌入多样性范围: [{np.min(embedding_diversity):.3f}, {np.max(embedding_diversity):.3f}]")
    
    return functional_coupling_index, high_coupling_edges


def generate_comprehensive_report(accessibility_scores, boundary_clusters, functional_coupling_index, 
                                save_path='comprehensive_analysis_report.png'):
    """生成综合分析报告"""
    print("=== 生成综合分析报告 ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 三个目标的综合评分
    # 标准化各个指标到0-1范围
    norm_accessibility = (accessibility_scores - np.min(accessibility_scores)) / (np.max(accessibility_scores) - np.min(accessibility_scores))
    norm_functional = (functional_coupling_index - np.min(functional_coupling_index)) / (np.max(functional_coupling_index) - np.min(functional_coupling_index))
    
    # 计算综合更新优先级（高通达性 + 低边界封闭 + 高功能耦合）
    update_priority = norm_accessibility + (1 - norm_functional)  # 功能耦合越高，边界越开放
    
    ax1.hist(update_priority, bins=50, alpha=0.7, color='gold', edgecolor='black')
    ax1.axvline(np.mean(update_priority), color='red', linestyle='--', 
                label=f'平均优先级: {np.mean(update_priority):.3f}')
    ax1.set_title('城市更新优先级分布')
    ax1.set_xlabel('更新优先级')
    ax1.set_ylabel('频次')
    ax1.legend()
    
    # 2. 三个指标的相关性分析
    correlation_matrix = np.corrcoef([norm_accessibility, norm_functional, update_priority])
    im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['通达性', '功能耦合', '更新优先级'])
    ax2.set_yticklabels(['通达性', '功能耦合', '更新优先级'])
    ax2.set_title('指标相关性矩阵')
    
    # 添加相关系数标签
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # 3. 更新策略建议
    high_priority_threshold = np.percentile(update_priority, 80)
    high_priority_edges = update_priority > high_priority_threshold
    
    ax3.scatter(norm_accessibility, norm_functional, 
               c=update_priority, cmap='viridis', alpha=0.6)
    ax3.scatter(norm_accessibility[high_priority_edges], norm_functional[high_priority_edges], 
               c='red', marker='*', s=100, label=f'高优先级 ({np.sum(high_priority_edges)}个)')
    ax3.set_title('更新策略建议')
    ax3.set_xlabel('标准化通达性')
    ax3.set_ylabel('标准化功能耦合')
    ax3.legend()
    
    # 4. 统计摘要
    stats_data = {
        '指标': ['总边段数', '高通达性', '弱连接', '高功能耦合', '高更新优先级'],
        '数量': [len(update_priority), 
                np.sum(norm_accessibility > 0.7),
                np.sum(norm_accessibility < 0.3),
                np.sum(norm_functional > 0.7),
                np.sum(high_priority_edges)],
        '比例(%)': [100,
                   np.sum(norm_accessibility > 0.7) / len(update_priority) * 100,
                   np.sum(norm_accessibility < 0.3) / len(update_priority) * 100,
                   np.sum(norm_functional > 0.7) / len(update_priority) * 100,
                   np.sum(high_priority_edges) / len(update_priority) * 100]
    }
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=pd.DataFrame(stats_data).values, 
                     colLabels=stats_data['指标'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('综合分析统计摘要')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"综合分析报告已保存到: {save_path}")
    print(f"建议优先更新的边段: {np.sum(high_priority_edges)} 个")


def main():
    """主函数"""
    print("开始针对城市更新目标的结果分析...")
    
    # 加载结果
    results_path = "edge_embedding_results.pt"
    edge_embeddings, structural_scores = load_results(results_path)
    
    print(f"加载了 {len(edge_embeddings)} 个边的嵌入和得分")
    
    # 目标1：提升通达性分析
    accessibility_scores, outliers = analyze_accessibility(edge_embeddings, structural_scores)
    
    # 目标2：缓解边界封闭分析
    boundary_clusters, weak_connections, silhouette_avg = analyze_boundary_enclosure(edge_embeddings, structural_scores)
    
    # 目标3：衔接功能断裂分析
    functional_coupling_index, high_coupling_edges = analyze_functional_discontinuity(edge_embeddings, structural_scores)
    
    # 生成综合分析报告
    generate_comprehensive_report(accessibility_scores, boundary_clusters, functional_coupling_index)
    
    print("所有分析完成！")


if __name__ == "__main__":
    main() 