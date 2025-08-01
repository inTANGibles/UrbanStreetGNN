"""
完整实验运行脚本
按顺序执行：数据转换 → 模型训练 → 模型评估 → 结果分析
"""

import os
import sys
import time
from datetime import datetime


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"开始执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = os.system(command)
    end_time = time.time()
    
    if result == 0:
        print(f"✅ {description} 完成！耗时: {end_time - start_time:.2f}秒")
    else:
        print(f"❌ {description} 失败！退出码: {result}")
        sys.exit(1)


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    required_packages = [
        'torch',
        'torch_geometric', 
        'networkx',
        'numpy',
        'matplotlib',
        'sklearn',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_data_files():
    """检查数据文件"""
    print("检查数据文件...")
    
    required_files = [
        "ego_graphs_20250731_194528/ego_graphs.pkl",
        "edge_embedding.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - 文件不存在")
    
    if missing_files:
        print(f"\n请确保以下文件存在: {missing_files}")
        return False
    
    return True


def main():
    """主函数"""
    print("🚀 开始边嵌入模型实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查环境和依赖
    print("\n📋 步骤1: 检查环境和依赖")
    if not check_dependencies():
        print("❌ 依赖检查失败，请安装缺失的包")
        return
    
    if not check_data_files():
        print("❌ 数据文件检查失败，请确保文件存在")
        return
    
    # 2. 训练模型
    print("\n🎯 步骤2: 训练边嵌入模型")
    run_command("python train_edge_embedding.py", "模型训练")
    
    # 3. 评估模型
    print("\n📊 步骤3: 评估模型")
    run_command("python evaluate_model.py", "模型评估")
    
    
    # 4. 分析结果
    print("\n📈 步骤5: 分析结果")
    run_command("python analyze_results.py", "结果分析")
    
    
    print(f"\n🎉 实验完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n生成的文件:")
    print("- edge_embedding_model.pt (训练好的模型)")
    print("- edge_embedding_results.pt (推理结果)")
    print("- embeddings_visualization.png (嵌入可视化)")
    print("- structural_scores_analysis.png (结构得分分析)")
    print("- cluster_analysis.png (聚类分析)")
    print("- experiment_report.txt (实验报告)")


def generate_report():
    """生成实验报告"""
    report_content = f"""
边嵌入模型实验报告
==================

实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

实验目标:
- 基于图神经网络的无监督边嵌入学习
- 以街道"边"为基本分析单元
- 生成结构重要性得分
- 识别超级街区边界

模型配置:
- 节点特征维度: 4 (经度、纬度、度数、中心性)
- 边特征维度: 3 (道路类型、宽度、长度)
- 隐藏层维度: 256
- 嵌入维度: 128
- GNN层数: 3
- 卷积类型: GCN
- Dropout率: 0.2

损失函数:
- ContrastiveLearningLoss (对比学习损失)
- 温度参数: 0.1
- 边界距离: 1.0

训练参数:
- 学习率: 1e-3
- 训练轮数: 100
- 优化器: Adam

数据信息:
- 数据源: ego_graphs_20250731_194528/ego_graphs.pkl
- 图数量: 100个ego-graph
- 覆盖城市: 北京、上海、深圳、重庆、成都、西安、香港

输出文件:
- edge_embedding_model.pt: 训练好的模型权重
- edge_embedding_results.pt: 边嵌入和结构得分
- embeddings_visualization.png: t-SNE可视化
- structural_scores_analysis.png: 结构得分分析
- cluster_analysis.png: 聚类分析结果

实验意义:
本研究创新性地将GNN引入街区尺度的道路嵌入建模与更新评估体系，
避免了对人为分类与监督标签的依赖，提出了以边嵌入为核心的城市形态"结构评分机制"，
为中国情境下的城市更新实践提供理论与技术支持。
"""
    
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("✅ 实验报告已生成: experiment_report.txt")


if __name__ == "__main__":
    main() 