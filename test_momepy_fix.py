import momepy
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class MomepyAutoFixTester:
    def __init__(self, shp_file):
        self.shp_file = shp_file
        self.original_streets = None
        self.fixed_streets = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.original_streets = gpd.read_file(self.shp_file)
        print(f"加载了 {len(self.original_streets)} 条道路")
        
        # 数据预处理
        print("正在进行数据预处理...")
        
        # 确保几何类型是LineString
        self.original_streets = self.original_streets[self.original_streets.geometry.geom_type == 'LineString']
        
        # 确保坐标系是投影坐标系（momepy需要）
        if self.original_streets.crs.is_geographic:
            print("  转换坐标系为投影坐标系...")
            # 使用UTM投影（根据数据位置选择合适的UTM区域）
            self.original_streets = self.original_streets.to_crs('EPSG:32650')  # UTM Zone 50N
        
        # 重置索引
        self.original_streets = self.original_streets.reset_index(drop=True)
        
        print(f"预处理后：{len(self.original_streets)} 条道路")
        print(f"坐标系：{self.original_streets.crs}")
        
    def auto_fix_with_momepy(self):
        """使用momepy进行自动修复"""
        print("=" * 50)
        print("使用momepy进行路网修复...")
        print("=" * 50)
        
        # 复制原始数据
        self.fixed_streets = self.original_streets.copy()
        
        original_count = len(self.fixed_streets)
        print(f"原始道路数量：{original_count}")
        
        # 存储每个步骤的结果用于可视化
        self.step_results = {
            'original': self.original_streets.copy(),
            'step1_remove_isolated_nodes': None,
            'step2_extend_roads': None,
            'step3_remove_false_nodes': None
        }
        
        # 检查momepy函数是否可用
        print("检查momepy函数可用性...")
        try:
            test_result = momepy.remove_false_nodes(self.original_streets.head(10))
            print("  ✓ remove_false_nodes 可用")
        except Exception as e:
            print(f"  ✗ remove_false_nodes 不可用：{e}")
        

        
        # 使用tqdm创建进度条
        with tqdm(total=3, desc="momepy修复进度", unit="步骤") as pbar:
            
            # 1. 移除孤立的节点
            pbar.set_description("步骤1/3: 移除孤立节点")
            try:
                self.fixed_streets = self._remove_isolated_nodes(self.fixed_streets)
                self.step_results['step1_remove_isolated_nodes'] = self.fixed_streets.copy()
                print(f"  移除孤立节点后：{len(self.fixed_streets)} 条道路")
            except Exception as e:
                print(f"  移除孤立节点时出错：{e}")
            pbar.update(1)
            
            # 2. 扩展道路（连接相近的节点）
            pbar.set_description("步骤2/3: 扩展道路")
            try:
                self.fixed_streets = self._extend_roads(self.fixed_streets, tolerance=10)
                self.step_results['step2_extend_roads'] = self.fixed_streets.copy()
                print(f"  扩展道路后：{len(self.fixed_streets)} 条道路")
            except Exception as e:
                print(f"  扩展道路时出错：{e}")
            pbar.update(1)
            
            # 3. 移除假节点（度为2的节点）
            pbar.set_description("步骤3/3: 移除假节点")
            try:
                self.fixed_streets = momepy.remove_false_nodes(self.fixed_streets)
                self.step_results['step3_remove_false_nodes'] = self.fixed_streets.copy()
                print(f"  移除假节点后：{len(self.fixed_streets)} 条道路")
            except Exception as e:
                print(f"  移除假节点时出错：{e}")
            pbar.update(1)
        
        final_count = len(self.fixed_streets)
        
        print("=" * 50)
        print("momepy路网修复完成！")
        print(f"原始道路数量：{original_count}")
        print(f"修复后道路数量：{final_count}")
        print(f"减少了 {original_count - final_count} 条道路")
        print(f"减少率：{(original_count - final_count) / original_count * 100:.2f}%")
        print("=" * 50)
    
    def _remove_isolated_nodes(self, streets_gdf):
        """移除孤立的节点（不连接任何边的节点）"""
        from shapely.geometry import Point, LineString
        import networkx as nx
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加所有边
        for idx, row in streets_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                start_node = coords[0]
                end_node = coords[-1]
                G.add_edge(start_node, end_node, idx=idx)
        
        # 找到孤立的节点（度为0的节点）
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
        
        if isolated_nodes:
            print(f"    发现 {len(isolated_nodes)} 个孤立节点")
            # 移除包含孤立节点的边
            edges_to_remove = []
            for node in isolated_nodes:
                for neighbor in G.neighbors(node):
                    edge_data = G.get_edge_data(node, neighbor)
                    edges_to_remove.append(edge_data['idx'])
            
            if edges_to_remove:
                streets_gdf = streets_gdf.drop(edges_to_remove).reset_index(drop=True)
                print(f"    删除了 {len(edges_to_remove)} 条包含孤立节点的边")
        
        return streets_gdf
    
    def _extend_roads(self, streets_gdf, tolerance=10):
        """扩展道路：连接相近的节点（优化版本）"""
        from shapely.geometry import Point, LineString
        import numpy as np
        from scipy.spatial import cKDTree
        
        # 收集所有节点
        nodes = set()
        for geom in streets_gdf.geometry:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                nodes.add(coords[0])  # 起点
                nodes.add(coords[-1])  # 终点
        
        nodes = list(nodes)
        if len(nodes) < 2:
            return streets_gdf
        
        # 创建KD树用于快速最近邻搜索
        node_array = np.array(nodes)
        tree = cKDTree(node_array)
        
        # 查找所有距离小于容差的节点对
        pairs = tree.query_pairs(tolerance)
        
        # 创建现有边的集合，用于快速检查
        existing_edges = set()
        for geom in streets_gdf.geometry:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                start_node = coords[0]
                end_node = coords[-1]
                # 存储边的两个方向
                existing_edges.add((start_node, end_node))
                existing_edges.add((end_node, start_node))
        
        new_edges = []
        
        # 检查找到的节点对
        for i, j in pairs:
            node1 = tuple(node_array[i])
            node2 = tuple(node_array[j])
            
            # 检查是否已存在连接这两个节点的边
            if (node1, node2) not in existing_edges:
                # 创建新的边
                new_line = LineString([node1, node2])
                new_edges.append(new_line)
        
        if new_edges:
            print(f"    添加了 {len(new_edges)} 条新边")
            # 创建新的GeoDataFrame并合并
            new_gdf = gpd.GeoDataFrame(geometry=new_edges, crs=streets_gdf.crs)
            streets_gdf = gpd.concat([streets_gdf, new_gdf], ignore_index=True)
        
        return streets_gdf
    
    def analyze_network(self):
        """分析路网特征"""
        print("正在分析路网特征...")
        
        # 计算基本统计信息
        stats = {
            'original_length': self.original_streets.geometry.length.sum(),
            'fixed_length': self.fixed_streets.geometry.length.sum(),
            'original_count': len(self.original_streets),
            'fixed_count': len(self.fixed_streets)
        }
        
        # 计算路网密度（如果有边界数据）
        try:
            # 使用边界框作为区域
            bounds = self.original_streets.total_bounds
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            density_original = stats['original_length'] / area
            density_fixed = stats['fixed_length'] / area
            
            stats['density_original'] = density_original
            stats['density_fixed'] = density_fixed
        except:
            pass
        
        return stats
    
    def visualize_comparison(self, output_dir="momepy_fix_results"):
        """可视化对比"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("正在创建可视化对比...")
        
        # 创建每个步骤的可视化
        self._visualize_each_step(output_dir)
        
        # 创建最终对比图
        self._visualize_final_comparison(output_dir)
        
        # 保存统计信息
        self._save_statistics(output_dir)
    
    def _visualize_each_step(self, output_dir):
        """可视化每个步骤的结果"""
        print("正在创建每个步骤的可视化...")
        
        # 定义步骤名称和颜色
        steps = [
            ('original', '原始路网', 'blue'),
            ('step1_remove_isolated_nodes', '步骤1: 移除孤立节点', 'green'),
            ('step2_extend_roads', '步骤2: 扩展道路', 'orange'),
            ('step3_remove_false_nodes', '步骤3: 移除假节点', 'red')
        ]
        
        # 为每个步骤创建单独的可视化
        for step_key, step_name, color in steps:
            print(f"检查步骤：{step_key} - {step_name}")
            
            # 检查步骤是否存在且有数据
            if step_key not in self.step_results:
                print(f"  ✗ 步骤不存在")
                continue
                
            if self.step_results[step_key] is None:
                print(f"  ✗ 数据为None")
                continue
                
            streets = self.step_results[step_key]
            print(f"  ✓ 有数据，{len(streets)} 条道路")
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 仅线条
            streets.plot(ax=ax1, color=color, linewidth=1.0, alpha=0.8)
            ax1.set_title(f'{step_name} - 线条\n{len(streets)} 条道路', fontsize=14)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # 带节点
            streets.plot(ax=ax2, color=color, linewidth=1.0, alpha=0.8)
            self._plot_nodes(ax2, streets, color=color, size=15)
            ax2.set_title(f'{step_name} - 带节点\n{len(streets)} 条道路', fontsize=14)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            output_path = os.path.join(output_dir, f'{step_key}_visualization.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以节省内存
            
            print(f"  {step_name} 可视化已保存")
    
    def _visualize_final_comparison(self, output_dir):
        """创建最终对比图"""
        print("正在创建最终对比图...")
        
        # 创建对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 原始路网 - 仅线条
        self.original_streets.plot(ax=ax1, color='blue', linewidth=0.8, alpha=0.8)
        ax1.set_title(f'原始路网 - 线条\n{len(self.original_streets)} 条道路', fontsize=12)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. 修复后路网 - 仅线条
        self.fixed_streets.plot(ax=ax2, color='red', linewidth=0.8, alpha=0.8)
        ax2.set_title(f'momepy修复后路网 - 线条\n{len(self.fixed_streets)} 条道路', fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 3. 原始路网 - 带节点
        self.original_streets.plot(ax=ax3, color='blue', linewidth=0.8, alpha=0.8)
        self._plot_nodes(ax3, self.original_streets, color='blue', size=20)
        ax3.set_title(f'原始路网 - 带节点\n{len(self.original_streets)} 条道路', fontsize=12)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # 4. 修复后路网 - 带节点
        self.fixed_streets.plot(ax=ax4, color='red', linewidth=0.8, alpha=0.8)
        self._plot_nodes(ax4, self.fixed_streets, color='red', size=20)
        ax4.set_title(f'momepy修复后路网 - 带节点\n{len(self.fixed_streets)} 条道路', fontsize=12)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, 'final_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"最终对比图已保存到: {output_path}")
    
    def _save_statistics(self, output_dir):
        """保存统计信息"""
        stats = self.analyze_network()
        stats_path = os.path.join(output_dir, 'momepy_fix_stats.txt')
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Momepy 路网修复统计信息\n")
            f.write("=" * 40 + "\n")
            f.write(f"原始道路数量: {stats['original_count']}\n")
            f.write(f"修复后道路数量: {stats['fixed_count']}\n")
            f.write(f"减少道路数量: {stats['original_count'] - stats['fixed_count']}\n")
            f.write(f"减少率: {(stats['original_count'] - stats['fixed_count']) / stats['original_count'] * 100:.2f}%\n")
            f.write(f"原始总长度: {stats['original_length']:.2f}\n")
            f.write(f"修复后总长度: {stats['fixed_length']:.2f}\n")
            if 'density_original' in stats:
                f.write(f"原始路网密度: {stats['density_original']:.4f}\n")
                f.write(f"修复后路网密度: {stats['density_fixed']:.4f}\n")
            
            # 添加每个步骤的统计信息
            f.write("\n各步骤统计信息:\n")
            f.write("-" * 30 + "\n")
            for step_key, step_name, _ in [
                ('original', '原始路网', 'blue'),
                ('step1_remove_isolated_nodes', '步骤1: 移除孤立节点', 'green'),
                ('step2_extend_roads', '步骤2: 扩展道路', 'orange'),
                ('step3_remove_false_nodes', '步骤3: 移除假节点', 'red')
            ]:
                if step_key in self.step_results and self.step_results[step_key] is not None:
                    streets = self.step_results[step_key]
                    f.write(f"{step_name}: {len(streets)} 条道路\n")
        
        print(f"统计信息已保存到: {stats_path}")
    
    def _plot_nodes(self, ax, streets_gdf, color='black', size=20):
        """在给定的轴上绘制路网节点"""
        from shapely.geometry import Point
        
        # 收集所有节点
        nodes = set()
        for geom in streets_gdf.geometry:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                nodes.add(coords[0])  # 起点
                nodes.add(coords[-1])  # 终点
        
        # 绘制节点
        for node_coord in nodes:
            ax.scatter(node_coord[0], node_coord[1], 
                      c=color, s=size, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    def save_fixed_network(self, output_file="momepy_fixed_roads.shp"):
        """保存修复后的路网"""
        if self.fixed_streets is not None:
            self.fixed_streets.to_file(output_file)
            print(f"修复后的路网已保存到: {output_file}")
        else:
            print("错误：没有修复后的路网数据")

def main():
    """主函数"""
    # 检查small_scale_roads.shp是否存在
    if not os.path.exists("small_scale_roads.shp"):
        print("错误：未找到 small_scale_roads.shp 文件！")
        print("请先运行数据裁剪步骤创建 small_scale_roads.shp")
        return
    
    # 创建测试器
    tester = MomepyAutoFixTester("small_scale_roads.shp")
    
    # 加载数据
    tester.load_data()
    
    # 执行momepy自动修复
    tester.auto_fix_with_momepy()
    
    # 创建可视化对比
    tester.visualize_comparison()
    
    # 保存修复后的路网
    tester.save_fixed_network()
    
    print("Momepy Auto-fix 测试完成！")

if __name__ == "__main__":
    main() 