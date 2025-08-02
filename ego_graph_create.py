import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point, LineString
from shapely.ops import split
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict

class EgoGraphCreator:
    def __init__(self, shp_file):
        self.shp_file = shp_file
        self.gdf = None
        self.graph = None
        
    def load_and_split_linestrings(self):
        """加载数据并打断LineString"""
        print("正在加载数据...")
        self.gdf = gpd.read_file(self.shp_file)
        
        # 只保留LineString类型
        self.gdf = self.gdf[self.gdf.geometry.geom_type == 'LineString']
        print(f"原始LineString数量: {len(self.gdf)}")
        
        # 打断LineString
        split_geometries = []
        split_attributes = []
        
        for idx, row in self.gdf.iterrows():
            line = row.geometry
            coords = list(line.coords)
            
            # 在每个顶点处打断
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i+1]])
                split_geometries.append(segment)
                split_attributes.append(row.drop('geometry').to_dict())
        
        # 创建新的GeoDataFrame
        self.gdf = gpd.GeoDataFrame(split_attributes, geometry=split_geometries, crs=self.gdf.crs)
        print(f"打断后LineString数量: {len(self.gdf)}")
        
        # 保存打断后的数据
        self.gdf.to_file("split_roads.shp")
        print("打断后的数据已保存为: split_roads.shp")
        
    def load_test_roads(self):
        """直接从打断后的文件加载数据"""
        print("正在加载打断后的数据...")
        self.gdf = gpd.read_file("fixed_roads.shp")
        print(f"加载LineString数量: {len(self.gdf)}")
        
    def create_network_graph(self):
        """创建网络图"""
        print("正在创建网络图...")
        self.graph = nx.Graph()
        
        # 为每个线段添加节点和边
        for idx, row in tqdm(self.gdf.iterrows(), total=len(self.gdf), desc="创建网络图"):
            line = row.geometry
            coords = list(line.coords)
            
            # 添加端点作为节点
            start_node = f"node_{coords[0][0]}_{coords[0][1]}"
            end_node = f"node_{coords[-1][0]}_{coords[-1][1]}"
            
            self.graph.add_node(start_node, pos=coords[0])
            self.graph.add_node(end_node, pos=coords[-1])
            
            # 添加边
            self.graph.add_edge(start_node, end_node, 
                            geometry=line, 
                            highway=row.get('highway', 'unknown'),
                            width=row.get('width', 0),
                            length=row.get('length', line.length),
                            original_idx=idx)

    def find_footway_ego_graphs(self, num_samples=30, buffer_distance=250):
        """找到多个footway的ego-graph（基于500m的直径的 buffer）"""
        # 找到所有footway
        footways = []
        for edge in self.graph.edges(data=True):
            if edge[2].get('highway') == 'footway':
                footways.append(edge)
        if not footways:
            raise ValueError("没有找到footway类型的道路")
        
        # 随机选择footway
        selected_footways = random.sample(footways, min(num_samples, len(footways)))
        ego_graphs = []
        
        for i, selected_footway in enumerate(tqdm(selected_footways, desc="生成ego-graph")):
            # 获取选中的footway几何信息
            selected_geometry = selected_footway[2]['geometry']
            
            # 创建500m buffer
            buffer_zone = selected_geometry.buffer(buffer_distance / 111000)  # 转换为度（粗略估算）
            
            # 找到在buffer范围内的所有边（而不仅仅是节点）
            ego_edges = set()
            ego_nodes = set()
            
            for edge in self.graph.edges(data=True):
                geometry = edge[2]['geometry']
                
                # 检查边的几何形状是否与buffer相交
                if buffer_zone.intersects(geometry):
                    ego_edges.add((edge[0], edge[1]))
                    ego_nodes.add(edge[0])
                    ego_nodes.add(edge[1])
            
            # 提取子图
            ego_graph = self.graph.subgraph(ego_nodes).copy()
            
            # 计算子图的节点特征
            degrees = dict(ego_graph.degree())
            centrality = nx.betweenness_centrality(ego_graph)
            
            # 将特征添加到节点
            for node in ego_graph.nodes():
                ego_graph.nodes[node]['degree'] = degrees[node]
                ego_graph.nodes[node]['centrality'] = centrality[node]
            
            # 计算子图的边特征
            for edge in ego_graph.edges(data=True):
                length = edge[2].get('length', 0)
                width = edge[2].get('width', 0)
                edge[2]['length'] = length
                edge[2]['width'] = width
            
            ego_graphs.append((ego_graph, selected_footway, i+1))
            
        return ego_graphs
    
    
    def save_ego_graph_info(self, ego_graphs, output_dir="ego_graph_results"):
        """保存ego-graph的详细信息"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f"ego_graphs_{timestamp}")
        os.makedirs(results_dir)
        
        # 保存总体统计信息
        summary_data = []
        
        for ego_graph, selected_footway, graph_id in ego_graphs:
            # 统计不同道路类型
            highway_types = {}
            total_length = 0
            for edge in ego_graph.edges(data=True):
                hw_type = edge[2].get('highway', 'unknown')
                highway_types[hw_type] = highway_types.get(hw_type, 0) + 1
                total_length += edge[2]['geometry'].length
            
            # 收集信息
            graph_info = {
                'graph_id': graph_id,
                'nodes_count': ego_graph.number_of_nodes(),
                'edges_count': ego_graph.number_of_edges(),
                'selected_footway_length': selected_footway[2]['geometry'].length,
                'total_length': total_length,
                'highway_types': highway_types,
                'selected_footway_coords': list(selected_footway[2]['geometry'].coords)
            }
            summary_data.append(graph_info)
        
        # 保存为JSON
        with open(os.path.join(results_dir, 'ego_graphs_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(results_dir, 'ego_graphs_summary.csv'), index=False, encoding='utf-8')
        
        print(f"统计信息已保存到: {results_dir}")
        return results_dir
    
    def visualize_individual_ego_graph(self, ego_graph, selected_footway, graph_id, results_dir):
        """可视化单个ego-graph"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 网络图可视化
        pos = nx.get_node_attributes(ego_graph, 'pos')
        nx.draw(ego_graph, pos, ax=ax1,
               node_color='lightblue',
               node_size=50,
               edge_color='gray',
               width=2,
               with_labels=False)
        
        # 高亮选中的footway（检查节点是否还存在）
        if selected_footway is not None:
            selected_edge = (selected_footway[0], selected_footway[1])
            if ego_graph.has_edge(selected_edge[0], selected_edge[1]):
                nx.draw_networkx_edges(ego_graph, pos,
                                     edgelist=[selected_edge],
                                     edge_color='red',
                                     width=4,
                                     ax=ax1)
            else:
                print(f"  警告：选中的footway边 {selected_edge} 在修复后不存在")
        else:
            print(f"  警告：选中的footway在修复后不存在")
        
        ax1.set_title(f'EGO-GRAPH {graph_id} - Network View\n{ego_graph.number_of_nodes()} 节点, {ego_graph.number_of_edges()} 边')
        
        # 2. 地图可视化
        # 绘制ego-graph中的所有边
        for edge in ego_graph.edges(data=True):
            geometry = edge[2]['geometry']
            highway_type = edge[2]['highway']
            
            # 不同等级道路颜色：红色(primary) > 橙色(secondary) > 紫色(tertiary) > 蓝色(footway)
            color_map = {
                'primary': 'red',
                'secondary': 'orange', 
                'tertiary': 'purple',
                'footway': 'blue'
            }
            
            color = color_map.get(highway_type, 'gray')
            
            # 根据道路等级设置线宽
            width_map = {
                'primary': 4,
                'secondary': 3,
                'tertiary': 2,
                'footway': 1
            }
            linewidth = width_map.get(highway_type, 1)
            ax2.plot(*geometry.xy, color=color, linewidth=linewidth, alpha=0.8)
        
        # 高亮选中的footway为绿色（检查边是否还存在）
        if selected_footway is not None:
            selected_edge = (selected_footway[0], selected_footway[1])
            if ego_graph.has_edge(selected_edge[0], selected_edge[1]):
                selected_geometry = selected_footway[2]['geometry']
                ax2.plot(*selected_geometry.xy, color='blue', linewidth=2, alpha=1.0)
            else:
                # 如果边不存在，尝试找到对应的几何信息
                print(f"  警告：选中的footway边 {selected_edge} 在修复后不存在，跳过高亮显示")
        else:
            print(f"  警告：选中的footway在修复后不存在，跳过高亮显示")
        
        ax2.set_title(f'EGO-GRAPH {graph_id} - Map View')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=4, label='Primary'),
            plt.Line2D([0], [0], color='orange', linewidth=3, label='Secondary'),
            plt.Line2D([0], [0], color='purple', linewidth=2, label='Tertiary'),
            plt.Line2D([0], [0], color='blue', linewidth=1, label='Footway'),
            plt.Line2D([0], [0], color='green', linewidth=6, label='Selected Path')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(results_dir, f'ego_graph_{graph_id:03d}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  EGO-GRAPH {graph_id} 可视化已保存")
    
    def visualize_all_ego_graphs(self, ego_graphs, results_dir):
        """在整个城市地图中可视化所有ego-graph"""
        print("正在创建整体可视化...")
        
        fig, ax = plt.subplots(1, 1, figsize=(24, 18))
        
        # 绘制所有道路（灰色）
        print("绘制所有道路...")
        for edge in self.graph.edges(data=True):
            geometry = edge[2]['geometry']
            ax.plot(*geometry.xy, color='lightgray', linewidth=0.5, alpha=0.6)
        
        # 为每个ego-graph绘制黑色加粗路网并添加方框
        for i, (ego_graph, selected_footway, graph_id) in enumerate(ego_graphs):
            # 绘制ego-graph范围内的所有边为黑色加粗
            for edge in ego_graph.edges(data=True):
                geometry = edge[2]['geometry']
                ax.plot(*geometry.xy, color='black', linewidth=3, alpha=0.8)
            
            # 高亮选中的footway为绿色（检查边是否还存在）
            selected_edge = (selected_footway[0], selected_footway[1])
            if ego_graph.has_edge(selected_edge[0], selected_edge[1]):
                selected_geometry = selected_footway[2]['geometry']
                ax.plot(*selected_geometry.xy, color='green', linewidth=5, alpha=1.0, zorder=10)
            else:
                print(f"  警告：选中的footway边 {selected_edge} 在修复后不存在，跳过高亮显示")
            
            # 计算ego-graph的边界框
            all_coords = []
            for edge in ego_graph.edges(data=True):
                geometry = edge[2]['geometry']
                all_coords.extend(list(geometry.coords))
            
            if all_coords:
                coords_array = np.array(all_coords)
                min_x, min_y = coords_array.min(axis=0)
                max_x, max_y = coords_array.max(axis=0)
                
                # 绘制方框
                rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                   fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(rect)
                
                # 在方框左上角添加序列标志
                ax.text(min_x, max_y, str(graph_id), color='red', fontsize=14, fontweight='bold',
                       ha='left', va='top', zorder=20,
                       bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.9))
        
        ax.set_title(f'南京市道路网络 - {len(ego_graphs)}条随机Footway的500m Buffer EGO-GRAPH\n(绿色为选中的路径，黑色为ego-graph范围，红色方框为边界)', 
                    fontsize=18, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        legend_elements = [
            plt.Line2D([0], [0], color='lightgray', linewidth=2, label='其他道路'),
            plt.Line2D([0], [0], color='black', linewidth=3, label='EGO-GRAPH范围'),
            plt.Line2D([0], [0], color='green', linewidth=5, label='选中的路径'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='边界框')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(results_dir, 'all_ego_graphs_overview.png'), dpi=300, bbox_inches='tight')
        
        # 显示图片
        plt.show()

    
    def print_statistics(self, ego_graphs):
        """打印统计信息"""
        print("\n" + "="*60)
        print("EGO-GRAPH统计信息")
        print("="*60)
        
        total_nodes = 0
        total_edges = 0
        total_length = 0
        
        for ego_graph, selected_footway, graph_id in ego_graphs:
            print(f"\n第 {graph_id} 条Footway:")
            print(f"  节点数量: {ego_graph.number_of_nodes()}")
            print(f"  边数量: {ego_graph.number_of_edges()}")
            print(f"  选中的footway长度: {selected_footway[2]['geometry'].length:.2f}")
            
            # 统计不同道路类型
            highway_types = {}
            graph_length = 0
            for edge in ego_graph.edges(data=True):
                hw_type = edge[2].get('highway', 'unknown')
                highway_types[hw_type] = highway_types.get(hw_type, 0) + 1
                graph_length += edge[2]['geometry'].length
            
            print(f"  总长度: {graph_length:.2f}")
            print(f"  道路类型分布:")
            for hw_type, count in highway_types.items():
                print(f"    {hw_type}: {count} 条")
            
            total_nodes += ego_graph.number_of_nodes()
            total_edges += ego_graph.number_of_edges()
            total_length += graph_length

# 智能主函数 - 自动检测fixed_roads.shp是否存在
def smart_main():
    """智能主函数：自动检测fixed_roads.shp是否存在，决定是否重新创建"""
    
    # 检查fixed_roads.shp是否存在
    if os.path.exists("fixed_roads.shp"):
        print("✓ 检测到 fixed_roads.shp 文件，直接使用现有数据")
        print("=" * 50)
        
        # 使用现有的fixed_roads.shp
        creator = EgoGraphCreator("fixed_roads.shp")
        creator.load_test_roads()
        
    else:
        print("✗ 未检测到 fixed_roads.shp 文件，需要重新创建")
        print("=" * 50)
        
        # 检查merged_roads.shp是否存在
        if not os.path.exists("fixed_roads.shp"):
            print("错误：未找到 fixed_roads.shp 文件！")
            print("请先运行数据合并步骤创建 fixed_roads.shp")
            return
        
        # 创建新的fixed_roads.shp
        creator = EgoGraphCreator("fixed_roads.shp")
        creator.load_and_split_linestrings()
    
    # 创建网络图
    creator.create_network_graph()

    # 找到100条footway的ego-graph（基于500m buffer）
    ego_graphs = creator.find_footway_ego_graphs(num_samples=30, buffer_distance=250)
    
    # 保存详细信息
    results_dir = creator.save_ego_graph_info(ego_graphs)
    
    # 保存ego-graph对象
    print("\n正在保存ego-graph对象...")
    import pickle
    
    # 保存所有ego-graph
    ego_graphs_data = []
    for ego_graph, selected_footway, graph_id in ego_graphs:
        ego_graphs_data.append({
            'graph': ego_graph,
            'footway': selected_footway,
            'id': graph_id
        })
    
    # 保存到results_dir目录
    with open(os.path.join(results_dir, 'ego_graphs.pkl'), 'wb') as f:
        pickle.dump(ego_graphs_data, f)
    
    print(f"✓ ego-graph对象已保存到: {os.path.join(results_dir, 'ego_graphs.pkl')}")
    
    # 为每个ego-graph创建可视化
    print("\n正在创建单个ego-graph可视化...")
    for ego_graph, selected_footway, graph_id in ego_graphs:
        creator.visualize_individual_ego_graph(ego_graph, selected_footway, graph_id, results_dir)
    
    # 创建整体可视化
    creator.visualize_all_ego_graphs(ego_graphs, results_dir)
    
    # 打印统计信息
    creator.print_statistics(ego_graphs)
    
    print(f"\n所有结果已保存到: {results_dir}")


if __name__ == "__main__":
    # 使用智能主函数
    smart_main() 