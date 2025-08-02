import copy
import logging
import random
import time
import uuid
import math
from collections import defaultdict
from typing import Union, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon

from gui import global_var as g
from lib.accelerator import cAccelerator, cRoadAccelerator, arrs_addr_len
from utils import RoadLevel, RoadState, point_utils, road_utils, RoadCluster
from utils.common_utils import timer

PRECISION = 10


class RoadCollection:
    """Road类提供了对道路数据进行操作和分析的方法。
    它可以添加、删除、合并和分割道路，计算道路的交点和相交关系，以及获取道路和节点的信息。
    此外，Road类还提供了缓存和自动修复功能，以提高数据处理效率。
    通过Road类，可以方便地处理和管理道路数据。"""
    __node_attrs = ['uid', 'geometry', 'coord', 'cache']
    #  UUID.uuid4 | shapely.Geometry | tuple | bool
    __edge_attrs = ['u', 'v', 'uid', 'geometry', 'coords', 'level', 'state', 'cache', 'geohash']
    # UUID.uuid4 | UUID.uuid4 | UUID.uuid4 | shapely.Geometry | np.ndarray | RoadLevel | RoadState | bool | int
    __node_gdf = gpd.GeoDataFrame(columns=__node_attrs)
    __node_gdf.set_index('uid')

    __edge_gdf = gpd.GeoDataFrame(columns=__edge_attrs)
    __edge_gdf.set_index('uid')

    __coord_to_node_uid = {}  # 使用hashset代替dataframe加速_node_uid_from_coord函数

    __cached_graph = None
    __cached_node_gdf = None
    __cached_edge_gdf = None

    _flag_cached_graph_need_update = False
    __uid = uuid.uuid4()  # 整体的uid
    # region 全局
    def uid(self):
        return self.__uid

    # endregion

    # region 节点相关

    def get_node_by_uid(self, uid):
        return self.__node_gdf.loc[uid]

    def get_node_by_index(self, idx: int) -> pd.Series:
        assert isinstance(idx, int)
        node = self.__node_gdf.iloc[idx]
        return node

    def get_nodes_by_indexes(self, idx_list: list[int]) -> gpd.GeoDataFrame:
        nodes = self.__node_gdf.iloc[idx_list]
        return nodes

    def get_node_coord(self, uid) -> tuple:
        return self.get_node_by_uid(uid)['coord']

    def _get_or_create_node_by_coord(self, coord: tuple) -> uuid.UUID:
        """
        返回所处coord坐标的node的uid信息。
        如果提供的coord坐标处存在现有的node，则直接返回该node的uid；
        否则将会自动在该坐标处创建一个新的node并返回新的node的uid。
        """
        # validation
        if not isinstance(coord, tuple):
            coord = tuple(coord)
        assert len(coord) == 2
        # 离散化
        x = round(coord[0] * PRECISION) / PRECISION
        y = round(coord[1] * PRECISION) / PRECISION
        coord = (x, y)
        # 判断是否已经存在
        if coord in self.__coord_to_node_uid:
            return self.__coord_to_node_uid[coord]
        else:
            # 创建新的node
            uid = uuid.uuid4()
            new_row = {'geometry': [Point(coord)], 'coord': [coord], 'uid': [uid], 'cache': [False]}
            new_gdf = gpd.GeoDataFrame(new_row, index=new_row['uid'])
            if not self.__node_gdf.empty:
                self.__node_gdf = gpd.pd.concat([self.__node_gdf, new_gdf], ignore_index=False)
            else:
                self.__node_gdf = new_gdf
            # 添加coord至__coord_to_node_uid
            self.__coord_to_node_uid[coord] = uid
            self.__uid = uuid.uuid4()  # 更新global uid
            return uid

    def _get_node_by_coord(self, coord: tuple) -> Optional[uuid.UUID]:
        """判断坐标处是否存在node，如果存在，返回node的uid， 如果不存在，返回None"""
        if not isinstance(coord, tuple):
            coord = tuple(coord)
        assert len(coord) == 2
        x = round(coord[0] * PRECISION) / PRECISION
        y = round(coord[1] * PRECISION) / PRECISION
        coord = (x, y)
        if coord in self.__coord_to_node_uid:
            return self.__coord_to_node_uid[coord]
        else:
            return None


    def _delete_node(self, uid):
        """直接删除node ，不考虑引用关系"""
        node = self.__node_gdf.loc[uid]
        coord = node['coord']
        x = round(coord[0] * PRECISION) / PRECISION
        y = round(coord[1] * PRECISION) / PRECISION
        coord = (x, y)
        self.__node_gdf.drop(uid, inplace=True)
        self.__coord_to_node_uid.pop(coord)  # 同时删除coord_to_node_uid中缓存的坐标
        self.__uid = uuid.uuid4()

    def _clear_node(self, uid):
        """如果没有任何road引用node， 才会将其删除"""
        if not self._any_road_using_node(uid):
            self._delete_node(uid)

    def clear_unused_nodes(self):
        """清除所有没有被使用的node"""
        valid_nodes = self._get_nodes_by_roads(self.get_all_roads())
        self.__node_gdf = valid_nodes
        self.rebuild_coord_to_uid_dict()

    def rebuild_coord_to_uid_dict(self):
        """重建Road.__coord_to_node_uid"""
        self.__coord_to_node_uid = {row['coord']: uid for uid, row in self.__node_gdf.iterrows()}

    def get_all_nodes(self):
        return self.__node_gdf

    def _get_nodes_by_roads(self, roads):
        """返回被roads引用的所有node"""
        return self.__node_gdf[self.__node_gdf['uid'].isin(roads['u']) | self.__node_gdf['uid'].isin(roads['v'])]

    def _any_road_using_node(self, uid) -> bool:
        """判断是否有其他任何道路正在使用该node"""
        return any(self.__edge_gdf['u'].eq(uid)) or any(self.__edge_gdf['v'].eq(uid))

    def get_close_nodes(self, eps=0.1, min_samples=2) -> dict[int:gpd.GeoDataFrame]:
        """
        获取几乎重叠的nodes组
        :param eps: 领域的大小
        :param min_samples: 领域内最小样本数, 默认为2
        :return: 靠近的nodes的分组dict， key为组别， value为GeoDataFrame， 包含了这个组的所有nodes
        """
        vertex_coords = np.vstack(self.__node_gdf['coord'].apply(np.array).values).astype(np.float32)
        # 从向量数组或距离矩阵执行 DBSCAN 聚类。
        # DBSCAN - 基于密度的噪声应用程序空间聚类。 找到高密度的核心样本并从中扩展簇。 适用于包含相似密度簇的数据。
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(vertex_coords)
        # 将点按簇进行分组
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(i)
        groups = dict(groups)
        groups.pop(-1)
        groups_nodes = {label: self.__node_gdf.iloc[idx_list] for label, idx_list in groups.items()}
        return groups_nodes

    @staticmethod
    def get_nodes_avg_coord(nodes: gpd.GeoDataFrame):
        avg_coord = nodes['coord'].apply(pd.Series).mean()
        return tuple(avg_coord)

    def merge_nodes(self, nodes: gpd.GeoDataFrame):
        logging.debug(f'================正在执行合并nodes===============')
        logging.debug(f'正在合并 [{len(nodes)}]个 nodes, 原始nodes的信息如下')
        logging.debug('----------------------------------------------------------------------------')
        logging.debug(f'坐标为: {", ".join([str(node["coord"]) for node_uid, node in nodes.iterrows()])}')
        logging.debug(f'uid为: {", ".join([str(node_uid) for node_uid, node in nodes.iterrows()])}')
        logging.debug('----------------------------------------------------------------------------')
        avg_coord = self.get_nodes_avg_coord(nodes)  # (tuple)
        new_node_uid = self._get_or_create_node_by_coord(avg_coord)  # create or get new node
        new_node_coord = self.get_node_coord(new_node_uid)  # 重新获取坐标 (tuple)
        logging.debug(f'新node的uid为: {new_node_uid}')
        logging.debug(f'新node的坐标为: {new_node_coord}')
        new_node_coord = np.array(new_node_coord)  # (2, )
        updated_roads_uid = set()
        i = 0
        for node_uid, node in nodes.iterrows():
            logging.debug(f'[node {i}] 正在分析node ({node["uid"]})')
            related_roads = self.get_roads_by_node(node)
            logging.debug(f'[node {i}] 找到[{len(related_roads)}]条相关road')
            j = 0
            for road_uid, road in related_roads.iterrows():
                if road_uid in updated_roads_uid:
                    logging.debug(f'    [road {j}] 遇到了已经计算过的road({road_uid})， 跳过')
                    continue
                updated_roads_uid.add(road_uid)
                logging.debug(f'    [road {j}] 准备更新 road ({road_uid})')
                u_coord = self.get_node_coord(road['u'])  # tuple
                v_coord = self.get_node_coord(road['v'])  # tuple
                u_dist_sq = (new_node_coord[0] - u_coord[0]) ** 2 + (new_node_coord[1] - u_coord[1]) ** 2
                v_dist_sq = (new_node_coord[0] - v_coord[0]) ** 2 + (new_node_coord[1] - v_coord[1]) ** 2
                logging.debug(f'    [road {j}] 新road两端距离新node的距离平方分别为: {u_dist_sq}, {v_dist_sq}')
                if u_dist_sq < v_dist_sq:
                    logging.debug(f'    [road {j}] u端更近')
                    # road_coords[0] = new_node_coord
                    self.replace_u(road, new_node_uid)

                else:
                    logging.debug(f'    [road {j}] v端更近')
                    # road_coords[-1] = new_node_coord
                    self.replace_v(road, new_node_uid)

                # self.update_road_points(road, road_coords)
                logging.debug(f'    [road {j}] 已更新road ({road["uid"]})')
                j += 1
            i += 1
        logging.debug(f'================合并完成===============')

    def merge_all_close_nodes(self):
        groups = self.get_close_nodes()
        for _, nodes in groups.items():
            self.merge_nodes(nodes)
        self.clear_unused_nodes()

    # endregion

    # region 创建删除

    def _create_road_by_coords(self, coords: np.ndarray,
                               level: RoadLevel,
                               state: RoadState) -> pd.Series:
        """
        从坐标创建road。
        仅创建，不添加到Road.__edge_gdf中。
        注意，虽然创建的road没有加到Road的edge gdf中，但这里的节点将直接加到Road的node gdf中
        """
        assert len(coords.shape) == 2, '提供的coords数组维度必须为2，例如(n, 2)， 其中n表示点的个数'
        coords = coords.copy()
        u = self._get_or_create_node_by_coord(coords[0])
        v = self._get_or_create_node_by_coord(coords[-1])
        # 重新更新首尾坐标
        coords[0] = np.array(self.get_node_coord(u))
        coords[-1] = np.array(self.get_node_coord(v))
        geometry = point_utils.points_to_geo(coords)
        uid = uuid.uuid4()
        new_row = {'u': [u],
                   'v': [v],
                   'geometry': [geometry],
                   'coords': [coords],
                   'level': [level],
                   'state': [state],
                   'uid': [uid],
                   'cache': False,
                   'geohash': hash(coords.tobytes())
                   }
        return gpd.GeoDataFrame(new_row, index=new_row['uid']).iloc[0]

    def create_roads_by_coords(self, coords_list: list[np.ndarray],
                               levels_list: list[RoadLevel],
                               states_list: list[RoadState]) -> gpd.GeoDataFrame:
        assert len(coords_list) == len(levels_list) == len(states_list)
        u_list = [self._get_or_create_node_by_coord(coords[0]) for coords in coords_list]
        v_list = [self._get_or_create_node_by_coord(coords[-1]) for coords in coords_list]
        # 重新计算首尾coords
        for i, coords in enumerate(coords_list):
            coords[0] = np.array(self.get_node_coord(u_list[i]))
            coords[-1] = np.array(self.get_node_coord(v_list[i]))
        geometry_list = [point_utils.points_to_geo(coords) for coords in coords_list]
        uid_list = [uuid.uuid4() for _ in coords_list]
        cache_list = [False for _ in coords_list]
        geohash_list = [hash(coords.tobytes()) for coords in coords_list]
        new_data = {'u': u_list,
                    'v': v_list,
                    'geometry': geometry_list,
                    'coords': coords_list,
                    'level': levels_list,
                    'state': states_list,
                    'uid': uid_list,
                    'cache': cache_list,
                    'geohash': geohash_list
                    }
        return gpd.GeoDataFrame(new_data, index=new_data['uid'])

    def add_road(self, road: pd.Series, return_uid: bool = True) -> Optional[uuid.UUID]:
        """添加road至Road.__edge_gdf"""
        assert isinstance(road, pd.Series)
        road_df = road.to_frame().T
        if not self.__edge_gdf.empty:
            self.__edge_gdf = gpd.pd.concat([self.__edge_gdf, road_df], ignore_index=False)
        else:
            self.__edge_gdf = road_df
        self.__uid = uuid.uuid4()
        if return_uid:
            return road['uid']

    def add_roads(self, roads: gpd.GeoDataFrame, return_uid: bool = False) -> Optional[list[uuid.UUID]]:
        """添加roads至Road.__edge_gdf"""
        assert isinstance(roads, gpd.GeoDataFrame)
        if not self.__edge_gdf.empty:
            self.__edge_gdf = gpd.pd.concat([self.__edge_gdf, roads], ignore_index=False)
        else:
            self.__edge_gdf = roads
        self.__uid = uuid.uuid4()
        if return_uid:
            return list(roads['uid'])

    def add_road_by_coords(self, coords: np.ndarray,
                           level: RoadLevel,
                           state: RoadState,
                           return_uid=True) -> Optional[uuid.UUID]:
        """通过坐标创建road并添加至Road.__edge_gdf"""
        road = self._create_road_by_coords(coords, level, state)
        return self.add_road(road, return_uid)

    def add_roads_by_coords(self, coords_list: list[np.ndarray],
                            levels_list: list[RoadLevel],
                            states_list: list[RoadState],
                            return_uid=False) -> Optional[list[uuid.UUID]]:
        """通过坐标创建roads并添加至Road.__edge_gdf"""
        roads = self.create_roads_by_coords(coords_list, levels_list, states_list)
        return self.add_roads(roads, return_uid)

    def delete_road(self, road: pd.Series, update_nodes_immediately=True):
        """
        删除road，默认自动清理node
        :param road: 要被删除的road
        :param update_nodes_immediately: 如果需要一次性大量删除road时，则可以不立即排查node是否要被删除。可以在所有road都被删除后统一安排清理
        :return:
        """
        assert isinstance(road, pd.Series)
        uid = road['uid']
        u = road['u']
        v = road['v']

        self.__edge_gdf.drop(uid, inplace=True)
        self.__uid = uuid.uuid4()

        # handle cache
        if road['cache']:
            # 如果road是cached过的， 那么任何更改都会导致cache失效
            self._flag_cached_graph_need_update = True

            # delete unused nodes
        if update_nodes_immediately:
            self._clear_node(u)
            self._clear_node(v)

    def delete_road_by_uid(self, uid, update_nodes_immediately=True):
        """通过uid删除road， 其他信息参考Road.delete_road方法"""
        self.delete_road(self.get_road_by_uid(uid), update_nodes_immediately)

    def delete_roads_by_uids_list(self, uids_list: list[uuid.UUID]):
        for uid in uids_list:
            self.delete_road_by_uid(uid)

    def delete_all(self, clear_cache=True):
        """清空所有内容， 默认cache也将被清除"""
        self.__edge_gdf.drop(self.__edge_gdf['uid'], inplace=True)
        self.__node_gdf.drop(self.__node_gdf['uid'], inplace=True)
        self.__coord_to_node_uid = {}
        if clear_cache:
            self.clear_cache()
        self.__uid = uuid.uuid4()

    # endregion

    # region 获取查找

    def get_edge_attrs(self):
        return self.__edge_attrs

    def get_node_attrs(self):
        return self.__node_attrs

    def get_road_by_uid(self, uid: uuid.UUID):
        assert isinstance(uid, uuid.UUID)
        road = self.__edge_gdf.loc[uid]
        return road

    def get_road_by_index(self, idx: int) -> pd.Series:
        assert isinstance(idx, int)
        road = self.__edge_gdf.iloc[idx]
        return road

    def get_roads_by_indexes(self, idx_list: list[int]) -> gpd.GeoDataFrame:
        roads = self.__edge_gdf.iloc[idx_list]
        return roads

    def get_roads_by_attr_and_value(self, attr: str, value: any) -> gpd.GeoDataFrame:
        assert attr in self.__edge_attrs, f'unexpected attr ({attr}), attr must be one of these: {self.__edge_attrs}'
        roads = self.__edge_gdf.loc[self.__edge_gdf[attr] == value]
        return roads

    def get_roads_except_attr_and_value(self, attr: str, value: any) -> gpd.GeoDataFrame:
        assert attr in self.__edge_attrs, f'unexpected attr ({attr}), attr must be one of these: {self.__edge_attrs}'
        roads = self.__edge_gdf.loc[self.__edge_gdf[attr] != value]
        return roads

    def get_node_by_attr_and_value(self, attr: str, value: any):
        assert attr in self.__node_attrs, f'unexpected attr ({attr}), attr must be one of these: {self.__node_attrs}'
        nodes = self.__node_gdf.loc[self.__node_gdf[attr] == value]
        return nodes

    def get_first_road(self):
        return self.get_road_by_index(0)

    def get_last_road(self):
        return self.get_road_by_index(-1)

    def get_all_roads(self):
        return self.__edge_gdf

    def get_roads_by_node(self, node):
        node_uid = node['uid']
        roads1 = self.get_roads_by_attr_and_value('u', node_uid)
        roads2 = self.get_roads_by_attr_and_value('v', node_uid)
        return pd.concat([roads1, roads2])

    def get_roads_by_cluster(self, cluster: RoadCluster):
        cluster = cluster.cluster
        uid_sets_by_attr = []
        for attr in cluster:
            gdfs = []
            if all(cluster[attr].values()):
                continue
            for key in cluster[attr]:
                if cluster[attr][key]:
                    _gdfs = self.get_roads_by_attr_and_value(attr, key)
                    gdfs.append(_gdfs)
            if len(gdfs) == 0:
                return None
            gdf = pd.concat(gdfs, ignore_index=False)
            uid_sets_by_attr.append(set(gdf.index))
        if len(uid_sets_by_attr) == 0:
            return self.get_all_roads()
        common_uid = list(set.intersection(*uid_sets_by_attr))
        return self.get_all_roads().loc[common_uid]

    def get_road_by_hash(self, hash_code) -> pd.Series:
        roads = self.get_roads_by_attr_and_value('geohash', hash_code)
        assert not roads.empty, '未找到road'
        assert len(roads) == 1, f'产生了hash碰撞, roads信息如下 {roads}'
        return roads.iloc[0]

    def get_roads_by_hashes(self, hashes) -> gpd.GeoDataFrame:
        hashes = set(hashes)  # 去重
        gdfs = []
        for h in hashes:
            roads = self.get_roads_by_attr_and_value('geohash', h)
            if not roads.empty:
                gdfs.append(roads)
        assert len(gdfs) > 0
        return pd.concat(gdfs, ignore_index=False)

    @staticmethod
    def get_valid_spawn_range(road: pd.Series):
        """求可以生成新路的位置，如果没有，返回None"""
        line_string: LineString = road['geometry']
        dist_threshold = road_utils.distance_threshold_by_road_level[road['level']]
        if line_string.length < 2 * dist_threshold:
            return None, None
        return dist_threshold, line_string.length - dist_threshold

    @staticmethod
    def get_road_last_point(road: pd.Series) -> np.ndarray:
        # TODO 这部分geometry可以被更新成coords了
        geo = road['geometry']
        coord = list(geo.coords[-1])
        return np.array([coord])

    @staticmethod
    def get_road_first_point(road: pd.Series) -> np.ndarray:
        # TODO 这部分geometry可以被更新成coords了
        geo = road['geometry']
        coord = list(geo.coords[0])
        return np.array([coord])

    @staticmethod
    def get_road_sum_distance(road: pd.Series) -> np.ndarray:
        from math import sqrt
        geo = road['geometry']
        coords = list(geo.coords)
        distance = 0
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            distance += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return np.array([distance])

    @staticmethod
    def get_road_length_using_coords(road: pd.Series):
        coords = road['coords']
        total_length = 0
        for i in range(len(coords) - 1):
            total_length += np.linalg.norm(coords[i] - coords[i + 1])
        return total_length

    @staticmethod
    def get_road_last_element(road: pd.Series) -> Union[LineString, Point]:
        geo = road['geometry']
        coords = list(geo.coords)
        if len(coords) == 0:
            return geo
        pt1 = coords[-1]
        pt2 = coords[-2]
        return LineString([pt1, pt2])

    @staticmethod
    def get_road_first_element(road: pd.Series) -> Union[LineString, Point]:
        geo = road['geometry']
        coords = list(geo.coords)
        if len(coords) == 0:
            return geo
        pt1 = coords[0]
        pt2 = coords[1]
        return LineString([pt1, pt2])

    def get_dead_nodes(self) -> gpd.GeoDataFrame:
        """返回所有非边界上的死节点（只连接一条路的节点）"""
        dead_nodes_gdf, _ = self.get_dead_ends()
        return dead_nodes_gdf if dead_nodes_gdf is not None else gpd.GeoDataFrame(columns=self.__node_gdf.columns)

    def get_cross_nodes(self) -> gpd.GeoDataFrame:
        """返回所有连接多条道路的交叉节点"""
        self.clear_unused_nodes()
        cross_nodes = []
        for node_uid, node in self.__node_gdf.iterrows():
            roads = self.get_roads_by_node(node)
            if len(roads) > 3:
                cross_nodes.append(node)
        if cross_nodes:
            return gpd.GeoDataFrame(cross_nodes, columns=self.__node_gdf.columns)
        else:
            return gpd.GeoDataFrame(columns=self.__node_gdf.columns)

    def get_connectable_nodes(self) -> gpd.GeoDataFrame:
        """返回所有连接 2 或 3 条道路的可连接节点"""
        self.clear_unused_nodes()
        connectable_nodes = []
        for node_uid, node in self.__node_gdf.iterrows():
            roads = self.get_roads_by_node(node)
            if len(roads) in [2, 3]:
                connectable_nodes.append(node)
        if connectable_nodes:
            return gpd.GeoDataFrame(connectable_nodes, columns=self.__node_gdf.columns)
        else:
            return gpd.GeoDataFrame(columns=self.__node_gdf.columns)

    def get_dead_ends(self):
        self.clear_unused_nodes()
        min_coords, max_coords = self.get_bbox()
        dead_nodes_gdf, dead_roads_gdf = None, None

        for node_uid, node in self.__node_gdf.iterrows():
            roads = self.get_roads_by_node(node)
            if len(roads) != 1:
                continue
            x, y = node['coord']
            if math.isclose(x, min_coords[0], abs_tol=50) or math.isclose(x, max_coords[0], abs_tol=50) or \
                    math.isclose(y, min_coords[1], abs_tol=50) or math.isclose(y, max_coords[1], abs_tol=50):
                continue
            dead_nodes_gdf = node.to_frame().T if dead_nodes_gdf is None else \
                gpd.pd.concat([dead_nodes_gdf, node.to_frame().T], ignore_index=False)
            dead_roads_gdf = roads if dead_roads_gdf is None else \
                gpd.pd.concat([dead_roads_gdf, roads], ignore_index=False)
        return dead_nodes_gdf, dead_roads_gdf

    def get_bbox(self, roads=None):
        """
        获取指定道路gdf的bounding box范围
        :param roads: 要获取范围的roads gdf， None表示所有道路
        :return:
        """
        if roads is None:
            roads = self.__edge_gdf
        all_coords = np.concatenate(roads['coords'].values)
        min_coords = np.amin(all_coords, axis=0)
        max_coords = np.amax(all_coords, axis=0)
        return min_coords, max_coords
    # endregion

    # region 编辑修改

    def add_point_to_road(self, road: pd.Series, point: np.ndarray, update_nodes_immediately=True):
        assert isinstance(road, pd.Series)
        if len(point.shape) == 1:
            point = np.unsqueeze(point, axis=0)
        return self.add_points_to_road(road, point, update_nodes_immediately)

    def add_points_to_road(self, road: pd.Series, points: np.ndarray, update_nodes_immediately=True):
        assert len(points.shape) == 2
        assert isinstance(road, pd.Series)
        org_coords = road['coords']
        new_points = np.vstack([org_coords, points])
        return self.update_road_points(road, new_points, update_nodes_immediately)

    def update_road_points(self, road: pd.Series, points: np.ndarray, update_nodes_immediately=True):
        assert len(points.shape) == 2
        assert isinstance(road, pd.Series)
        assert points.shape[0] >= 2
        uid = road['uid']
        org_u = road['u']
        org_v = road['v']

        new_points = points
        new_u = self._get_or_create_node_by_coord(new_points[0])
        new_v = self._get_or_create_node_by_coord(new_points[-1])
        # 重新根据u和v的node的坐标 更新 new_points
        new_points[0] = np.array(self.get_node_coord(new_u))
        new_points[-1] = np.array(self.get_node_coord(new_v))
        new_geo = LineString(new_points)

        self.__edge_gdf.at[uid, 'coords'] = new_points
        self.__edge_gdf.loc[uid, 'geometry'] = new_geo
        self.__edge_gdf.loc[uid, 'u'] = new_u
        self.__edge_gdf.loc[uid, 'v'] = new_v
        # handle org nodes
        if update_nodes_immediately:
            self._clear_node(org_u)
            self._clear_node(org_v)
        # handle cache
        try:
            if road['cache']:
                self._flag_cached_graph_need_update = True
        except Exception as e:
            print(e)
        return self.get_road_by_uid(uid)

    def replace_u(self, road, u_uid):
        road_uid = road['uid']
        new_coords = road['coords'].copy()
        new_coords[0] = np.array(self.get_node_coord(u_uid))
        new_geo = LineString(new_coords)
        self.__edge_gdf.at[road_uid, 'coords'] = new_coords
        self.__edge_gdf.loc[road_uid, 'geometry'] = new_geo
        self.__edge_gdf.loc[road_uid, 'u'] = u_uid

    def replace_v(self, road, v_uid):
        road_uid = road['uid']
        new_coords = road['coords'].copy()
        new_coords[-1] = np.array(self.get_node_coord(v_uid))
        new_geo = LineString(new_coords)
        self.__edge_gdf.at[road_uid, 'coords'] = new_coords
        self.__edge_gdf.loc[road_uid, 'geometry'] = new_geo
        self.__edge_gdf.loc[road_uid, 'v'] = v_uid

    @staticmethod
    def detect_intersection(candidate_roads: pd.DataFrame, all_roads: pd.DataFrame):
        """
        计算道路相交， all_roads必须全包含roads。该方法依赖c#实现
        :param candidate_roads: 准备计算相交的道路
        :param all_roads: 参与计算的道路域
        :return: 返回谁和谁相交在哪 1. 相交的道路uid：list[tuple[uuid.UUID, uuid.UUID]] 2. 交点：list[tuple[float, float]]
        """
        _start_time = time.time()
        in_first = np.empty(len(all_roads), dtype=np.int32)
        in_num = np.empty(len(all_roads), dtype=np.int32)
        vertex_coords = all_roads['coords']
        i = 0
        total = 0
        for coords in vertex_coords:
            num = len(coords)
            in_first[i] = total
            in_num[i] = num
            total += num
            i += 1
        vertex_coords = np.concatenate(vertex_coords.values, dtype=np.float32, axis=0)
        in_vertex_x = vertex_coords[:, 0].copy()
        in_vertex_y = vertex_coords[:, 1].copy()
        in_candidate_idx = np.array(
            [all_roads.index.get_loc(uid) for uid in candidate_roads.index], dtype=np.int32
        )
        logging.debug(f'准备数据时间 {time.time() - _start_time}s')
        _start_time = time.time()
        logging.debug(f'进入c#处理')
        """
        /// <summary>
        /// 批量计算道路相交
        /// </summary>
        /// <param name="addrVerticesX">顶点坐标x的内存地址</param>
        /// <param name="lenVerticesX">顶点坐标x的个数</param>
        /// <param name="addrVerticesY">顶点坐标y的内存地址</param>
        /// <param name="lenVerticesY">顶点坐标y的个数</param>
        /// <param name="addrFirst">记录了polyline起始点的index序号的array的内存地址</param>
        /// <param name="lenFirst">记录了polyline起始点的index序号的array的长度， 即polyline的个数</param>
        /// <param name="addrNumVerticesPerPolyline">记录了每个polyline顶点个数的array的内存地址</param>
        /// <param name="lenNumVerticesPerPolyline">记录了每个polyline顶点个数的array的长度， 即polyline的个数</param>
        /// <param name="addrIdxToCalculate">记录了哪些idx的polyline需要参与交点计算的array的内存地址</param>
        /// <param name="lenIdxToCalculate">记录了哪些idx的polyline需要参与交点计算的array的长度， 即需要计算的polyline的个数</param>
        ///  returns:
        ///   idx1    | idx2   | x      | y
        ///   int32   | int32  | float32| float32
        ///   4 bytes | 4 bytes| 4 bytes| 4 bytes
        """
        c_params = arrs_addr_len(in_vertex_x, in_vertex_y, in_first, in_num, in_candidate_idx)
        buffer = cRoadAccelerator.RoadIntersection(*c_params)
        logging.debug(f'c# 处理数据总耗时(含数据传输和解析) {time.time() - _start_time}s')

        data = np.frombuffer(bytes(buffer), dtype=np.int32).reshape(-1, 4)  # idx1, idx2, pt_x, pt_y
        idx_data = data[:, :2].astype(np.int32)
        intersection_data = data[:, 2:].view(np.float32)
        out_valid_uid_data: list[tuple[uuid.UUID, uuid.UUID]] = []
        out_valid_intersection_data: list[tuple[float, float]] = []
        # 去重，去正常相交
        calculated_idx_pair = set()
        for i in range(len(idx_data)):
            idx1 = int(idx_data[i][0])
            idx2 = int(idx_data[i][1])
            if not (idx1, idx2) in calculated_idx_pair:
                calculated_idx_pair.add((idx1, idx2))
                calculated_idx_pair.add((idx2, idx1))
            else:
                continue  # 去除已经计算过的道路
            road1, road2 = all_roads.iloc[idx1], all_roads.iloc[idx2]
            uid1, uid2 = road1['uid'], road2['uid']
            uid_set = {road1['u'], road1['v'], road2['u'], road2['v']}
            if len(uid_set) <= 3: continue  # 去除掉本身就首尾相接的道路
            out_valid_uid_data.append((uid1, uid2))
            out_valid_intersection_data.append(tuple(intersection_data[i]))
        # 完成去重
        return out_valid_uid_data, out_valid_intersection_data

    @staticmethod
    def build_STRTree(roads) -> int:
        in_first = np.empty(len(roads), dtype=np.int32)
        in_num = np.empty(len(roads), dtype=np.int32)
        vertex_coords = roads['coords']
        i = 0
        total = 0
        for coords in vertex_coords:
            num = len(coords)
            in_first[i] = total
            in_num[i] = num
            total += num
            i += 1
        vertex_coords = np.concatenate(vertex_coords.values, dtype=np.float32, axis=0)
        in_vertex_x = vertex_coords[:, 0].copy()
        in_vertex_y = vertex_coords[:, 1].copy()

        c_params = arrs_addr_len(in_vertex_x, in_vertex_y, in_first, in_num)
        key: int = cRoadAccelerator.BuildSTRTree(*c_params)
        return key

    @staticmethod
    def remove_STRTree(key):
        cRoadAccelerator.RemoveSTRTree(key)

    @staticmethod
    def detect_intersection_fast(STRTree_key, candidate_road: pd.Series):
        """只检查最后一段线段"""
        coords = candidate_road['coords']  # shape: [n, 2]
        if len(coords) < 3:
            return False
        in_lst_element = np.array(coords[-2:].flatten(), dtype=np.float32)  # shape: [2, 2]
        seg_addr, seg_len = arrs_addr_len(in_lst_element)
        # seg_addr = Int64(seg_addr)
        # seg_len = Int64(seg_len)
        intersect = cRoadAccelerator.RoadIntersectionFast(STRTree_key, seg_addr, seg_len)
        return intersect

    @staticmethod
    def _UidUidPoint_to_UidPoints(road_uid_data, intersection_data):
        # 转置为每条道路上的分割点
        uid_points_dict: dict[uuid.UUID:set[tuple[float, float]]] = {}  # {uuid, set(intersection points)} 记录每条道路上的分割点
        for i in range(len(road_uid_data)):
            uid1, uid2 = road_uid_data[i][0], road_uid_data[i][1]
            intersection_point = intersection_data[i]  # tuple
            if uid1 not in uid_points_dict.keys():
                uid_points_dict[uid1] = set()
            if uid2 not in uid_points_dict.keys():
                uid_points_dict[uid2] = set()
            uid_points_dict[uid1].add(intersection_point)
            uid_points_dict[uid2].add(intersection_point)
        # 至此得到了uid道路的分割点位置的dict，  key为uid, value为set[tuple(float, float)]
        return uid_points_dict

    @staticmethod
    def interpolate_road(road, distance, normalized) -> Optional[np.ndarray]:
        """在road上取点， 失败返回None"""
        uid = road['uid']
        geo = road['geometry']
        if not isinstance(geo, LineString):
            logging.warning(f"Road geometry does not contain any LineString. "
                            f"This road ({uid}) will be skipped")
            return None
        cut_point: Point = geo.interpolate(distance, normalized)
        if cut_point.is_empty:
            logging.warning(f"Cut failed. Cut_point is empty. Please check your road geometry."
                            f"This road ({uid}) will be skipped")
            return None
        coord = np.array(cut_point.coords)[0]  # shape= (2, )
        return coord

    def interpolate_road_by_random_position(self, road: pd.Series):
        dist_min, dist_max = self.get_valid_spawn_range(road)
        if dist_min is None or dist_max is None:
            return None
        dist = random.uniform(dist_min, dist_max)
        return self.interpolate_road(road, dist, normalized=False)

    def interpolate_road_by_fixed_interval(self, road: pd.Series, interval: float):
        dist_min, dist_max = self.get_valid_spawn_range(road)
        if dist_min is None or dist_max is None:
            return None

        # 计算距离范围内的所有插值点
        distances = np.arange(dist_min, dist_max, interval)  # 每隔 interval 生成一个距离
        if len(distances) == 0:
            return None  # 如果没有有效距离，返回 None

        # 选择第一个插值点作为 spawn_point
        selected_distance = distances[0]
        return self.interpolate_road(road, selected_distance, normalized=False)

    def identification_of_cutoff_roads(self):
        return
    def find_end_points_cutoff_roads(self):
        return

    def split_road(self, road: pd.Series, distance: float, normalized: bool) -> list[uuid.uuid4()]:
        """先取点，后分割，返回分割后的list[uid]， 失败返回自身uid"""
        assert isinstance(road, pd.Series)
        coord = self.interpolate_road(road, distance, normalized)  # (2, )
        if coord is None: return [road['uid']]
        return self.split_road_by_coord(road, coord)

    def split_road_by_coord(self, road: pd.Series, split_coord: np.ndarray) -> list[uuid.uuid4()]:
        """通过坐标点分割道路， 返回分割后的list[uid]， 失败返回自身uid"""
        split_coord = split_coord.reshape(2)
        uid = self._get_node_by_coord(split_coord)
        if road['u'] == uid or road['v'] == uid:
            return [road['uid']]
        split_coord = split_coord.reshape(-1, 2)
        return self.split_road_by_coords(road, split_coord)

    def split_road_by_coords(self, road: pd.Series, split_coords: np.ndarray) -> list[uuid.uuid4()]:
        """通过一系列坐标点分割道路，返回分割后的list[uid]， 失败返回自身uid
        该方法调用c#实现"""
        assert isinstance(split_coords, np.ndarray)
        split_coords = split_coords.reshape(-1, 2)  # (n, 2)
        road_coords = road['coords']
        road_coords_x = road_coords[:, 0].astype(np.float32).copy()
        road_coords_y = road_coords[:, 1].astype(np.float32).copy()
        split_coords_x = split_coords[:, 0].astype(np.float32).copy()
        split_coords_y = split_coords[:, 1].astype(np.float32).copy()
        # c# code
        c_params = arrs_addr_len(road_coords_x, road_coords_y, split_coords_x, split_coords_y)
        result = cRoadAccelerator.SplitRoad(*c_params)

        new_road_coords = np.frombuffer(bytes(result.Item1), dtype=np.float32).reshape(-1, 2)
        new_road_coords_num = np.frombuffer(bytes(result.Item2), dtype=np.int32)
        segmented_road_coords = np.split(new_road_coords, np.cumsum(new_road_coords_num)[:-1])
        org_level = road['level']
        org_state = road['state']
        self.delete_road(road)
        new_uids = []
        for road_coords in segmented_road_coords:
            uid = self.add_road_by_coords(road_coords, org_level, org_state)
            new_uids.append(uid)
        return new_uids

    def split_roads_by_intersection_data(self, road_uid_data: list[tuple[uuid.UUID, uuid.UUID]],
                                         intersection_data: list[tuple[float, float]]):
        uid_point_dict = self._UidUidPoint_to_UidPoints(road_uid_data, intersection_data)
        for uid, intersection_points in uid_point_dict.items():
            logging.debug(f'正在分割road {uid}, 上面有{len(intersection_points)}个点')
            road = self.get_road_by_uid(uid)
            intersection_points = list(intersection_points)  # list[tuple(float, float)]
            intersection_points = np.array(intersection_points).reshape(-1, 2)  # (n, 2)
            self.split_road_by_coords(road, intersection_points)

    def merge_roads(self, road1: pd.Series, road2: pd.Series, update_nodes_immediately=True):
        assert isinstance(road1, pd.Series)
        assert isinstance(road2, pd.Series)
        if road1['level'] != road2['level']:
            logging.debug('[!]不同level的道路无法合并')
            return
        if road1['state'] != road2['state']:
            logging.debug('[!]不同state的道路无法合并')
            return
        coords1 = road1['coords']
        coords2 = road2['coords']
        if len(coords1) <= 1 or len(coords2) <= 1:
            logging.debug('只有一个点的道路暂不支持合并')
            return
        node_uids = {road1['u'], road1['v'], road2['u'], road2['v']}
        if len(node_uids) >= 4:
            logging.debug('没有找到共享的node， 无法合并')
            return
        if len(node_uids) <= 2:
            logging.debug('道路完全重合')
            self.delete_road(road2, update_nodes_immediately=update_nodes_immediately)
            return road1
        if road1['u'] == road2['u']:
            # (<v---coords1---u) (u---coords2---v>)
            coords1 = coords1[::-1, :]  # 翻转
        elif road1['u'] == road2['v']:
            # (<v---coords1---u) (<v---coords2---u)
            coords1 = coords1[::-1, :]  # 翻转
            coords2 = coords2[::-1, :]  # 翻转
        elif road1['v'] == road2['u']:
            # (u---coords1---v>) (u---coords2---v>)
            pass
        elif road1['v'] == road2['v']:
            # (u---coords1---v>) (<v---coords2---u)
            coords2 = coords2[::-1, :]  # 翻转
        else:
            raise Exception
        new_coords = np.vstack((coords1[:-1, :], coords2))
        self.add_road_by_coords(new_coords, road1['level'], road1['state'])
        self.delete_road(road1, update_nodes_immediately=update_nodes_immediately)
        self.delete_road(road2, update_nodes_immediately=update_nodes_immediately)

    def simplify_roads(self):
        org_nodes = self.__node_gdf.copy()  # 这里必须创建一份副本，否则对node 的删改会改变列表
        org_node_count = len(org_nodes)
        for node_uid, node in org_nodes.iterrows():
            roads = self.get_roads_by_node(node)
            if len(roads) == 2:
                self.merge_roads(roads.iloc[0], roads.iloc[1])
        curr_node_count = len(self.__node_gdf)
        logging.info(
            f'simplify_roads 结果：{org_node_count - curr_node_count}节点被优化, 优化后节点数: {curr_node_count}')

    def examine_invalid_roads(self) -> list[uuid.UUID]:
        invalids_road_uids = []
        for road_uid, road in self.__edge_gdf.iterrows():
            road_uid: uuid.UUID = road_uid
            geo = road['geometry']
            if not isinstance(geo, LineString):
                invalids_road_uids.append(road_uid)
                continue
            length = geo.length
            if length < 1.0 / PRECISION * 2:
                invalids_road_uids.append(road_uid)
                continue
        return invalids_road_uids

    def auto_fix(self):
        logging.info('正在执行自动修复程序')
        # merge nodes
        self.merge_all_close_nodes()
        self.delete_roads_by_uids_list(self.examine_invalid_roads())
        count = 0
        while True:
            a, b = self.detect_intersection(self.get_all_roads(), self.get_all_roads())
            if len(a) == 0: break  # no intersection
            self.split_roads_by_intersection_data(a, b)
            self.merge_all_close_nodes()
            self.delete_roads_by_uids_list(self.examine_invalid_roads())
            count += 1
            if count > 10:
                logging.warning('达到最大清理循环上限，依旧检测到相交道路，请手动检查')
                break
        self.simplify_roads()
        logging.info(f"自动修复完成, 共计{count}轮")

    # endregion

    # region 绘图相关

    @staticmethod
    def get_vertices_data_legacy(roads, style_factory):
        # time logger
        start_time = time.time()
        # colors and width
        params = style_factory(roads)  # tuple( colors and widths)
        in_colors = np.array(params[0], dtype=np.float32)  # float32
        in_widths = np.array(params[1], dtype=np.float32) * g.LINE_WIDTH_SCALE  # width multiplier  float32
        if in_colors.shape[1] == 3:
            in_colors = np.concatenate((in_colors, np.ones((len(in_colors), 1), dtype=np.float32)), axis=1)
        # coords, first and num
        vertex_coords = roads['coords']  # pandas
        in_first = np.concatenate(([0], np.cumsum(vertex_coords.apply(len).to_numpy()[:-1]))).astype(np.int32)  # int32
        in_num = vertex_coords.apply(len).to_numpy().astype(np.int32)  # int32
        in_vertex_coords = np.concatenate(vertex_coords.values, axis=0).astype(np.int32)  # (n, 2) int32

        # convert to bytes
        in_vertex_coords = in_vertex_coords.tobytes()  # 4 + 4 bytes
        in_first = in_first.tobytes()  # 4 byte
        in_num = in_num.tobytes()  # 4 bytes
        in_colors = in_colors.tobytes()  # 4 + 4 + 4 + 4  bytes
        in_widths = in_widths.tobytes()  # 4 bytes

        # time logger
        logging.debug(f'prepare bytes 消耗时间 = {time.time() - start_time}s')
        start_time = time.time()

        # c# code
        buffer = cAccelerator.TriangulatePolylines(in_vertex_coords, in_first, in_num, in_colors, in_widths)

        # time logger
        logging.debug(f'c# 代码消耗时间 = {time.time() - start_time}s')
        start_time = time.time()

        # convert to numpy
        vertices = np.frombuffer(bytes(buffer), np.float32).reshape(-1, 6)
        if np.any(np.isnan(vertices)):
            logging.warning("There are NaN values in the vertices array.")

        # time logger
        logging.debug(f'转换为numpy消耗时间 = {time.time() - start_time}s')
        return vertices

    @staticmethod
    def get_vertices_data(roads, style_factory):
        # print(f'getting road vertices data ({len(roads)}) {inspect.stack()}')
        # time logger
        start_time = time.time()
        # colors and width
        params = style_factory(roads)  # tuple( colors and widths)
        colors = np.array(params[0], dtype=np.float32)  # float32
        in_widths = np.array(params[1], dtype=np.float32) * g.LINE_WIDTH_SCALE  # width multiplier  float32
        if colors.shape[1] == 3:
            colors = np.concatenate((colors, np.ones((len(colors), 1), dtype=np.float32)), axis=1)
        in_r, in_g, in_b, in_a = colors[:, 0].copy(), colors[:, 1].copy(), colors[:, 2].copy(), colors[:, 3].copy()

        # coords, first and num
        vertex_coords = roads['coords']  # pandas
        in_first = np.concatenate(([0], np.cumsum(vertex_coords.apply(len).to_numpy()[:-1])), dtype=np.int32)  # int32
        in_num = vertex_coords.apply(len).to_numpy().astype(np.int32)  # int32
        vertex_coords = np.concatenate(vertex_coords.values, axis=0).astype(np.float32)  # (n, 2) float32
        in_x, in_y = vertex_coords[:, 0].copy(), vertex_coords[:, 1].copy()
        # time logger
        logging.debug(f'prepare bytes 消耗时间 = {time.time() - start_time}s')
        start_time = time.time()

        # c# code
        c_params = arrs_addr_len(in_x, in_y, in_first, in_num, in_r, in_g, in_b, in_a, in_widths)
        buffer = cAccelerator.TriangulatePolylines(*c_params)

        # time logger
        logging.debug(f'c# 代码消耗时间 = {time.time() - start_time}s')
        start_time = time.time()

        # convert to numpy
        vertices = np.frombuffer(bytes(buffer), np.float32).reshape(-1, 6)
        if np.any(np.isnan(vertices)):
            logging.warning("There are NaN values in the vertices array.")

        # time logger
        logging.debug(f'转换为numpy消耗时间 = {time.time() - start_time}s')
        return vertices

    @staticmethod
    def get_node_vertices_data_legacy(nodes: gpd.GeoDataFrame, style_factory):
        params = style_factory(nodes)
        colors = params[0]
        widths = params[1]
        vertex_coords = np.concatenate(nodes['coord'].apply(np.array).values, axis=0).astype(np.float32)
        print(vertex_coords.shape)
        vertex_coords = vertex_coords.tobytes()  # 4 + 4 bytes
        colors = np.array(colors, dtype=np.float32)
        if colors.shape[1] == 3:
            colors = np.concatenate((colors, np.ones((len(colors), 1), dtype=np.float32)), axis=1)
        colors = colors.tobytes()  # 4 + 4 + 4 + 4  bytes
        widths = np.array(widths, dtype=np.float32) * g.LINE_WIDTH_SCALE  # width multiplier
        widths = widths.tobytes()  # 4 bytes
        buffer = cAccelerator.TriangulatePoints(vertex_coords, colors, widths)
        py_bytes = bytes(buffer)
        vertices = np.frombuffer(py_bytes, np.float32).reshape(-1, 6)
        return vertices

    @staticmethod
    def get_node_vertices_data(nodes: gpd.GeoDataFrame, style_factory, road_collection=None):
        try:
            params = style_factory(nodes, road_collection=road_collection)
            in_widths = np.array(params[1], dtype=np.float32) * g.LINE_WIDTH_SCALE  # width multiplier
            colors = np.array(params[0], dtype=np.float32)
            if colors.shape[1] == 3:
                colors = np.concatenate((colors, np.ones((len(colors), 1), dtype=np.float32)), axis=1)
            in_r, in_g, in_b, in_a = colors[:, 0].copy(), colors[:, 1].copy(), colors[:, 2].copy(), colors[:, 3].copy()
            vertex_coords = np.concatenate(nodes['coord'].apply(np.array).values, axis=0).astype(np.float32).reshape(-1,
                                                                                                                     2)
            in_x, in_y = vertex_coords[:, 0].copy(), vertex_coords[:, 1].copy()

            # c# code
            c_params = arrs_addr_len(in_x, in_y, in_r, in_g, in_b, in_a, in_widths)
            buffer = cAccelerator.TriangulatePoints(*c_params)

            vertices = np.frombuffer(bytes(buffer), np.float32).reshape(-1, 6)
            return vertices
        except Exception as e:
            print(nodes.columns)
            raise e

    # endregion

    # region 类型转换

    def to_graph(self, edge_df=None, node_df=None, use_cache=True):
        """
        fast implement, version 3
        此版特性包括：
        将计算内容转移到道路的创建与修改过程中，道路将维护u和v的变量，因此创建graph时无需重新计算，大大加速了graph的创建速度
        引入缓存机制，缓存过的graph部分将不再重新创建
        """
        start_time = time.time()
        # Give initial value when value is None
        if edge_df is None:
            edge_df = self.__edge_gdf
        if node_df is None:
            node_df = self.__node_gdf
        if self.__cached_graph is None and use_cache:
            use_cache = False
        # handle cache
        # if use_cache:
        #     if self._flag_cached_graph_need_update:
        #         logging.warning("You have operated on objects in the cache in previous operations without updating "
        #                         "the cache. This is not allowed and may cause errors.")
        #     G = self.__cached_graph  # use cached graph
        #     if 'cache' not in edge_df.columns:
        #         edge_df['cache'] = False
        #     if 'cache' not in node_df.columns:
        #         node_df['cache'] = False
        #     edge_df = edge_df[~edge_df['cache']]  # filter non-cached edges
        #     node_df = node_df[~node_df['cache']]  # filter non-cached nodes
        else:
            G = nx.Graph()
        # add nodes first
        for index, row in node_df.iterrows():
            uid = row['uid']
            x, y = row['coord']
            geometry = row['geometry']
            G.add_node(uid,
                       x=x,
                       y=y,
                       geometry=geometry)
        # add edges
        for index, row in edge_df.iterrows():
            u = row['u']
            v = row['v']
            uid = row['uid']
            geometry = row['geometry']
            coords = row['coords']
            state = row['state']
            level = row['level']
            length = 0
            speed = road_utils.speed_by_road_level[level]
            for i in range(len(coords) - 1):
                length += np.linalg.norm(coords[i] - coords[i + 1])
            G.add_edge(u, v,
                       length=length,
                       time=length/speed,
                       uid=uid,
                       geometry=geometry,
                       coords=coords,
                       level=level,
                       state=state)
        end_time = time.time()
        print(f"roads_to_graph 转换耗时 {(end_time - start_time) * 1000} ms")
        return G

    def from_graph(self, G):
        """
        仅使用原有的edges数据进行创建，
        highway数据将自动转化为对应的level，geometry数据将会保留
        u，v信息将被抛弃，节点信息由geometry自动计算得到
        该方法速度较慢
        """
        self.delete_all()
        coords_list = []
        level_list = []
        state_list = []
        for u, v, data in G.edges.data():
            if 'geometry' not in data:
                # logging.warning('No coords in data!')
                continue

            if 'level' in data:
                level = data['level']
            elif 'highway' in data:
                level = road_utils.highway_to_level(highway=data['highway'])
            else:
                level = RoadLevel.UNDEFINED
            level_list.append(level)

            if 'state' in data:
                state = data['state']
            else:
                state = RoadState.RAW
            coords = list(data['geometry'].coords)
            state_list.append(state)
            coords_list.append(np.array(coords))
        self.add_roads_by_coords(coords_list, level_list, state_list)

    @timer
    def data_to_roads(self, data: dict):
        assert 'roads' in data, 'invalid data'
        self.delete_all()
        roads_data = data['roads']
        assert isinstance(roads_data, list)
        # print(f"共有{len(roads_data)}条道路数据")
        points_list = []
        level_list = []
        state_list = []
        for i in range(len(roads_data)):
            road_data = roads_data[i]
            points_list.append(np.array(road_data['points']))
            level_list.append(road_data['level'])
            state_list.append(road_data['state'])

        uid_list = self.add_roads_by_coords(points_list, level_list, state_list)

        return uid_list

    def roads_to_data(self, out_data: dict):
        if 'roads' not in out_data:
            out_data['roads'] = []
        for uid, road in self.get_all_roads().iterrows():
            road_data = {
                'points': road['coords'],
                'level': road['level'],
                'state': road['state']
            }
            out_data['roads'].append(road_data)

    # endregion

    # region 其他工具

    def quick_roads(self):
        points = np.array([
            [0, 0],
            [0, 100],
            [-20, 20],
            [120, 20],
            [0, 20]
        ])
        uid1 = self.add_road_by_coords(np.array([points[0], points[4]]), RoadLevel.TRUNK, RoadState.RAW)
        uid2 = self.add_road_by_coords(np.array([points[4], points[1]]), RoadLevel.TRUNK, RoadState.RAW)
        uid3 = self.add_road_by_coords(np.array([points[2], points[4]]), RoadLevel.SECONDARY, RoadState.RAW)
        uid4 = self.add_road_by_coords(np.array([points[4], points[3]]), RoadLevel.SECONDARY, RoadState.RAW)
        return [uid1, uid2, uid3, uid4]

    def show_info(self):
        print(f"==================================================")
        print(f"道路数量: {len(self.__edge_gdf)}")
        print(f"节点数量: {len(self.__node_gdf)}")
        print(f"coord_to_uid字典长度 {len(list(self.__coord_to_node_uid.keys()))}")
        print(f"cached graph: {self.__cached_graph is not None}")
        print(f"cached edge gdf: {self.__cached_edge_gdf is not None}")
        print(f"cached node gdf: {self.__cached_node_gdf is not None}")
        print(f"==================================================")

    def copy(self):
        new = RoadCollection()
        new.__node_gdf = self.__node_gdf.copy(deep=True)
        new.__edge_gdf = self.__edge_gdf.copy(deep=True)
        new.__coord_to_node_uid = self.__coord_to_node_uid.copy()
        # new.__cached_graph = copy.deepcopy(self.__cached_graph)
        new.__cached_node_gdf = self.__cached_node_gdf.copy(deep=True) if self.__cached_node_gdf is not None else None
        new.__cached_edge_gdf = self.__cached_edge_gdf.copy(deep=True) if self.__cached_edge_gdf is not None else None
        new._flag_cached_graph_need_update = self._flag_cached_graph_need_update
        new.__uid = uuid.uuid4()  # 或者你可以保留旧 UID，看你语义需求
        return new

    def cache(self, roads=None):
        if roads is None:
            roads = self.__edge_gdf
            nodes = self.__node_gdf
        else:
            nodes = self._get_nodes_by_roads(roads)
        self.clear_cache()
        # self.__cached_graph = self.to_graph(edge_df=roads, node_df=nodes, use_cache=False)
        self.__cached_node_gdf = gpd.GeoDataFrame(columns=self.__node_gdf.columns,
                                                  data=copy.deepcopy(self.__node_gdf.values),
                                                  index=self.__node_gdf.index)  # deep copy
        self.__cached_edge_gdf = gpd.GeoDataFrame(columns=self.__edge_gdf.columns,
                                                  data=copy.deepcopy(self.__edge_gdf.values),
                                                  index=self.__edge_gdf.index)  # deep copy
        self._flag_cached_graph_need_update = False  # update dirty variable
        for index, row in roads.iterrows():
            roads.loc[index, 'cache'] = True
        for index, row in nodes.iterrows():
            nodes.loc[index, 'cache'] = True

    def clear_cache(self):
        self.__edge_gdf['cache'] = False
        self.__node_gdf['cache'] = False

        # self.__cached_graph = None
        self.__cached_edge_gdf = None
        self.__cached_node_gdf = None

    def restore(self):
        if self.__cached_edge_gdf is not None and self.__cached_node_gdf is not None:
            self.__node_gdf = gpd.GeoDataFrame(columns=self.__cached_node_gdf.columns,
                                               data=copy.deepcopy(self.__cached_node_gdf.values),
                                               index=self.__cached_node_gdf.index)  # deep copy

            self.__edge_gdf = gpd.GeoDataFrame(columns=self.__cached_edge_gdf.columns,
                                               data=copy.deepcopy(self.__cached_edge_gdf.values),
                                               index=self.__cached_edge_gdf.index)  # deep copy
            self._flag_cached_graph_need_update = False
            self.rebuild_coord_to_uid_dict()  # 这里这么做是因为偷懒没有在cache的时候保存coord_to_uid的dict的副本， 因此当node gdf改变时需要更新一下
        else:
            logging.warning("[Road] no cache to restore")
        self.__uid = uuid.uuid4()

    # endregion


if __name__ == "__main__":
    pass
