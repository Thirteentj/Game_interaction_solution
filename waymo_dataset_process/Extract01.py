import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from waymo_open_dataset.protos.scenario_pb2 import Scenario
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from shapely.geometry import LineString

#本文件包含静态地图信息（包括车道线、交叉口信息）提取的子函数

def get_file_list(filepath):
    all_files = sorted(glob.glob(filepath))
    segs_name_index = []
    for file in all_files:
        segment_name = os.path.basename(file)
        segs_name_index.append(segment_name[-14:-9])
    # print(segs_name_all)
    # print(segs_name_index)
    return all_files, segs_name_index

global point_has_been_pointed
point_has_been_pointed = []

def plot_top_view_single_pic_map(trj_in, file_index, scenario_id_in, scenario, target_left, target_right,
                                 loc_target_intersection, intersection_range, filepath,lane_turn_right_id_real=[]):  #包含交叉口识别结果
    global point_has_been_pointed
    # plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 20),
               fontsize=5)
    plt.yticks(np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 20),
               fontsize=5)
    # ax = plt.subplots(121)
    map_features = scenario.map_features
    road_edge_count = 0
    lane_count = 0
    road_line = 0
    all_element_count = 0
    for single_feature in map_features:
        all_element_count += 1
        id_ = single_feature.id
        # print("id is %d"%id_)
        if list(single_feature.road_edge.polyline) != []:
            road_edge_count += 1
            single_line_x = []
            single_line_y = []
            # print("road_edge id is %d"%single_feature.id)
            for polyline in single_feature.road_edge.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linewidth=0.3)  # 道路边界为黑色

        if list(single_feature.lane.polyline) != []:
            lane_count += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.lane.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            # z1 = np.polyfit(single_line_x,single_line_y,8)
            # p1 = np.poly1d(z1)
            # y_hat = p1(single_line_x)
            # ax.plot(single_line_x,y_hat,color='green', linewidth=0.5)
            if id_ in target_left:
                ax[0].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
                ax[1].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
            elif id_ in target_right:
                if id_ in lane_turn_right_id_real:  # 目标交叉口的右转车道
                    ax[0].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                else:
                    ax[0].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
            else:
                ax[0].plot(single_line_x, single_line_y, color='blue', linewidth=0.5)  # 道路中心线为蓝色
            if (single_line_x[0], single_line_y[0]) not in point_has_been_pointed:
                ax[0].text(single_line_x[0], single_line_y[0], id_, fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0], single_line_y[0]))
            else:
                ax[0].text(single_line_x[0] - 5, single_line_y[0] - 5, id_, color='red', fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0] - 5, single_line_y[0] - 5))

        if list(single_feature.road_line.polyline) != []:
            road_line += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.road_line.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linestyle='-', linewidth=0.3)  # 道路标线为  虚线
    ax[0].plot(loc_target_intersection[0],loc_target_intersection[1],'om',markersize=1.5)
    range_x, range_y = intersection_range[0], intersection_range[1]
    loc_target_intersection = (loc_target_intersection[0] - range_x / 2, loc_target_intersection[1] - range_y / 2)
    # print(loc_target_intersection)
    ax[0].add_patch(patches.Rectangle(
        xy=loc_target_intersection,
        width=range_x,
        height=range_y,
        facecolor='none',
        linewidth=0.8,
        edgecolor='black'))
    #filepath = 'figure_save/intersection_topo_figure/'
    fig_save_name = filepath + 'top_view_segment_'  + str(
        file_index) + '_scenario_' + str(
        scenario_id_in) + '_trajectory.jpg'
    plt.savefig(fig_save_name, dpi=600)
    # plt.show()
    plt.close('all')
    return road_edge_count, lane_count, road_line, all_element_count

def plot_top_view_single_pic_map_2(trj_in,file_index,scenario_id_in,scenario,target_left, target_right,filepath,lane_turn_right_id_real=[]):  #仅仅是识别后的原始地图场景
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 20),
               fontsize=5)
    plt.yticks(np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 20),
               fontsize=5)
    # ax = plt.subplots(121)
    map_features = scenario.map_features
    road_edge_count = 0
    lane_count = 0
    road_line = 0
    all_element_count = 0
    for single_feature in map_features:
        all_element_count += 1
        id_ = single_feature.id
        # print("id is %d"%id_)
        if list(single_feature.road_edge.polyline) != []:
            road_edge_count += 1
            single_line_x = []
            single_line_y = []
            # print("road_edge id is %d"%single_feature.id)
            for polyline in single_feature.road_edge.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linewidth=0.3)  # 道路边界为黑色

        if list(single_feature.lane.polyline) != []:
            lane_count += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.lane.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            # z1 = np.polyfit(single_line_x,single_line_y,8)
            # p1 = np.poly1d(z1)
            # y_hat = p1(single_line_x)
            # ax.plot(single_line_x,y_hat,color='green', linewidth=0.5)
            if id_ in target_left:
                ax[0].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
                ax[1].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
            elif id_ in target_right:
                if id_ in lane_turn_right_id_real:  # 目标交叉口的右转车道
                    ax[0].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                else:
                    ax[0].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
            else:
                ax[0].plot(single_line_x, single_line_y, color='blue', linewidth=0.5)  # 道路中心线为蓝色
            if (single_line_x[0], single_line_y[0]) not in point_has_been_pointed:
                ax[0].text(single_line_x[0], single_line_y[0], id_, fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0], single_line_y[0]))
            else:
                ax[0].text(single_line_x[0] - 5, single_line_y[0] - 5, id_, color='red', fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0] - 5, single_line_y[0] - 5))

        if list(single_feature.road_line.polyline) != []:
            road_line += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.road_line.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linestyle='-', linewidth=0.3)  # 道路标线为  虚线


    # filepath = 'figure_save/intersection_topo_figure/'
    fig_save_name = filepath + 'top_view_segment_' + str(
        file_index) + '_scenario_' + str(
        scenario_id_in) + '_trajectory.jpg'
    plt.savefig(fig_save_name, dpi=600)
    # plt.show()
    plt.close('all')
    return road_edge_count, lane_count, road_line, all_element_count

def plot_top_view_single_pic_map_3(trj_in,file_index,scenario_id_in, scenario,target_left,target_right,intersection_range,trafficlight_lane,lane_turn_right_id_real=[]):  #带有动态地图信息
    global point_has_been_pointed
    #plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots(1,2,figsize=(14,7))
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 20),fontsize=5)
    plt.yticks(np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 20), fontsize=5)
    #ax = plt.subplots(121)
    map_features = scenario.map_features
    road_edge_count = 0
    lane_count = 0
    road_line = 0
    all_element_count = 0
    for single_feature in map_features:
        all_element_count += 1
        id_ = single_feature.id
        #print("id is %d"%id_)
        if list(single_feature.road_edge.polyline)!= []:
            road_edge_count += 1
            single_line_x = []
            single_line_y = []
            # print("road_edge id is %d"%single_feature.id)
            for polyline in single_feature.road_edge.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linewidth=0.3)  # 道路边界为黑色

        if list(single_feature.lane.polyline)!= []:
            lane_count += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.lane.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            #z1 = np.polyfit(single_line_x,single_line_y,8)
            #p1 = np.poly1d(z1)
            #y_hat = p1(single_line_x)
            #ax.plot(single_line_x,y_hat,color='green', linewidth=0.5)
            if id_ in target_left:
                ax[0].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
                ax[1].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
            elif id_ in target_right:
                if id_ in lane_turn_right_id_real:  #目标交叉口的右转车道
                    ax[0].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                else:
                    ax[0].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)  #deeppink
            elif id_ in trafficlight_lane:  #有信号灯数据的车道
                ax[0].plot(single_line_x, single_line_y, color='deeppink', linewidth=0.5)
                ax[1].plot(single_line_x, single_line_y, color='deeppink', linewidth=0.5)

            else:
                ax[0].plot(single_line_x, single_line_y, color='blue', linewidth=0.5)  # 道路中心线为蓝色
            if (single_line_x[0],single_line_y[0]) not in point_has_been_pointed:
                ax[0].text(single_line_x[0], single_line_y[0], id_, fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0],single_line_y[0]))
            else:
                ax[0].text(single_line_x[0]-5, single_line_y[0]-5, id_,color='red', fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0]-5, single_line_y[0]-5))

        if list(single_feature.road_line.polyline)!=[]:
            road_line += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.road_line.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linestyle=':', linewidth=0.3)  # 道路标线为  虚线

    #fig_save_name = 'E:/Result_save/figure_save/intersection_topo_figure_test/top_view_segment_' + str(file_index) + '_scenario_' + str(scenario_id_in) + '_trajectory.jpg'
    fig_save_name = 'figure_save/intersection_topo_figure_test/top_view_segment_' + str(
        file_index) + '_scenario_' + str(scenario_id_in) + '_trajectory.jpg'
    #print(fig_save_name)
    plt.savefig(fig_save_name, dpi=600)
    #plt.show()
    plt.close('all')
    return road_edge_count, lane_count, road_line,all_element_count



def get_lane_min_dis(single_scenario_all_feature, map_features_id_list, ego_lane_id, other_lanes, connect_type):
    ego_index = map_features_id_list.index(ego_lane_id)
    ego_lane_info = single_scenario_all_feature[ego_index]
    lane_inter_dis = []  # 用于记录本车道最尽头和目标车道最尽头之间的距离，用于判定是否为交叉口内部
    ego_lane_point = ()
    other_lane_point = []
    for other_lane_id in other_lanes:
        other_lane_index = map_features_id_list.index(other_lane_id)
        other_lane_info = single_scenario_all_feature[other_lane_index]
        if connect_type == 'entry':
            x1, y1 = ego_lane_info.lane.polyline[0].x, ego_lane_info.lane.polyline[0].y
            x2, y2 = other_lane_info.lane.polyline[0].x, other_lane_info.lane.polyline[0].y
            ego_lane_point = (
            ego_lane_info.lane.polyline[0].x, ego_lane_info.lane.polyline[0].y)  # 如果是进入的关系，则返回该车道的第一个点
            other_lane_point.append((other_lane_info.lane.polyline[0].x, other_lane_info.lane.polyline[0].y))
        if connect_type == 'exit':
            x1, y1 = ego_lane_info.lane.polyline[-1].x, ego_lane_info.lane.polyline[-1].y
            x2, y2 = other_lane_info.lane.polyline[-1].x, other_lane_info.lane.polyline[-1].y
            ego_lane_point = (
                ego_lane_info.lane.polyline[-1].x, ego_lane_info.lane.polyline[-1].y)  # 如果是驶出的关系，则返回该车道的最后一个点
            other_lane_point.append((other_lane_info.lane.polyline[-1].x, other_lane_info.lane.polyline[-1].y))

        dis = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        lane_inter_dis.append(dis)
    return lane_inter_dis, ego_lane_point, other_lane_point


def cal_angle(x1, y1, x2, y2):
    if x2 != x1:
        angle = (math.atan((y2 - y1) / (x2 - x1))) * 180 / np.pi
    else:
        angle = 90  # 避免斜率不存在的情况
    return angle


def get_lane_angle_chane(polyline_list, ego_lane_id):  # 计算
    angle_start = 0
    angle_end = 0
    length = len(polyline_list)
    x_list = []
    y_list = []
    turn_type = 'straight'
    x1, y1 = polyline_list[0].x, polyline_list[0].y
    x4, y4 = polyline_list[-1].x, polyline_list[-1].y
    for polyline in polyline_list:
        x_list.append(polyline.x)
        y_list.append(polyline.y)
    try:
        # print(polyline_list)
        x2, y2 = polyline_list[3].x, polyline_list[3].y
        x3, y3 = polyline_list[-3].x, polyline_list[-3].y
        angle_start = cal_angle(x1, y1, x2, y2)
        angle_end = cal_angle(x3, y3, x4, y4)
        delta_angle = angle_end - angle_start  # 大于0为左转，小于0为右转
    except:
        angle_start = angle_end = delta_angle = None

    # 判断左右转信息
    index_mid = int(length / 2)
    x_mid = polyline_list[index_mid].x
    y_mid = polyline_list[index_mid].y
    # p1 = np.array((x_mid-x1,y_mid-y1))
    # p2 = np.array((x4-x_mid,y4-y_mid))
    p3 = (x_mid - x1) * (y4 - y_mid) - (y_mid - y1) * (x4 - x_mid)
    # print(p3)
    if p3 > 0:
        turn_type = 'left'
    elif p3 < 0:
        turn_type = 'right'
    # print("Turn type is %s"%turn_type)
    return angle_start, angle_end, delta_angle, turn_type


def cal_lane_slpoe(polyline):
    x1, y1 = polyline[0].x, polyline[0].y  # 直线起点xy坐标
    x2, y2 = polyline[-1].x, polyline[-1].y  # 直线终点xy坐标
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = 90
    return slope


def map_topo_info_extract(map_features,single_scenario_all_feature,map_features_id_list,file_index,scenario_label):  # 提取车道连接关系等拓扑信息
    single_feature_all_lane_polyline = []  # 一个scenario中所有的车道信息（仅包括散点）
    lane_id_all = []  # 一个scenario中所有车道的ID信息
    all_lane_entry_exit_info = []  # 记录所有车道的进出车道信息
    single_map_dict = {}
    lane_turn_left_id = []  # 记录交叉口内部所有左转车道ID
    lane_turn_right_id = []  # 记录交叉口内部所有右转车道ID

    for single_feature in map_features:  # 先将每根车道线的信息保存成列表
        single_scenario_all_feature.append(single_feature)
        map_features_id_list.append(single_feature.id)

    for single_feature in map_features:
        single_lane_entry_exit_info = {}  # 记录一个车道的进出车道信息，包括与之相邻车道终点与该车道终点的距离（用于和交叉口尺寸阈值进行判定）
        if list(single_feature.lane.polyline) != []:
            ego_lane_id = single_feature.id
            entry_lanes = single_feature.lane.entry_lanes
            exit_lanes = single_feature.lane.exit_lanes
            entry_lanes_dis, ego_lane_point_entry, entry_lane_point = get_lane_min_dis(single_scenario_all_feature,
                                                                                       map_features_id_list,
                                                                                       ego_lane_id, entry_lanes,
                                                                                       'entry')
            exit_lanes_dis, ego_lane_point_exit, exit_lane_point = get_lane_min_dis(single_scenario_all_feature,
                                                                                    map_features_id_list, ego_lane_id,
                                                                                    exit_lanes, 'exit')
            angle_start, angle_end, delta_angle, turn_type = get_lane_angle_chane(single_feature.lane.polyline,
                                                                                  ego_lane_id)  # 该线段的角度变化值
            single_lane_entry_exit_info['file_index'] = file_index
            single_lane_entry_exit_info['scenario_id'] = scenario_label
            single_lane_entry_exit_info['lane_id'] = ego_lane_id
            single_lane_entry_exit_info['angle_start'] = angle_start
            single_lane_entry_exit_info['angle_end'] = angle_end
            single_lane_entry_exit_info['ego_lane_angle_change'] = delta_angle
            single_lane_entry_exit_info['is_a_turn'] = ''
            if delta_angle:
                if 150>abs(delta_angle) > 35:  # 由50暂时修正为35
                    # if 120>delta_angle >50: #为左转
                    # if delta_angle > 50:  # 为左转
                    if turn_type == 'left':
                        single_lane_entry_exit_info['is_a_turn'] = 'left'
                        lane_turn_left_id.append(ego_lane_id)
                    # elif -120<delta_angle <-50:
                    # elif delta_angle < -50:
                    elif turn_type == 'right':
                        single_lane_entry_exit_info['is_a_turn'] = 'right'
                        lane_turn_right_id.append(ego_lane_id)  # 整个交叉口右转车道只有四根
                    else:
                        single_lane_entry_exit_info['is_a_turn'] = 'straight'
                else:
                    single_lane_entry_exit_info['is_a_turn'] = 'straight'

            if single_lane_entry_exit_info['is_a_turn'] == 'straight':
                single_lane_entry_exit_info['lane_slope'] = cal_lane_slpoe(
                    single_feature.lane.polyline)  # 如果是直线车道，计算这条车道的斜率，用于后续计算交叉口角度
            # single_lane_entry_exit_info['is a_plan turn'] = turn_type
            # 相连接车道信息提取
            single_lane_entry_exit_info['ego_lane_point_entry'] = ego_lane_point_entry
            single_lane_entry_exit_info['ego_lane_point_exit'] = ego_lane_point_exit
            single_lane_entry_exit_info['entry_lanes'] = entry_lanes
            single_lane_entry_exit_info['entry_lanes_dis'] = entry_lanes_dis
            single_lane_entry_exit_info['entry_lane_point'] = entry_lane_point
            single_lane_entry_exit_info['exit_lanes'] = exit_lanes
            single_lane_entry_exit_info['exit_lanes_dis'] = exit_lanes_dis
            single_lane_entry_exit_info['exit_lane_point'] = exit_lane_point
            # 相邻车道信息提取
            lane_index_all = len(list(single_feature.lane.polyline))  # 这条路被分成的小片段序列总数（即散点总数）
            single_lane_entry_exit_info['left_neighbors_id'] = -1  # 先初始化
            single_lane_entry_exit_info['right_neighbors_id'] = -1
            if list(single_feature.lane.left_neighbors) != []:  # 左边有车道
                left_neighbors_temp = list(single_feature.lane.left_neighbors)
                flag1 = 0
                for left_neighbor in left_neighbors_temp:
                    # print('index')
                    # print(ego_lane_id)
                    # print(lane_index_all)
                    # print(left_neighbor.self_end_index)
                    if abs(left_neighbor.self_end_index - lane_index_all) < 2:
                        left_neighbors_id = left_neighbor.feature_id  # 记录满足条件的那条左边相邻车道的ID
                        # print("left_neighbors %d" % left_neighbors_id)
                        flag1 = 1
                        break
                if flag1 == 1:
                    single_lane_entry_exit_info['left_neighbors_id'] = left_neighbors_id
                    # print(left_neighbors_id)

            if list(single_feature.lane.right_neighbors) != []:  # 右边右车道
                right_neighbors_temp = list(single_feature.lane.right_neighbors)
                flag2 = 0
                for right_neighbor in right_neighbors_temp:
                    # print('index222')
                    # print(lane_index_all)
                    # print(left_neighbor.self_end_index)
                    if abs(right_neighbor.self_end_index - lane_index_all) < 2:
                        right_neighbors_id = right_neighbor.feature_id
                        # print("right_neighbors %d"%right_neighbors_id)
                        flag2 = 1
                        break
                if flag2 == 1:
                    single_lane_entry_exit_info['right_neighbors_id'] = right_neighbors_id

            # 将一些信息重新记录，便于数据检索和处理
            lane_id_all.append(single_feature.id)
            all_lane_entry_exit_info.append(single_lane_entry_exit_info)
            single_feature_all_lane_polyline.append((single_feature.id, single_feature.lane.polyline))
            single_map_dict[single_feature.id] = single_feature.lane.polyline  # 使用字典进行检索，得到所有车道的坐标点信息
    # print('qqqq')
    # print(single_scenario_all_lane_entry_exit_info)
    # print(single_map_dict,lane_turn_left_id,lane_turn_right_id)
    return all_lane_entry_exit_info, single_map_dict, lane_turn_left_id, lane_turn_right_id,single_scenario_all_feature,map_features_id_list


def intersection_angle_cal(df_all_lan_topo_info):
    from sklearn.cluster import KMeans
    intersection_angle = 0
    slope_list = pd.unique(df_all_lan_topo_info[df_all_lan_topo_info['is_a_turn'] == 'straight']['lane_slope'].tolist())
    # print('slope_list:')
    # print(slope_list)
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(np.array(slope_list).reshape(-1, 1))  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    k1 = np.mean(slope_list[label_pred == 0])  # 这一类斜率的平均值计算
    k2 = np.mean(slope_list[label_pred == 1])
    # print('k1:%f,k2:%f'%(k1,k2))
    intersection_angle = math.atan(abs((k2 - k1) / (1 + k1 * k2))) * 180 / np.pi
    return intersection_angle


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_point_order(points):  # 得到正确的点的连线顺序，以得到正确的多边形
    points_new = []
    points_new_plus = []  # 将交叉口范围稍微扩大,保证范围冗余
    left_point = {}  # 某个点左侧最近的点
    right_point = {}  # 某个点右侧最近的点
    for A in points:
        angle_vec = 0
        for B in points:
            for C in points:
                if (A.x != B.x and A.y != B.y) and (C.x != B.x and C.y != B.y) and (C.x != A.x and C.y != A.y):
                    vec_AB = Point(B.x - A.x, B.y - A.y)
                    vec_AC = Point(C.x - A.x, C.y - A.y)
                    vec_BA = Point(-vec_AB.x, -vec_AB.y)
                    # print(vec_AB.x,vec_AB.y,vec_AC.x,vec_AC.y,A.x,A.y,C.x,C.y)
                    # print(abs(math.sqrt(vec_AB.x**2+vec_AB.y**2)*math.sqrt(vec_AC.x**2+vec_AC.y**2)))
                    cos_angle = (vec_AB.x * vec_AC.x + vec_AB.y * vec_AC.y) / abs(
                        math.sqrt(vec_AB.x ** 2 + vec_AB.y ** 2) * math.sqrt(vec_AC.x ** 2 + vec_AC.y ** 2))
                    # print(cos_angle)
                    angle_vec_temp = abs(math.acos(cos_angle) * 180 / np.pi)
                    # print(angle_vec_temp)
                    if angle_vec_temp > angle_vec:
                        angle_vec = angle_vec_temp
                        # print(angle_vec)
                        p5 = vec_BA.x * vec_AC.y - vec_AC.x * vec_BA.y
                        if p5 > 0:
                            left_point[points.index(A)] = points.index(B)
                            right_point[points.index(A)] = points.index(C)
                        elif p5 < 0:
                            left_point[points.index(A)] = points.index(C)
                            right_point[points.index(A)] = points.index(B)
    x_all = []
    y_all = []
    A = points[0]
    points_new.append((A.x, A.y))
    x_all.append(A.x)
    y_all.append(A.y)
    #points_new_plus.append((A.x + 5, A.y + 5))
    points_new_index = [0]
    for i in range(20):
        A = points[left_point[points.index(A)]]
        points_new.append((A.x, A.y))
        x_all.append(A.x)
        y_all.append(A.y)
        #points_new_plus.append((A.x + 5, A.y + 5))
        if left_point[points.index(A)] == 0:
            break
        points_new_index.append(left_point[points.index(A)])
    #对原坐标围成的多边形范围进行扩充，范围扩大五米
    x_all.sort(reverse=False)
    y_all.sort(reverse=False)
    x_min,x_max = [x_all[0],x_all[1]],[x_all[-1],x_all[-2]]
    y_min,y_max = [y_all[0],y_all[1]],[y_all[-1],y_all[-2]]
    for point in points_new:
        x,y = point[0],point[1]
        if x in x_min:
            x = x -5
        elif x in x_max:
            x = x+5
        if y in y_min:
            y = y+5
        elif y in y_max:
            y = y+5
        points_new_plus.append((x,y))

    return points_new,points_new_plus


def rayCasting(p, poly):  # 判断一个点是否在多边形内部
    px, py = p[0], p[1]
    flag = -1
    i = 0
    l = len(poly)
    j = l - 1
    # for(i = 0, l = poly.length, j = l - 1; i < l; j = i, i++):
    while i < l:
        sx = poly[i][0]
        sy = poly[i][1]
        tx = poly[j][0]
        ty = poly[j][1]
        # 点与多边形顶点重合
        if (sx == px and sy == py) or (tx == px and ty == py):
            flag = 1
        # 判断线段两端点是否在射线两侧
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            # 点在多边形的边上
            if x == px:
                flag = 1
            # 射线穿过多边形的边界
            if x > px:
                flag = -flag
        j = i
        i += 1
    # 射线穿过多边形边界的次数为奇数时点在多边形内
    return flag


def judge_lane_in_intersection(lane_polyline, intersection_range):
    flag = 0  # 如果flag=1 则这条车道在交叉口内部，否则不在
    point_start = (lane_polyline[0].x, lane_polyline[0].y)
    point_end = (lane_polyline[-1].x, lane_polyline[-1].y)
    if (rayCasting(point_start, intersection_range) == 1) and (rayCasting(point_end, intersection_range) == 1):
        flag = 1
    return flag


def get_one_direction_lane_info(right_lane_id, df_all_lane_topo_info, single_map_dict, lane_turn_left_id):
    lane_in_num, lane_out_num = 0, 0  # 这里统计的出口道信息是进口道方向逆时针转90°后方向的出口道
    lane_in_id, lane_out_id = [], []
    entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['entry_lanes'].iloc[0][0]
    exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['exit_lanes'].iloc[0][0]
    #print('entry{},exit{}'.format(entry_lane_id,exit_lane_id))
    lane_entry_direction = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['lane_direction'].values  #右转车道连接的进口道的方向
    lane_exit_direction = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['lane_direction'].values
    #print('right_id{},{},{}'.format(right_lane_id,lane_entry_direction,lane_exit_direction))
    # ----------------进口道信息提取--------------------
    # 车道ID及数量
    # 右转车道右侧的车道一般是自行车车道，暂时不做处理
    lane_in_id.append(entry_lane_id)

    lane_in_left = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['left_neighbors_id']
    while (lane_in_left.tolist()[0] != -1):  # 如果左侧没有车道，则为单车道
        flag = 0
        lane_in_left = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['left_neighbors_id']
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_direction'] = lane_entry_direction
        #print('ttt', df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['lane_direction'].values)
        # print(lane_in_left.tolist())
        # print(lane_in_left.values)
        # print(lane_in_left.tolist()[0] ==-1)
        if (lane_in_left.tolist()[0] == -1):
            break
        lane_in_id.append(lane_in_left.tolist()[0])
        entry_lane_id = lane_in_left.tolist()[0]
        # print(entry_lane_id)
    lane_in_num = len(lane_in_id)
    # -------------------出口道信息提取---------------------------
    # 车道ID及数量

    lane_out_id.append(exit_lane_id)
    lane_out_left = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['left_neighbors_id']

    while (lane_out_left.tolist()[0] != -1):  # 如果左侧没有车道，则为单车道
        flag = 0
        lane_out_left = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['left_neighbors_id']
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_direction'] = lane_exit_direction
        #print('sss', df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['lane_direction'].values)
        if (lane_out_left.tolist()[0] == -1):
            break
        lane_out_id.append(lane_out_left.tolist()[0])
        exit_lane_id = lane_out_left.tolist()[0]
    lane_out_num = len(lane_out_id)

    return lane_in_num, lane_in_id, lane_out_num, lane_out_id, df_all_lane_topo_info,lane_entry_direction,lane_exit_direction


def get_lane_width(df_all_lan_topo_info, single_map_dict, lane_in_num, lane_in_id, lane_out_num, lane_out_id):
    lane_in_width, lane_out_width = 0, 0
    # 计算进口道车道宽度
    # 车道宽度计算
    if lane_in_num > 1:
        width_in_sum = 0
        for i in range(lane_in_num - 1):
            lane_in_cal_1_id = lane_in_id[i]
            lane_in_cal_2_id = lane_in_id[i + 1]
            x1, y1 = single_map_dict[lane_in_cal_1_id][-1].x, single_map_dict[lane_in_cal_1_id][
                -1].y  # 进口道这里是最后一个点，出口道这里应该第1个点
            x2, y2 = single_map_dict[lane_in_cal_2_id][-1].x, single_map_dict[lane_in_cal_2_id][-1].y
            # print("one lane width is %f"%(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
            width_in_sum += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lane_in_width = width_in_sum / (lane_in_num - 1)
        # print("lane width ave = %f"%lane_in_width)
    elif lane_in_num == 1:
        lane_in_width = 3.5  # 这里需要统筹对向出口道才行

    # 计算出口道车道宽度
    if lane_out_num > 1:
        width_out_sum = 0
        for i in range(lane_out_num - 1):
            lane_out_cal_1_id = lane_out_id[i]
            lane_out_cal_2_id = lane_out_id[i + 1]
            x1, y1 = single_map_dict[lane_out_cal_1_id][0].x, single_map_dict[lane_out_cal_1_id][0].y
            x2, y2 = single_map_dict[lane_out_cal_2_id][0].x, single_map_dict[lane_out_cal_2_id][0].y
            width_out_sum += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lane_out_width = width_out_sum / (lane_out_num - 1)
    elif lane_out_num == 1:
        lane_out_width = 3.5
    return lane_in_width, lane_out_width

def get_lane_direction_new(lane_turn_right_id,single_map_dict,df_all_lane_topo_info):
    lane_entry_direction_dict, lane_exit_direction_dict = {}, {}
    entry_exit_link = {}  #右转车道相连的进口道和出口道的连接关系记录
    lane_right_entry = df_all_lane_topo_info[
        (df_all_lane_topo_info['lane_function'] == 'right') & (df_all_lane_topo_info['entry_or_exit'] == 'entry')][
        'lane_id'].tolist()
    lane_right_exit = df_all_lane_topo_info[
        (df_all_lane_topo_info['lane_function'] == 'right') & (df_all_lane_topo_info['entry_or_exit'] == 'exit')][
        'lane_id'].tolist()
    for right_lane_id in lane_turn_right_id:
        entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['entry_lanes'].iloc[0]
        exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['exit_lanes'].iloc[0]
        #print('sss{},{}'.format(entry_lane_id,exit_lane_id))
        #print('sss',right_lane_id,entry_lane_id[0],exit_lane_id[0])
        entry_exit_link[entry_lane_id[0]] = exit_lane_id[0]

    lane_entry_direction_dict = {}  # 对每个车道方向进行标记
    x_all = []
    y_all = []
    # print('list{}'.format(lane_right_entry))
    for lane_entry in lane_right_entry:
        x1, y1 = single_map_dict[lane_entry][-1].x, single_map_dict[lane_entry][-1].y
        x_all.append(x1)
        y_all.append(y1)
    x_all.sort(reverse=False)
    y_all.sort(reverse=False)
    # print('x_all{}'.format(x_all))
    # print('y_all{}'.format(y_all))
    if len(lane_right_entry) == 4:  # 表明有四个方向
        for lane_entry in lane_right_entry:
            try:
                lane_exit = entry_exit_link[lane_entry]
                x1, y1 = single_map_dict[lane_entry][-1].x, single_map_dict[lane_entry][-1].y
                # print('x{},y{}'.format(x1,y1))
                if x1 == min(x_all):
                    lane_entry_direction_dict[lane_entry] = 'W'
                    lane_exit_direction_dict[lane_exit] = 'S'
                elif x1 == max(x_all):
                    lane_entry_direction_dict[lane_entry] = 'E'
                    lane_exit_direction_dict[lane_exit] = 'N'
                elif y1 == min(y_all):
                    lane_entry_direction_dict[lane_entry] = 'S'
                    lane_exit_direction_dict[lane_exit] = 'E'
                elif y1 == max(y_all):
                    lane_entry_direction_dict[lane_entry] = 'N'
                    lane_exit_direction_dict[lane_exit] = 'W'
            except:
                continue
    else:  # 表明有三个方向，暂时按照四个方向处理
        for lane_entry in lane_right_entry:
            #print('a_plan{},{}'.format(lane_entry,entry_exit_link))
            try:
                lane_exit = entry_exit_link[lane_entry]
                x1, y1 = single_map_dict[lane_entry][-1].x, single_map_dict[lane_entry][-1].y
                # print('x{},y{}'.format(x1,y1))
                if x1 == min(x_all):
                    lane_entry_direction_dict[lane_entry] = 'W'
                    lane_exit_direction_dict[lane_exit] = 'S'
                elif x1 == max(x_all):
                    lane_entry_direction_dict[lane_entry] = 'E'
                    lane_exit_direction_dict[lane_exit] = 'N'
                elif y1 == min(y_all):
                    lane_entry_direction_dict[lane_entry] = 'S'
                    lane_exit_direction_dict[lane_exit] = 'E'
                elif y1 == max(y_all):
                    lane_entry_direction_dict[lane_entry] = 'N'
                    lane_exit_direction_dict[lane_exit] = 'W'
            except:
                continue

    #print('lane_entry_direction_dict {}'.format(lane_entry_direction_dict))

    return lane_entry_direction_dict, lane_exit_direction_dict


def intersection_info_extract(df_all_lane_topo_info, single_map_dict, lane_turn_left_id_ori, lane_turn_right_id_ori,
                              intersection_center_loc, intersection_range,segment_id,scenario_label):
    intersection_info = {}
    intersection_info['file_index'] = segment_id
    intersection_info['scenario_id'] = scenario_label
    intersection_info['intersection_center_point'] = intersection_center_loc
    # ---------------------筛选该目标交叉口范围的所有左转、右转车道------------------------

    range_x,range_y = intersection_range[0],intersection_range[1]
    A = (intersection_center_loc[0] - range_x / 2, intersection_center_loc[1] + range_y / 2)
    B = (intersection_center_loc[0] + range_x / 2, intersection_center_loc[1] + range_y / 2)
    C = (intersection_center_loc[0] + range_x / 2, intersection_center_loc[1] - range_y / 2)
    D = (intersection_center_loc[0] - range_x / 2, intersection_center_loc[1] - range_y / 2)
    intersection_range_approximate = [A, B, C, D]
    lane_turn_left_id = []
    lane_turn_right_id = []
    # 筛选目标交叉口内部的左转、右转车道
    for left_id in lane_turn_left_id_ori:
        if (judge_lane_in_intersection(single_map_dict[left_id], intersection_range_approximate) == 1):
            lane_turn_left_id.append(left_id)
    for right_id in lane_turn_right_id_ori:
        if (judge_lane_in_intersection(single_map_dict[right_id], intersection_range_approximate) == 1):  # !!!
            lane_turn_right_id.append(right_id)

    intersection_info['lane_id_turn_left_inside'] = lane_turn_left_id
    intersection_info['lane_id_turn_right_inside'] = lane_turn_right_id
    # 以下的处理全部基于所有右转、左转车道的分流点、合流点进行
    merging_points_right = []  # 右转车道的合流点
    diverging_points_right = []  # 右转车道的分流点
    all_lane_id = pd.unique(df_all_lane_topo_info['lane_id'].tolist())
    points_key = []  # 右转合流点、分流点集合
    df_all_lane_topo_info.loc[:, 'entry_or_exit'] = -1
    df_all_lane_topo_info.loc[:,'lane_function'] = -1

    for right_lane_id in lane_turn_right_id:
        # 提取所有与右转车道连接的分流点、合流点
        point_start_x = single_map_dict[right_lane_id][0].x  # 起点，即分流点
        point_start_y = single_map_dict[right_lane_id][0].y
        point_end_x = single_map_dict[right_lane_id][-1].x  # 终点，即合流点
        point_end_y = single_map_dict[right_lane_id][-1].y
        merging_points_right.append((point_end_x, point_end_y))
        diverging_points_right.append((point_start_x, point_start_y))
        point_start = Point(point_start_x, point_start_y)
        point_end = Point(point_end_x, point_end_y)
        points_key.append(point_start)
        points_key.append(point_end)
        # 得到该右转车道的进入车道和驶出车道
        # print(df_single_scenario_lane_topo_info[df_single_scenario_lane_topo_info['lane_id']==right_lane_id]['entry_lanes'])
        entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['entry_lanes'].iloc[0]
        exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['exit_lanes'].iloc[0]
        # 判断与右转车道相连接的车道是进口道还是出口道
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'entry_or_exit'] = 'entry'
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'entry_or_exit'] = 'exit'
        df_all_lane_topo_info.loc[
            df_all_lane_topo_info['lane_id'] == right_lane_id, 'entry_or_exit'] = 'inside'  # 右转车道自身位于交叉口内部
        # 将进口道、出口道的车道功能标记一下
        df_all_lane_topo_info.loc[
            df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_function'] = 'right'  # 记录车道功能，本身是直行车道，功能是用于右转
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_function'] = 'right'

    # print(points_key,lane_turn_right_id)

    for left_lane_id in lane_turn_left_id:  # 对于所有的左转车道
        # df_single_scenario_lane_topo_info[df_single_scenario_lane_topo_info['lane_id'] == left_lane_id]['entry_or_exit'] = 'inside'  # 左转车道自身位于位于交叉口内部
        df_all_lane_topo_info.loc[
            df_all_lane_topo_info['lane_id'] == left_lane_id, 'entry_or_exit'] = 'inside'  # 左转车道自身位于位于交叉口内部
        entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == left_lane_id]['entry_lanes'].iloc[0]
        exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == left_lane_id]['exit_lanes'].iloc[0]
        # 记录进出口道功能
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'entry_or_exit'] = 'entry'
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'entry_or_exit'] = 'exit'
        # 标记车道功能
        if (df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['lane_function'].values == [-1]):
            df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_function'] = 'left'  # 本身是直行车道，功能是用于左转
        if (df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['lane_function'].values == [-1]):
            df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_function'] = 'left'

    #对于剩下所有车道，判断是否为交叉口内部车道线
    points_key_new,points_key_new_plus = get_point_order(
        points_key)  # 得到正确的能够将所有右转合流、分流点连接为多边形的顺序  points_key_new_plus 将整个多边形向外扩展5m，以满足冗余
    for lane_in_id in all_lane_id:
        if (lane_in_id not in lane_turn_left_id) and (lane_in_id not in lane_turn_right_id):
            if judge_lane_in_intersection(single_map_dict[lane_in_id], points_key_new_plus) == 1:
                df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == lane_in_id, 'entry_or_exit'] = 'inside'  # 车道位于交叉口内部,为直行车道
                entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == lane_in_id]['entry_lanes'].iloc[0]
                exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == lane_in_id]['exit_lanes'].iloc[0]
                # print('a_plan',entry_lane_id,exit_lane_id)
                # 记录进出口道功能
                df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'entry_or_exit'] = 'entry'
                df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'entry_or_exit'] = 'exit'
                # 标记车道功能,如果该车道功能没有标记过，则进行标记

                if (df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == entry_lane_id]['lane_function'].values==[-1]):
                    df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_function'] = 'straight'  # 本身是直行车道，功能是用于直行
                if (df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == exit_lane_id]['lane_function'].values==[-1]):
                    df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_function'] = 'straight'

    #交叉口各车道方向确定,这里只标记右转车道对应的车道方向，剩下的车道的方向在提取相邻车道函数中进行标记
    #output_file(df_all_lane_topo_info,'df_all_lane_topo_info')
    df_all_lane_topo_info.loc[:,'lane_direction'] = -1
    #lane_in_intersection = df_all_lane_topo_info[(df_all_lane_topo_info['entry_or_exit']=='inside')]['lane_id'].tolist()  #得到所有在交叉口内部的车道线
    lane_right_outside_entry = df_all_lane_topo_info[(df_all_lane_topo_info['lane_function']=='right') & (df_all_lane_topo_info['entry_or_exit']=='entry')]['lane_id'].tolist()
    lane_right_outside_exit = df_all_lane_topo_info[(df_all_lane_topo_info['lane_function']=='right') & (df_all_lane_topo_info['entry_or_exit']=='exit')]['lane_id'].tolist()

    lane_entry_direction,lane_exit_direction = get_lane_direction_new(lane_turn_right_id,single_map_dict,df_all_lane_topo_info)

    for lane_id in lane_right_outside_entry:
        try:
            df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id']==lane_id,'lane_direction'] = lane_entry_direction[lane_id]
        except:
            continue
    for lane_id in lane_right_outside_exit:
        try:
            df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id']==lane_id,'lane_direction'] = lane_exit_direction[lane_id]
        except:
            continue

    #交叉口角度计算
    intersection_info['intersection_anlge'] = round(intersection_angle_cal(df_all_lane_topo_info), 2)

    # 交叉口类型判断，这里需要修正，需要讨论更多的类型
    if len(merging_points_right) >= 4 and len(diverging_points_right) >= 4:
        if 105 >= intersection_info['intersection_anlge'] >= 75:
            intersection_info['Type'] = 'Cross'  # 十字型交叉口
        elif 0 < intersection_info['intersection_anlge'] < 75 or 105 < intersection_info['intersection_anlge'] < 180:
            intersection_info['Type'] = 'X'  # X型交叉口
    elif len(merging_points_right) >= 2 and len(diverging_points_right) >= 2:
        intersection_info['Type'] = 'T'  # T型交叉口

    # ----------Extract the lane number and width of the intersection
    extract_direction_index = 1
    # print(lane_turn_right_id)
    intersection_info['direction_num'] = len(lane_turn_right_id)
    # if len(lane_turn_right_id)<=4 : #大于4的检测一定是检测错误
    for right_lane_id in lane_turn_right_id:
        lane_in_num, lane_in_id, lane_out_num, lane_out_id,df_all_lane_topo_info,lane_entry_direction,lane_exit_direction = get_one_direction_lane_info(
            right_lane_id, df_all_lane_topo_info, single_map_dict, lane_turn_left_id)
        lane_in_width, lane_out_width = get_lane_width(df_all_lane_topo_info, single_map_dict, lane_in_num, lane_in_id,
                                                       lane_out_num, lane_out_id)  # 计算车道宽度
        # print(lane_in_num, lane_in_id, lane_in_width,lane_out_num, lane_out_id, lane_out_width)
        # 进口道车道数量、车道id、车道宽度信息记录
        intersection_info['direction_' + str(extract_direction_index) + '_in' ] = lane_entry_direction[0]
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_num'] = lane_in_num
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_id_list'] = lane_in_id
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_width'] = round(lane_in_width, 2)
        # 出口道车道数量、车道id、车道宽度信息记录
        intersection_info['direction_' + str(extract_direction_index) + '_out'] = lane_exit_direction[0]
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_num'] = lane_out_num
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_id_list'] = lane_out_id
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_width'] = round(lane_out_width)
        extract_direction_index += 1

    return intersection_info, df_all_lane_topo_info, lane_turn_right_id



def scenario_to_txt(target_seg_id,target_label):  #将一个scenario的所有数据输出为txt，一般用于检查信息
    file_index = segment_id_to_file_index(target_seg_id)
    filepath = 'E:/waymo_motion_dataset/training_20s.tfrecord-' + file_index + '-of-01000'
    segment_dataset = tf.data.TFRecordDataset(filepath)
    segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())
    scenario_label = 0
    for one_record in segment_dataset:  # one_scenario 就是一个scenario
        scenario_label += 1
        if scenario_label == target_label:
            scenario = Scenario()
            scenario.ParseFromString(one_record.numpy())  # 数据格式转化
            output = "D:/Myproject/waymo-od/waymo-od/data_save/" + str(target_seg_id) + '_segment_id_' + str(target_label) + "scenario_" + ".txt"  #输出路径需要更改
            data = open(output, 'w+')
            print(scenario, file=data)
            data.close()


def file_index_to_segment_id(file_index):
    if file_index == '00000':
        segment_id = 0
    else:
        index = -1
        for i in range(1,len(file_index)):
            if (file_index[i-1] == '0') and (file_index[i] != '0'):
                index = i
                break
        segment_id = eval(file_index[index:])

    return segment_id
def segment_id_to_file_index(seg_id):
    file_index = -1
    if 0<=seg_id <= 9:
        file_index = '0000' + str(seg_id)
    elif 10<=seg_id <= 99:
        file_index = '000' + str(seg_id)
    elif 100<=seg_id <= 999:
        file_index = '00' + str(seg_id)
    elif seg_id ==1000:
        file_index = '0' + str(seg_id)
    return file_index
def judge_traffic_light(df):
    df = df.copy()
    df.insert(0, 'traffic_light', 0)  # 要指定插入列的位置，则使用insert函数
    for i in range(len(df)):
        seg_id = df['segment_id'].iloc[i]
        scenario_id = df['scenario_id'].iloc[i]
        file_index = segment_id_to_file_index(seg_id)
        path = 'E:/Result_save/data_save/dynamic_map_info/'  + file_index + '_dynamic_map_single_file_all_scenario.csv'
        df_light = pd.read_csv(path)
        scenario_id_light = pd.unique(df_light['scenario_index'])
        if scenario_id in scenario_id_light:
            df['traffic_light'].iloc[i] = 1
        else:
            df['traffic_light'].iloc[i] = 0
    return df
def map_lane_point_extract_single_scenaraio(map_features,file_index,scenario_label):
    map_point_list = []
    for single_feature in map_features:
        if list(single_feature.road_edge.polyline) != []:  #类型为road_edge
            for polyline in single_feature.road_edge.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'road_edge'
                if single_feature.road_edge.type == 0:
                    dic_map['line_type'] = 'UNKNOWN'
                elif single_feature.road_edge.type ==1:
                    dic_map['line_type'] = 'ROAD_EDGE_BOUNDARY'
                elif single_feature.road_edge.type == 2:
                    dic_map['line_type'] = 'ROAD_EDGE_MEDIAN'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                map_point_list.append(dic_map)
        if list(single_feature.lane.polyline) != []:
            for polyline in single_feature.lane.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'lane'
                if single_feature.lane.type == 0:
                    dic_map['line_type'] = 'UNDEFINED'
                elif single_feature.lane.type ==1:
                    dic_map['line_type'] = 'FREEWAY'
                elif single_feature.lane.type == 2:
                    dic_map['line_type'] = 'SURFACE_STREET'
                elif single_feature.lane.type == 3:
                    dic_map['line_type'] = 'BIKE_LANE'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                dic_map['entry_lanes'] = single_feature.lane.entry_lanes
                dic_map['exit_lanes'] = single_feature.lane.exit_lanes
                # dic_map['left_neighbors_id'] = single_feature.lane.left_neighbors.feature_id
                # dic_map['right_neighbors_id'] = single_feature.lane.right_neighbors.feature_id
                map_point_list.append(dic_map)

        if list(single_feature.road_line.polyline) != []:
            for polyline in single_feature.road_line.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'road_line'
                if single_feature.road_line.type == 0:
                    dic_map['line_type'] = 'UNKNOWN'
                elif single_feature.road_line.type ==1:
                    dic_map['line_type'] = 'BROKEN_SINGLE_WHITE'
                elif single_feature.road_line.type == 2:
                    dic_map['line_type'] = 'SOLID_SINGLE_WHITE'
                elif single_feature.road_line.type == 3:
                    dic_map['line_type'] = 'SOLID_DOUBLE_WHITE'
                elif single_feature.road_line.type == 4:
                    dic_map['line_type'] = 'BROKEN_SINGLE_YELLOW'
                elif single_feature.road_line.type == 5:
                    dic_map['line_type'] = 'BROKEN_DOUBLE_YELLOW'
                elif single_feature.road_line.type == 6:
                    dic_map['line_type'] = 'SOLID_SINGLE_YELLOW'
                elif single_feature.road_line.type == 7:
                    dic_map['line_type'] = 'SOLID_DOUBLE_YELLOW'
                elif single_feature.road_line.type == 8:
                    dic_map['line_type'] = 'PASSING_DOUBLE_YELLOW'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                map_point_list.append(dic_map)
    df_single_scenario_map_point = pd.DataFrame(map_point_list)
    return df_single_scenario_map_point

def map_lane_point_extract_single_seg(segment_dataset,file_index,test_state,target_scenario=-1):

    scenario_label = 0  # 所有场景的ID数量记录
    df_single_seg_all_scenario_map_point_info = pd.DataFrame()  # 静态地图所有车道线的散点信息提取
    for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
        scenario_label += 1
        # print('Now is the scenario:%s' % scenario_label)
        if test_state == 1:
            if scenario_label < target_scenario:
                continue
            elif scenario_label > target_scenario:
                break
        scenario = Scenario()
        scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
        map_features = scenario.map_features

        df_single_scenario_map_point = map_lane_point_extract_single_scenaraio(map_features, file_index,
                                                                               scenario_label)  # 提取地图中所有车道的散点信息
        df_single_seg_all_scenario_map_point_info = pd.concat(
            [df_single_seg_all_scenario_map_point_info, df_single_scenario_map_point], axis=0)  # 纵向方向合并

    return df_single_seg_all_scenario_map_point_info

def get_single_seg_all_scenario_lane_topo_info(segment_dataset,file_index,seg_trj,length,test_state,target_scenario = -1):

    single_scenario_all_feature = []
    scenario_label = 0  # 所有场景的ID数量记录
    df_single_seg_all_scenario_lane_topo_info = pd.DataFrame()
    for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
        scenario_label += 1
        # print('Now is the scenario:%s' % scenario_label)
        if test_state == 1:
            if scenario_label < target_scenario:
                continue
            elif scenario_label > target_scenario:
                break
        scenario = Scenario()
        scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
        map_features = scenario.map_features
        map_features_id_list = []
        scenario_trj = seg_trj[seg_trj['scenario_label'] == scenario_label]
        single_scenario_all_lane_entry_exit_info, single_map_dict, lane_turn_left_id, lane_turn_right_id,single_scenario_all_feature,map_features_id_list = map_topo_info_extract(
            map_features,single_scenario_all_feature,map_features_id_list,file_index,scenario_label)  # 车道拓扑关系提取

        df_single_scenario_lane_topo_info = pd.DataFrame(single_scenario_all_lane_entry_exit_info)
        df_single_seg_all_scenario_lane_topo_info = pd.concat(
            [df_single_seg_all_scenario_lane_topo_info, df_single_scenario_lane_topo_info], axis=0)
        filepath_fig_test = 'figure_save/intersection_topo_figure/'
        road_edge_count, lane_count, road_line, all_element_count = plot_top_view_single_pic_map_2(scenario_trj,file_index,scenario_label,
                                                                                                 scenario,lane_turn_left_id, lane_turn_right_id,length,filepath_fig_test)  #不包含交叉口信息
        print('road_edge_count {},lane_count {},road_line {},all_count {}'.format(road_edge_count, lane_count, road_line,all_element_count))
    return df_single_seg_all_scenario_lane_topo_info


def get_intersection_info(segment_dataset,df_turn_left_scenario,file_index,seg_trj,length,test_state,target_scenario = -1):
    all_intersection_info = []
    scenario_label = 0  # 所有场景的ID数量记录
    df_single_seg_all_scenario_lane_topo_info = pd.DataFrame()
    for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
        scenario_label += 1
        try:
            single_scenario_all_feature = []

            # print('Now is the scenario:%s' % scenario_label)
            if test_state == 1:
                if scenario_label < target_scenario:
                    continue
                elif scenario_label > target_scenario:
                    break
            scenario = Scenario()
            scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
            map_features = scenario.map_features

            map_features_id_list = []
            segment_id = file_index_to_segment_id(file_index)
            intersection_center_loc_list = df_turn_left_scenario[(df_turn_left_scenario['segment_id'] == segment_id) & (
                    df_turn_left_scenario['scenario_id'] == scenario_label)]['intersection_center_loc'].tolist()
            if intersection_center_loc_list != []:
                intersection_center_loc_sim = eval(intersection_center_loc_list[0])  #此时的位置是基于车辆交互位置近似得到的，不够准确，需要基于车道拓扑与几何关系重新进行提取
                # print(type(intersection_center_loc_sim[0]))
                # intersection_center_loc_sim = intersection_center_loc_sim[0]
                # print(scenario_label,intersection_center_loc_sim)
                scenario_trj = seg_trj[seg_trj['scenario_label'] == scenario_label]

                single_scenario_all_lane_entry_exit_info, single_map_dict, lane_turn_left_id, lane_turn_right_id, single_scenario_all_feature, map_features_id_list = map_topo_info_extract(
                    map_features, single_scenario_all_feature, map_features_id_list, file_index, scenario_label)  # 车道拓扑关系提取

                df_single_scenario_lane_topo_info = pd.DataFrame(single_scenario_all_lane_entry_exit_info)

                intersection_center_loc_real,intersection_range = get_real_intersection_center_point(intersection_center_loc_sim,df_single_scenario_lane_topo_info,
                                                                                  single_map_dict,lane_turn_left_id, lane_turn_right_id )  #基于提取得到的车道与交叉口信息，得到真正的交叉口中心点，同时确定交叉口提取范围
                #intersection_range 为一个元组，(l,w)包含交叉口范围的长度和宽度，之后所有基于length的代码均需要相应修改
                # 合流点、分流点信息确定，确定交叉口范围
                single_intersection_info, df_single_scenario_lane_topo_info, lane_turn_right_id_real = intersection_info_extract(
                    df_single_scenario_lane_topo_info, single_map_dict, lane_turn_left_id, lane_turn_right_id,
                    intersection_center_loc_real, intersection_range,segment_id,scenario_label)
                # print(single_intersection_info)
                all_intersection_info.append(single_intersection_info)

                # if segment_id == 10:
                #     print('aaa')
                #     print(single_intersection_info)

                df_single_seg_all_scenario_lane_topo_info = pd.concat(
                    [df_single_seg_all_scenario_lane_topo_info, df_single_scenario_lane_topo_info], axis=0)
                # print(df_single_seg_all_scenario_lane_info)
                filepath_fig_test = 'figure_save/intersection_topo_figure/'
                filepath_fig_test = 'E:/Result_save/figure_save/intersection_topo_figure/'
                road_edge_count, lane_count, road_line, all_element_count = plot_top_view_single_pic_map(scenario_trj, file_index, scenario_label, scenario,
                                                                                                         lane_turn_left_id, lane_turn_right_id, intersection_center_loc_real,
                                                                                                         intersection_range, filepath_fig_test, lane_turn_right_id_real)
        except:
            continue
    df_all_intersection_info = pd.DataFrame(all_intersection_info)
    return df_all_intersection_info,df_single_seg_all_scenario_lane_topo_info

def get_intersection_info_2(segment_dataset,df_turn_left_scenario,file_index,seg_trj,length,test_state,target_scenario = -1):
    all_intersection_info = []
    scenario_label = 0  # 所有场景的ID数量记录
    df_single_seg_all_scenario_lane_topo_info = pd.DataFrame()
    for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
        single_scenario_all_feature = []
        scenario_label += 1
        print('Now is the scenario:%s' % scenario_label)
        if test_state == 1:
            if scenario_label < target_scenario:
                continue
            elif scenario_label > target_scenario:
                break
        scenario = Scenario()
        scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
        map_features = scenario.map_features

        map_features_id_list = []
        segment_id = file_index_to_segment_id(file_index)
        intersection_center_loc_list = df_turn_left_scenario[(df_turn_left_scenario['segment_id'] == segment_id) & (
                df_turn_left_scenario['scenario_id'] == scenario_label)]['intersection_center_loc'].tolist()
        try:
            if intersection_center_loc_list != []:
                intersection_center_loc_sim = eval(intersection_center_loc_list[0])  #此时的位置是基于车辆交互位置近似得到的，不够准确，需要基于车道拓扑与几何关系重新进行提取

                scenario_trj = seg_trj[seg_trj['scenario_label'] == scenario_label]

                single_scenario_all_lane_entry_exit_info, single_map_dict, lane_turn_left_id, lane_turn_right_id, single_scenario_all_feature, map_features_id_list = map_topo_info_extract(
                    map_features, single_scenario_all_feature, map_features_id_list, file_index, scenario_label)  # 车道拓扑关系提取

                df_single_scenario_lane_topo_info = pd.DataFrame(single_scenario_all_lane_entry_exit_info)

                intersection_center_loc_real,intersection_range = get_real_intersection_center_point(intersection_center_loc_sim,df_single_scenario_lane_topo_info,
                                                                                  single_map_dict,lane_turn_left_id, lane_turn_right_id )  #基于提取得到的车道与交叉口信息，得到真正的交叉口中心点，同时确定交叉口提取范围
                #intersection_range 为一个元组，(l,w)包含交叉口范围的长度和宽度，之后所有基于length的代码均需要相应修改
                # 合流点、分流点信息确定，确定交叉口范围
                # single_intersection_info, df_single_scenario_lane_topo_info, lane_turn_right_id_real = intersection_info_extract(
                #     df_single_scenario_lane_topo_info, single_map_dict, lane_turn_left_id, lane_turn_right_id,
                #     intersection_center_loc_real, intersection_range)
                # # print(single_intersection_info)
                # all_intersection_info.append(single_intersection_info)
                #
                # df_single_seg_all_scenario_lane_topo_info = pd.concat(
                #     [df_single_seg_all_scenario_lane_topo_info, df_single_scenario_lane_topo_info], axis=0)
                # print(df_single_seg_all_scenario_lane_info)
                filepath_fig_test = 'figure_save/intersection_topo_figure/'
                #filepath_fig_test = 'E:/Result_save/figure_save/topo_fig_correct/'
                road_edge_count, lane_count, road_line, all_element_count = plot_top_view_single_pic_map(scenario_trj, file_index, scenario_label, scenario,
                                                                                                         lane_turn_left_id, lane_turn_right_id, intersection_center_loc_real,
                                                                                                         intersection_range, filepath_fig_test)
        except:
            continue
    df_all_intersection_info = pd.DataFrame(all_intersection_info)
    df_all_intersection_info, df_single_seg_all_scenario_lane_topo_info= pd.DataFrame(),pd.DataFrame()
    return df_all_intersection_info,df_single_seg_all_scenario_lane_topo_info

def get_dynamic_map_target_lane(file_index,scenario_label):
    target_lane = []
    filepath = 'E:/Result_save/data_save/'
    filepath = 'data_save/dynamic_map_info/' + file_index + '_dynamic_map_single_file_all_scenario.csv'
    #filepath = 'E:/Result_save/data_save/dynamic_map_info/' + file_index + '_dynamic_map_single_file_all_scenario.csv'
    segment_id = file_index_to_segment_id(file_index)
    df_dynamic = pd.read_csv(filepath)
    target_lane = pd.unique(df_dynamic[df_dynamic['scenario_index']==scenario_label].lane)
    return target_lane

def judge_traffic_signalphase(df_single_seg_all_scenario_lane_topo_info,dynamic_map_target_lane):  #得到该交叉口的信号相位信息
    flag = -1
    return 0

def get_joint_real_left_lane(real_lane,df_intersection_loc_point_dict):
    real_lane_list = []
    l1 = df_intersection_loc_point_dict[df_intersection_loc_point_dict['lane_i']==real_lane]['lane_j'].tolist()
    df_intersection_loc_point_dict.loc[df_intersection_loc_point_dict['lane_i']==real_lane,'check_flag'] = 1
    l2 = df_intersection_loc_point_dict[df_intersection_loc_point_dict['lane_j']==real_lane]['lane_i'].tolist()
    df_intersection_loc_point_dict.loc[df_intersection_loc_point_dict['lane_j'] == real_lane, 'check_flag'] = 1
    real_lane_list = real_lane_list + l1
    real_lane_list = real_lane_list + l2
    return real_lane_list,df_intersection_loc_point_dict

def get_real_intersection_center_point(intersection_center_loc_sim,df_single_scenario_lane_topo_info,
                                       single_map_dict,lane_turn_left_id, lane_turn_right_id):
    x_center_temp,y_center_temp = intersection_center_loc_sim[0],intersection_center_loc_sim[1]
    intersection_center_loc_real, intersection_range = (),()  #基于左转车道确定的交叉口位置，交叉口的范围确定
    lane_left_center_point = {}  #以字典形式存储不同左转车道的中心点位置
    dis_left_center_point_to_intersection = {}  #存储所有左转车道中心点距离车辆位置点的距离
    real_left_lane = []  #记录目标交叉口的所有左转车道
    #存储所有左转车道中心点的位置,并计算该点到车辆位置点的距离
    for lane in lane_turn_left_id:
        lane_point = list(single_map_dict[lane])
        lane_point_num = len(lane_point)
        lane_point_mid = lane_point[lane_point_num//2]
        x,y = lane_point_mid.x,lane_point_mid.y
        lane_left_center_point[lane] = (x,y)
        dis = np.sqrt((x_center_temp-x)**2+(y_center_temp-y)**2)
        dis_left_center_point_to_intersection[lane] = dis

    #得到所有左转车道之间的交点
    list_point_temp = []
    #intersection_loc_point_dict = {}  #交叉口左转车道的交点以及对应车道记录
    jiaodian_list = []  #所有交点记录
    for lane_i in lane_turn_left_id:
        for lane_j in lane_turn_left_id:
            if lane_i!=lane_j:
                dict_point_temp = {}
                lane_i_x, lane_i_y = [], []
                for point in single_map_dict[lane_i]:
                    lane_i_x.append(point.x)
                    lane_i_y.append(point.y)
                lane_i_x = np.array(lane_i_x)
                lane_i_y = np.array(lane_i_y)
                line_i = LineString(np.column_stack((lane_i_x, lane_i_y)))  # 将车辆轨迹转化为shapely对象
                lane_j_x, lane_j_y = [], []
                for point in single_map_dict[lane_j]:
                    lane_j_x.append(point.x)
                    lane_j_y.append(point.y)
                line_j = LineString(np.column_stack((lane_j_x, lane_j_y)))
                interscetion = line_i.intersection(line_j)  # 得到轨迹交点，即冲突点

                # print(interscetion.xy[0])
                # print(len(interscetion.xy[0]))
                try:
                    if len(interscetion.xy[0])>0:
                        dict_point_temp['lane_i'] = lane_i
                        dict_point_temp['lane_j'] = lane_j
                        dict_point_temp['lane_intersection'] = (interscetion.xy[0][0], interscetion.xy[1][0])
                        dict_point_temp['check_flag'] = 0
                        list_point_temp.append(dict_point_temp)
                        #intersection_loc_point_dict[(lane_i,lane_j)] = (interscetion.xy[0][0], interscetion.xy[1][0])
                except:
                    continue
    df_intersection_loc_point_dict = pd.DataFrame(list_point_temp) #记录所有左转车道的交点信息

    #对所有交点进行检索，目标交叉口的几个交点一定是距离目标车辆位置点最近的，然后根据拓扑关系进行遍历检索
    min_dis = 9999
    point_min_dis = ()
    lane_pair_min_dis = ()
    for i in range(len(df_intersection_loc_point_dict)):
        x_left,y_left = df_intersection_loc_point_dict['lane_intersection'].iloc[i]
        dis = np.sqrt((x_center_temp-x_left)**2+(y_center_temp-y_left)**2)
        if dis<min_dis:
            min_dis = dis
            point_min_dis = (x_left,y_left)
            lane_pair_min_dis = (df_intersection_loc_point_dict['lane_i'].iloc[i],df_intersection_loc_point_dict['lane_j'].iloc[i])
    real_left_lane.append(lane_pair_min_dis[0])
    real_left_lane.append(lane_pair_min_dis[1])
    real_lane_i = lane_pair_min_dis[0]
    real_lane_j = lane_pair_min_dis[1]
    df_intersection_loc_point_dict.loc[df_intersection_loc_point_dict['lane_intersection'] == point_min_dis,'check_flag'] = 1

    real_lane_i_joint,df_intersection_loc_point_dict = get_joint_real_left_lane(real_lane_i,df_intersection_loc_point_dict)
    real_lane_j_joint,df_intersection_loc_point_dict = get_joint_real_left_lane(real_lane_j,df_intersection_loc_point_dict)
    real_left_lane += real_lane_i_joint
    real_left_lane += real_lane_j_joint
    for i in range(3):
        flag_all_check = 0
        for lane in real_left_lane:
            check_flag_i = df_intersection_loc_point_dict[df_intersection_loc_point_dict['lane_i']==lane]['check_flag'].tolist()
            check_flag_j = df_intersection_loc_point_dict[df_intersection_loc_point_dict['lane_j']==lane]['check_flag'].tolist()
            if (0 in check_flag_i) or (0 in check_flag_j):
                real_lane_k_joint, df_intersection_loc_point_dict = get_joint_real_left_lane(lane,df_intersection_loc_point_dict)
                flag_all_check = 1
                #print('real_lane_k_joint {}'.format(real_lane_k_joint))
                real_left_lane += real_lane_k_joint
            real_left_lane = list(set(real_left_lane))  #去掉重复元素
        if flag_all_check == 0:
            break #说明目标交叉口的所有车道均已经被检索到
    x_left_lane_all = []
    y_left_lane_all = []
    lane_x_range_all = []  #所有左转车道x坐标轴上变化范围
    lane_y_range_all = []
    for lane in real_left_lane:
        x_left_lane_all.append(lane_left_center_point[lane][0])
        y_left_lane_all.append(lane_left_center_point[lane][1])
        #对该车道的x、y坐标变化范围进行记录
        lane_x, lane_y = [], []
        for point in single_map_dict[lane]:
            lane_x.append(point.x)
            lane_y.append(point.y)
        x_range = max(lane_x)-min(lane_x)
        y_range = max(lane_y)-min(lane_y)
        lane_x_range_all.append(x_range)
        lane_y_range_all.append(y_range)

    #确定实际的交叉口位置
    x_mean = np.mean(x_left_lane_all)
    y_mean = np.mean(y_left_lane_all)
    intersection_center_loc_real = (x_mean,y_mean)
    intersection_range_x = int(np.mean(lane_x_range_all)*3)
    intersection_range_y = int(np.mean(lane_y_range_all)*3)
    intersection_range = (intersection_range_x,intersection_range_y)

    #还可以根据交点个数重新判断一下交叉口的类型
    # interscetion_type = ''
    # if len(real_left_lane) == 2:
    #     interscetion_type = 'T'
    # elif len(real_left_lane) >=4:
    #     interscetion_type = 'Cross_or_X'
    return intersection_center_loc_real,intersection_range

def output_file(df,path2):  #输出文件测试
    path1 = 'E:/Result_save/data_save'
    path1 = 'data/test_file/'
    path = path1 + path2 + '.csv'
    df.to_csv(path)
    print('test_file has been printed')

def get_interest_od(df_obj):
    interest_dict = {}
    scenario_label_list = df_obj['scenario_label'].tolist()
    for label in scenario_label_list:
        # print(df_obj[df_obj['scenario_label']==label].objects_of_interest.tolist()[0])
        interest_dict[label] = eval(df_obj[df_obj['scenario_label'] == label].objects_of_interest.tolist()[0])
    return interest_dict

def get_lane_num_diff_direction(veh_id, df,map_features,df_intersection_info_single_scenario):
    single_map_dict = {}
    for single_feature in map_features:
        if list(single_feature.lane.polyline) != []:
            single_map_dict[single_feature.id] = single_feature.lane.polyline  # 使用字典进行检索，得到所有车道的坐标点信息
    lane_num_in, lane_width_in,lane_num_out,lane_width_out = -1, -1,-1,-1
    lane_id_in,lane_id_out = -1,-1
    direction_num = df_intersection_info_single_scenario['direction_num'].iloc[0]

    lane_dict = {}
    df_veh = df[df['obj_id'] == veh_id]
    # print(df_veh)
    veh_point_list = []
    for i in range(len(df_veh)):
        veh_x = df_veh['center_x'].iloc[i]
        veh_y = df_veh['center_y'].iloc[i]
        veh_point_list.append((veh_x, veh_y))
    # print(veh_point_list)
    min_dis_in, id_min_dis_in, direction_min_dis_in = 9999, -1, -1
    min_dis_out, id_min_dis_out, direction_min_dis_out = 9999, -1, -1
    for i in range(1,direction_num+1):
        lane_dict['lane_'+str(i)+'_in'] = df_intersection_info_single_scenario['direction_' + str(i) + '_in_lane_id_list']
        lane_dict['width_'+str(i)+'_in'] = df_intersection_info_single_scenario['direction_' + str(i) + '_in_lane_width']
        lane_dict['lane_' + str(i) + '_out'] = df_intersection_info_single_scenario['direction_' + str(i) + '_out_lane_id_list']
        lane_dict['width_' + str(i) + '_out'] = df_intersection_info_single_scenario['direction_' + str(i) + '_out_lane_width']
        #进口道车道信息检索
        lane_in_list = df_intersection_info_single_scenario['direction_' + str(i) + '_in_lane_id_list']
        for list_lane_in in lane_in_list:
            # print(type(eval(list_lane)))
            for lane_id in eval(list_lane_in):
                # print('single',lane_id)
                lane_point_list_in = []
                # print(lane_id)
                # print(single_map_dict.keys())
                for polyline in single_map_dict[lane_id]:
                    lane_point_list_in.append((polyline.x,polyline.y))
                x_point_last = lane_point_list_in[-1][0]  #进口道应该选最后一个点
                y_point_last = lane_point_list_in[-1][1]
                temp_min_dis_in = 9999
                for j in range(len(veh_point_list)):
                    x_veh,y_veh = veh_point_list[j][0],veh_point_list[j][1]
                    dis = np.sqrt((x_veh-x_point_last)**2+(y_veh-y_point_last)**2)
                    # if veh_id == 1859 and (lane_id == 53 or lane_id == 68):
                    #     print('lane_id is {},dis_plan is {}'.format(lane_id,dis_plan))
                    if dis< temp_min_dis_in:
                        temp_min_dis_in = dis
                if temp_min_dis_in < min_dis_in:
                    min_dis_in = temp_min_dis_in
                    id_min_dis_in = lane_id
                    direction_min_dis_in = i
        #出口道信息检索
        lane_out_list = df_intersection_info_single_scenario['direction_' + str(i) + '_out_lane_id_list']
        for list_lane_out in lane_out_list:
            for lane_id in eval(list_lane_out):
                lane_point_list_out = []
                for polyline in single_map_dict[lane_id]:
                    lane_point_list_out.append((polyline.x, polyline.y))
                x_point_first = lane_point_list_out[0][0]  #出口道应该选第一个点
                y_point_first = lane_point_list_out[0][1]
                temp_min_dis_out = 9999
                for j in range(len(veh_point_list)):
                    x_veh,y_veh = veh_point_list[j][0],veh_point_list[j][1]
                    dis = np.sqrt((x_veh - x_point_first) ** 2 + (y_veh - y_point_first) ** 2)
                    if dis< temp_min_dis_out:
                        temp_min_dis_out = dis
                if temp_min_dis_out < min_dis_out:
                    min_dis_out = temp_min_dis_out
                    id_min_dis_out = lane_id
                    direction_min_dis_out = i
    # print(veh_id,len(eval(lane_dict['lane_'+str(direction_min_dis_in)+'_in'].iloc[0])),lane_dict['lane_'+str(direction_min_dis_out)+'_in'].iloc[0])
    lane_num_in,lane_num_out = len(eval(lane_dict['lane_'+str(direction_min_dis_in)+'_in'].iloc[0])),len(eval(lane_dict['lane_'+str(direction_min_dis_out)+'_in'].iloc[0]))
    lane_width_in,lane_width_out = float(lane_dict['width_'+str(direction_min_dis_in)+'_in']),float(lane_dict['width_'+str(direction_min_dis_out)+'_in'])
    lane_id_in,lane_id_out = id_min_dis_in,id_min_dis_out

    return lane_num_in, lane_width_in,lane_id_in,lane_num_out,lane_width_out,lane_id_out


def get_lane_num_diff_direction_all_seg(df_turn_left_scenario,df_all_intersection_info,interest_od_file_list, data_file_list, data_file_index_list, all_file_list, file_index_list,test_state,test_seg,test_scenario):
    all_left_seg = pd.unique(df_turn_left_scenario['segment_id'].tolist())
    veh_lane_info_all_seg = []
    for i in tqdm(range(len(file_index_list))):
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        segment_id = file_index_to_segment_id(file_index)
        if segment_id not in all_left_seg:
            continue
        print('Now is the file:%s' % file_index)
        all_scenario_label_single_seg = pd.unique(df_turn_left_scenario[df_turn_left_scenario['segment_id']==segment_id]['scenario_id'].tolist())
        all_scenario_label_single_seg_intersection = pd.unique(df_all_intersection_info[df_all_intersection_info['file_index']==segment_id]['scenario_id'].tolist())
        if test_state == 1 :
            if segment_id< test_seg :
                continue
            if segment_id > test_seg :
                break

        df_trj = pd.read_csv(data_file_list[i])  #轨迹信息提取
        #-------------Tf格式文件数据类型转换--------------------
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

        scenario_label = 0
        for one_scenario in segment_dataset:
            veh_lane_info_single_seg_single_scenario = {}
            scenario_label += 1
            if (scenario_label not in all_scenario_label_single_seg) or (scenario_label not in all_scenario_label_single_seg_intersection):
                # print('scenario_label {} is not in all_seg or not in intersection info'.format(scenario_label))
                continue

            df_intersection_info_single_scenario = df_all_intersection_info[
                (df_all_intersection_info['file_index'] == segment_id) & (
                            df_all_intersection_info['scenario_id'] == scenario_label)]
            # print(segment_id,scenario_label)
            # print(df_intersection_info_single_scenario)
            # print(len(df_intersection_info_single_scenario['direction_num'])==0)
            if (len(df_intersection_info_single_scenario['direction_num'])==0) :
                continue
            if (df_intersection_info_single_scenario['direction_num'].iloc[0]) >4: #检测出5个及以上交叉口的一定有问题
                continue
            # print("file {},scenario is {}".format(file_index,scenario_label))
            scenario = Scenario()
            scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
            map_features = scenario.map_features

            df = df_trj[(df_trj['scenario_label'] == scenario_label) & (df_trj['valid'] == True) & (df_trj['obj_type'] == 1)]
            left_veh_id = int(df_turn_left_scenario[(df_turn_left_scenario['segment_id']==segment_id) &
                                                    (df_turn_left_scenario['scenario_id']==scenario_label)]['turn_left_veh_id'])
            veh_temp = [int(df_turn_left_scenario[(df_turn_left_scenario['segment_id']==segment_id) &
                                                    (df_turn_left_scenario['scenario_id']==scenario_label)]['interactive_veh_1_id']),
                        int(df_turn_left_scenario[(df_turn_left_scenario['segment_id']==segment_id) &
                                                    (df_turn_left_scenario['scenario_id']==scenario_label)]['interactive_veh_2_id'])]
            for v_id in veh_temp:
                if v_id != left_veh_id:
                    straight_veh_id = v_id  #得到直行车辆id
            # print(file_index,segment_id,scenario_label,left_veh_id,straight_veh_id)
            # print(df)
            # print(df[df['obj_id']==left_veh_id])
            # print(pd.unique(list(df['obj_id'])))

            lane_num_left_in, lane_width_left_in,lane_id_left_in,lane_num_left_out,\
            lane_width_left_out,lane_id_left_out = get_lane_num_diff_direction(left_veh_id, df,map_features,df_intersection_info_single_scenario)  #可以使用交叉口已经提取出来的信息进行判断
            lane_num_straight_in, lane_width_straight_in,lane_id_straight_in,lane_num_straight_out,\
            lane_width_straight_out,lane_id_straight_out = get_lane_num_diff_direction(straight_veh_id, df,map_features,df_intersection_info_single_scenario)

            veh_lane_info_single_seg_single_scenario['segment_id'] = segment_id
            veh_lane_info_single_seg_single_scenario['scenario_id'] = scenario_label
            veh_lane_info_single_seg_single_scenario['left_veh_id'] = left_veh_id
            veh_lane_info_single_seg_single_scenario['lane_id_left_in'] = lane_id_left_in
            veh_lane_info_single_seg_single_scenario['lane_num_left_in'] = lane_num_left_in
            veh_lane_info_single_seg_single_scenario['lane_width_left_in'] = lane_width_left_in
            veh_lane_info_single_seg_single_scenario['lane_id_left_out'] = lane_id_left_out
            veh_lane_info_single_seg_single_scenario['lane_num_left_out'] = lane_num_left_out
            veh_lane_info_single_seg_single_scenario['lane_width_left_out'] = lane_width_left_out
            veh_lane_info_single_seg_single_scenario['straight_veh_id'] = straight_veh_id
            veh_lane_info_single_seg_single_scenario['lane_id_straight_in'] = lane_id_straight_in
            veh_lane_info_single_seg_single_scenario['lane_num_straight_in'] = lane_num_straight_in
            veh_lane_info_single_seg_single_scenario['lane_width_straight_in'] = lane_width_straight_in
            veh_lane_info_single_seg_single_scenario['lane_id_straight_out'] = lane_id_straight_out
            veh_lane_info_single_seg_single_scenario['lane_num_straight_out'] = lane_num_straight_out
            veh_lane_info_single_seg_single_scenario['lane_width_straight_out'] = lane_width_straight_out

            veh_lane_info_all_seg.append(veh_lane_info_single_seg_single_scenario)

    df_veh_lane_info_all_seg = pd.DataFrame(veh_lane_info_all_seg)
    outpath = 'data/veh_lane_info_test.csv'
    outpath = 'E:/Result_save/data_save/veh_lane_info_all.csv'
    df_veh_lane_info_all_seg.to_csv(outpath)

def csv_concat():
    filepath1 = 'E:/Result_save/data_save/all_intersection_info/*all_intersection_info.csv'
    filepath2 = 'E:/Result_save/data_save/all_lane_topo_info/*all_seg_all_lane_topo_info.csv'

    all_files_1 = sorted(glob.glob(filepath1))
    df_all_intersection_info = pd.DataFrame()
    for file in all_files_1:
        df_1 = pd.read_csv(file)
        df_all_intersection_info = pd.concat([df_all_intersection_info, df_1], axis=0)
    out_path1 = 'E:/Result_save/data_save/all_intersection_info.csv'
    df_all_intersection_info.to_csv(out_path1)

    all_files_2 = sorted(glob.glob(filepath2))
    df_all_seg_all_lane_topo = pd.DataFrame()
    for file in all_files_2:
        df_2 = pd.read_csv(file)
        df_all_seg_all_lane_topo = pd.concat([df_all_seg_all_lane_topo, df_2], axis=0)
    out_path2 = 'E:/Result_save/data_save/all_seg_all_lane_topo_info.csv'
    df_all_seg_all_lane_topo.to_csv(out_path2)





