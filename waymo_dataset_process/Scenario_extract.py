import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from shapely.geometry import LineString


def data_heading_check(df):  # 异常数据处理  数据平滑
    df_new = df.copy()
    length = df_new.shape[0]
    for i in range(1, length - 1):
        # if k < 3:
        #     #print(df.columns)
        #     print(df['heading_180'].iloc[i])
        #     print(df.iloc[i,-1])
        #     k += 1
        if (abs(float(df_new['heading_180'].iloc[i]) - float(df_new['heading_180'].iloc[i - 1])) > 30) and (
                abs(float(df_new['heading_180'].iloc[i]) - float(df_new['heading_180'].iloc[i + 1])) > 30):
            df_new.iloc[i, -1] = (float(df_new['heading_180'].iloc[i - 1]) + float(df_new['heading_180'].iloc[i + 1])) / 2
        if (float(df_new['heading_180'].iloc[i]) - float(df_new['heading_180'].iloc[i - 1])) > 350:
            df_new.iloc[i, -1] -= 360
        if (float(df_new['heading_180'].iloc[i]) - float(df_new['heading_180'].iloc[i - 1])) < -350:
            df_new.iloc[i, -1] += 360
    if (float(df_new['heading_180'].iloc[length - 1]) - float(df_new['heading_180'].iloc[length - 2])) > 350:
        df_new.iloc[length - 1, -1] -= 360
    if (float(df_new['heading_180'].iloc[length - 1]) - float(df_new['heading_180'].iloc[length - 2])) < -350:
        df_new.iloc[length - 1, -1] += 360
    return df_new

def get_time_interval(veh1_id,veh2_id,df):  #寻找两辆车同时出现的公共时间区间
    t_begin,t_end = -1,-1
    df_veh1 = df[df['obj_id'] == veh1_id]
    df_veh2 = df[df['obj_id'] == veh2_id]
    # 寻找共同的时间区间
    t_begin = max(df_veh1['time_stamp'].min(), df_veh2['time_stamp'].min())
    t_end = min(df_veh1['time_stamp'].max(), df_veh2['time_stamp'].max())  # 取结束时间的较小值
    return t_begin,t_end

def process_data(df, interest_od):
    df_veh11 = df[df['obj_id'] == interest_od[0]]
    df_veh1 = df_veh11.copy()
    df_veh1.loc[:,'heading_180'] = df_veh1.loc[:,'heading'] * 180 / np.pi  # 将航向角从弧度制转为角度制
    df_veh1 = data_heading_check(df_veh1)
    df_veh22 = df[df['obj_id'] == interest_od[1]]
    df_veh2 = df_veh22.copy()
    df_veh2.loc[:,'heading_180'] = df_veh2['heading'] * 180 / np.pi  # 将航向角从弧度制转为角度制
    df_veh2 = data_heading_check(df_veh2)
    # 寻找共同的时间区间
    # t_begin = max(df_veh1['time_stamp'].min(), df_veh2['time_stamp'].min())
    # t_end = min(df_veh1['time_stamp'].max(), df_veh2['time_stamp'].max())  # 取结束时间的较小值
    t_begin,t_end = get_time_interval(interest_od[0],interest_od[1],df)
    # print("time_range_index {},{}".format(t_begin,t_end))
    df_veh1 = df_veh1[(df_veh1['time_stamp'] >= t_begin) & (df_veh1['time_stamp'] <= t_end)]
    df_veh2 = df_veh2[(df_veh2['time_stamp'] >= t_begin) & (df_veh2['time_stamp'] <= t_end)]

    return df_veh1, df_veh2


def get_interest_od(df_obj):
    interest_dict = {}
    scenario_label_list = df_obj['scenario_label'].tolist()
    for label in scenario_label_list:
        # print(df_obj[df_obj['scenario_label']==label].objects_of_interest.tolist()[0])
        interest_dict[label] = eval(df_obj[df_obj['scenario_label'] == label].objects_of_interest.tolist()[0])
    return interest_dict


def judge_dis(df_veh1, df_veh2):
    flag = 0  # 判断相对距离是否小于距离阈值
    dis_each = []
    length = min(df_veh1.shape[0], df_veh2.shape[0])  # 数据长度,即为时间戳的长度
    # print('length {},{}'.format(df_veh1.shape[0],df_veh2.shape[0]))
    min_dis = 9999
    time_stamp_min_dis = -1  # 距离最近时的时刻
    time_min_list_temp = []  # 用于记录可能的最近记录对应的时刻
    time_range_interactive = ()  # 用于记录车辆交互行为的起、终点时刻
    time_inter_list = []  # 用于记录车辆交互的时间范围，在距离阈值内时刻都会记录下来
    loc_list_temp = []
    loc_veh_1_min_dis = ()  # 用于记录距离最近时对应的两辆车的坐标位置 2辆车*2维信息
    loc_veh_2_min_dis = ()
    # print(df_veh1)
    for i in range(length):
        x1, y1 = df_veh1['center_x'].iloc[i], df_veh1['center_y'].iloc[i]
        x2, y2 = df_veh2['center_x'].iloc[i], df_veh2['center_y'].iloc[i]
        dis = np.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

        if dis < 15:  # 小于15m即认为满足距离阈值限制
            flag = 1
            dis_each.append(dis)
            time_min_list_temp.append(float(df_veh1['time_stamp'].iloc[i]))
            loc_veh1 = (x1, y1)
            loc_veh2 = (x2, y2)
            loc_list_temp.append((loc_veh1, loc_veh2))
        if dis < 50:
            time_inter_list.append(float(df_veh1['time_stamp'].iloc[i]))
    if dis_each:
        min_dis = min(dis_each)
        min_dis_index = dis_each.index(min_dis)
        time_stamp_min_dis = time_min_list_temp[min_dis_index]
        loc_veh_1_min_dis, loc_veh_2_min_dis = loc_list_temp[min_dis_index]
    else:
        min_dis = 9999
    time_range_interactive = (min(time_inter_list), max(time_inter_list))  # 得到这个列表内的时间最大值和最小值，即为这段交互的整个时间段

    return flag, min_dis, time_stamp_min_dis, time_range_interactive, loc_veh_1_min_dis, loc_veh_2_min_dis

def cal_angle(x1, y1, x2, y2):
    if x2 != x1:
        angle = (math.atan((y2 - y1) / (x2 - x1))) * 180 / np.pi
    else:
        angle = 90  # 避免斜率不存在的情况
    return angle

def get_veh_angle_chane(df,veh_id):  # 计算
    df_veh = df[df['obj_id'] == veh_id]
    angle_start = 0
    angle_end = 0
    length = len(df_veh)
    x_list = df_veh['center_x'].tolist()
    y_list = df_veh['center_y'].tolist()
    turn_type = 'straight'
    x1, y1 = x_list[0], y_list[0]
    x4, y4 = x_list[-1], y_list[-1]
    try:
        # print(polyline_list)
        x2, y2 = x_list[3], y_list[3]
        x3, y3 = x_list[-3], y_list[-3]
        angle_start = cal_angle(x1, y1, x2, y2)
        angle_end = cal_angle(x3, y3, x4, y4)
        delta_angle = angle_end - angle_start  # 大于0为左转，小于0为右转
    except:
        angle_start = angle_end = delta_angle = None

    # 判断左右转信息
    index_mid = int(length / 2)
    x_mid = x_list[index_mid]
    y_mid = y_list[index_mid]
    p1 = (x_mid - x1) * (y4 - y_mid) - (y_mid - y1) * (x4 - x_mid)
    # print(p1)
    if abs(delta_angle) > 35 :
        if p1 > 0:
            turn_type = 'left'
        elif p1 < 0:
            turn_type = 'right'
    else:
        turn_type = 'straight'
    # print("Turn type is %s"%turn_type)
    return delta_angle, turn_type


def judge_angle(df_veh1, df_veh2,df):
    # 如果是-的，加360
    flag = 0  # 判断航向角转向是否小于角度阈值
    left_flag = 0  # 判断是左转还是右转，左转flag为1
    angle_range1 = float(df_veh1['heading_180'].max() - df_veh1['heading_180'].min())
    index_1 = [df_veh1['heading_180'].idxmax(), df_veh1['heading_180'].idxmin()]  # idxmax()  返回最大值对应的索引
    angle_range2 = float(df_veh2['heading_180'].max() - df_veh2['heading_180'].min())
    index_2 = [df_veh2['heading_180'].idxmax(), df_veh2['heading_180'].idxmin()]
    max_index = 0
    max_index_list = []
    if angle_range1 > angle_range2:
        max_angle = angle_range1
        max_index_list = index_1
        min_angle = angle_range2
        left_veh_id = df_veh1['obj_id'].iloc[0]
        straight_veh_id = df_veh2['obj_id'].iloc[0]
    else:
        max_angle = angle_range2
        max_index_list = index_2
        min_angle = angle_range1
        left_veh_id = df_veh2['obj_id'].iloc[0]
        straight_veh_id = df_veh1['obj_id'].iloc[0]
    # print(max_index_list)
    # print(max_angle,min_angle)
    if max_angle > 70 and min_angle < 40:
        flag = 1
        delta_angle, turn_type = get_veh_angle_chane(df,left_veh_id)
        if (max_index_list[0] > max_index_list[1]) and (turn_type == 'left'):
            left_flag = 1
    return flag, left_flag, max_index_list, angle_range1, angle_range2, left_veh_id,straight_veh_id


def get_file_list(filepath):
    all_files = sorted(glob.glob(filepath))
    segs_name_index = []
    for file in all_files:
        segment_name = os.path.basename(file)
        segs_name_index.append(segment_name[:5])
    # print(segs_name_all)
    # print(all_files)
    return all_files, segs_name_index

def scenario_extract_v1(df_trj,df_obj,all_left_scenario,segment_id):  #左转+直行的冲突场景筛选

    scenario_label_list = pd.unique(df_trj['scenario_label'].tolist())
    # print(scenario_label_list)
    interest_od_dict = get_interest_od(df_obj)
    dic_heading = {}
    for i in range(len(scenario_label_list)):
        try:
            dic_single_left_scenario = {}
            label = scenario_label_list[i]
            interest_od = interest_od_dict[label]
            df = df_trj[(df_trj['scenario_label'] == label) & (df_trj['valid'] == True) & (df_trj['obj_type']==1)]
            df_veh1, df_veh2 = process_data(df, interest_od)
            flag_dis, dis_each_min, time_stamp_min_dis, time_range_interactive, loc_veh_1_min_dis, loc_veh_2_min_dis = judge_dis(df_veh1, df_veh2)
            flag_angle, left_flag, angle_index_list, angle_veh_1, angle_veh_2, left_veh_id,straight_veh_id = judge_angle(df_veh1, df_veh2,df)
            # print("Scenario {:.2f}, Angle is {:.2f} and {:.2f},flag_angle is {} ,distence is {:.2f},angle_index {},left_flag {}".format(label,angle_veh_1,angle_veh_2,flag_angle,dis_each_min,angle_index_list,left_flag))

            if flag_dis == 1 and flag_angle == 1 and left_flag == 1:
                dic_single_left_scenario['segment_id'] = segment_id
                dic_single_left_scenario['scenario_id'] = label
                dic_single_left_scenario['interactive_veh_1_id'] = interest_od[0]
                dic_single_left_scenario['interactive_veh_2_id'] = interest_od[1]
                dic_single_left_scenario['turn_left_veh_id'] = left_veh_id
                dic_single_left_scenario['min_dis'] = dis_each_min
                dic_single_left_scenario['time_stamp_min_dis'] = time_stamp_min_dis
                dic_single_left_scenario['time_range_interactive'] = time_range_interactive
                dic_single_left_scenario['loc_veh_1_interactive'] = loc_veh_1_min_dis
                dic_single_left_scenario['loc_veh_2_interactive'] = loc_veh_2_min_dis
                dic_single_left_scenario['intersection_center_loc'] = (
                    (loc_veh_1_min_dis[0] + loc_veh_2_min_dis[0]) / 2,
                    (loc_veh_1_min_dis[1] + loc_veh_2_min_dis[1]) / 2)  # 用两车的位置的平均值近似代替交叉口的中心位置
                dic_single_left_scenario['angle_veh_1'] = angle_veh_1
                dic_single_left_scenario['angle_veh_2'] = angle_veh_2
                all_left_scenario.append(dic_single_left_scenario)
                print(
                    '存在左转场景第{}个,两车OD为{}、{},最小距离为{:.2f}m,两车是航向角变化分别为{:.2f}°,{:.2f}°,'
                    'left_veh_id {}'.format(label,interest_od[0],interest_od[1],dis_each_min,angle_veh_1,angle_veh_2,left_veh_id))
        except:
            continue

    return all_left_scenario

def get_left_veh_single_scenario(df):
    left_veh_count = 0
    left_veh_list = []
    right_veh_count = 0
    right_veh_list = []
    straight_veh_count = 0
    straight_veh_list = []
    veh_id_list = pd.unique(df['obj_id'].tolist())
    for veh_id in veh_id_list:
        delta_angle, turn_type = get_veh_angle_chane(df, veh_id)
        if turn_type == 'left':
            left_veh_count += 1
            left_veh_list.append(veh_id)
        elif turn_type == 'right':
            right_veh_count += 1
            right_veh_list.append(veh_id)
        elif turn_type == 'straight':
            straight_veh_count += 1
            straight_veh_list.append(veh_id)
    return left_veh_count,left_veh_list,right_veh_count,right_veh_list,straight_veh_count,straight_veh_list

def get_real_type_veh(interest_veh_id,all_veh_list,time_stamp_min_dis,df):
    real_type_veh_list = []
    real_type_veh_list.append(interest_veh_id)
    df_veh_interest = df[(df['obj_id']==interest_veh_id) & (df['time_stamp']==time_stamp_min_dis)]
    x1,y1 = float(df_veh_interest['center_x']),float(df_veh_interest['center_y'])
    for veh_id in all_veh_list:
        if veh_id != interest_veh_id:
            df_veh_ori = df[(df['obj_id']==veh_id) & (df['time_stamp']==time_stamp_min_dis)]
            if len(df_veh_ori)>0:
                x2,y2 = float(df_veh_ori['center_x']), float(df_veh_ori['center_y'])
                #print('x2={},y2={}'.format(x2, y2))
                dis = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                #print('dis is {}'.format(dis))
                if dis < 50:
                    real_type_veh_list.append(veh_id)
    return  real_type_veh_list

def get_conflict_point(left_veh_id,straight_veh_id,df):  #基于左转和直行交互车辆的轨迹得到冲突点坐标
    #t_begin,t_end = get_time_interval(left_veh_id,straight_veh_id,df)
    df_veh_left = df[df['obj_id']==left_veh_id]
    veh_left_trj = np.column_stack((df_veh_left['center_x'],df_veh_left['center_y']))  #得到车辆的轨迹信息
    df_veh_straight = df[df['obj_id']==straight_veh_id]
    veh_straight_trj = np.column_stack((df_veh_straight['center_x'],df_veh_straight['center_y']))
    line_left = LineString(veh_left_trj)  #将车辆轨迹转化为shapely对象
    line_straight = LineString(veh_straight_trj)
    interscetion = line_left.intersection(line_straight)  #得到轨迹交点，即冲突点
    # print('intesection {}'.format(interscetion))
    try:
        conflict_point = (interscetion.xy[0][0], interscetion.xy[1][0])
    except:
        conflict_point = ()
    return conflict_point


def get_time_stamp_veh_passing_by(veh_list,conflict_point,df,target_veh_id=-1):
    #print(veh_list,target_veh_id)
    time_list_passing = []
    veh_real_passing = []  #确实经过该冲突点的车辆
    target_veh_time_stamp_passing = -1
    x_con,y_con = conflict_point[0],conflict_point[1]
    for veh_id in veh_list:
        dis_min = 999
        index_min = -1
        df_veh = df[df['obj_id']==veh_id]
        for i in range(len(df_veh)):
            x,y = df_veh['center_x'].iloc[i],df_veh['center_y'].iloc[i]
            dis = np.sqrt((x-x_con)**2+(y-y_con)**2)
            #print(dis)
            if dis < dis_min:
                dis_min = dis
                index_min = i
        #print('veh_id is {},dis_min is {}'.format(veh_id,dis_min))
        if dis_min < 3:  #是否经过冲突点的判断阈值，这里暂时设置为3
            time_stamp_i = df_veh['time_stamp'].iloc[index_min]
            #print(time_stamp_i)
            time_list_passing.append(time_stamp_i)
            veh_real_passing.append(veh_id)
            if veh_id == target_veh_id:
                target_veh_time_stamp_passing = time_stamp_i
    # print('time_list_passing is {},all_veh is {},target_veh is {},targrt_time_passing is {}'.format(time_list_passing,veh_real_passing,target_veh_id,target_veh_time_stamp_passing))
    return time_list_passing,veh_real_passing,target_veh_time_stamp_passing
def cal_trj_slpoe(x,y):
    x1, y1 = x[0], y[0]  # 直线起点xy坐标
    x2, y2 = x[-1], y[-1] # 直线终点xy坐标
    dis = np.sqrt((x1-x2)**2+(y1-y2)**2)
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = 90
    return dis,slope

def get_diff_direction_veh(straight_veh_id,straight_veh_list,df):
    straight_veh_list_direction_1,straight_veh_list_direction_2 = [],[]
    temp_id_1,temp_id_2 = [],[]
    slope_list, id_label = [], {}  #直行车辆轨迹的斜率以及对应的id标签
    count_id = 0
    for i in range(len(straight_veh_list)):
        veh_id = straight_veh_list[i]
        df_veh = df[df['obj_id'] == veh_id]
        if len(df_veh>20):
            x = df_veh['center_x'].tolist()
            y = df_veh['center_y'].tolist()
            dis,slope = cal_trj_slpoe(x,y)
            if dis > 10:
                slope_list.append(slope)
                id_label[count_id] = veh_id
                count_id += 1
    from sklearn.cluster import KMeans
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(np.array(slope_list).reshape(-1, 1))  # 聚类  每次聚类label打1还是2 这是随机的
    label_pred = estimator.labels_  # 获取聚类标签
    label_index_1 = np.where(label_pred==0)[0]  #寻找指定元素的位置
    label_index_2 = np.where(label_pred==1)[0]
    for index_1 in label_index_1:
        temp_id_1.append(id_label[index_1])
    for index_2 in label_index_2:
        temp_id_2.append(id_label[index_2])
    #print(temp_id_1,temp_id_2)
    if straight_veh_id in temp_id_1:
        straight_veh_list_direction_1 = temp_id_1
    else:
        straight_veh_list_direction_1 = temp_id_2

    #print(straight_veh_list_direction_1,straight_veh_list_direction_2)
    return straight_veh_list_direction_1,straight_veh_list_direction_2

def scenario_extract_v2(df_trj,df_obj,all_left_scenario,segment_id):   #一个场景中存在多个左转车辆的情况筛选
    scenario_label_list = pd.unique(df_trj['scenario_label'].tolist())
    for i in range(len(scenario_label_list)):
        dic_single_left_scenario = {}
        label = scenario_label_list[i]
        df = df_trj[(df_trj['scenario_label'] == label) & (df_trj['valid'] == True) & (df_trj['obj_type']==1)]
        left_veh_count,left_veh_list,right_veh_count,right_veh_list,straight_veh_count,straight_veh_list = get_left_veh_single_scenario(df)
        if left_veh_count > 5:
            dic_single_left_scenario['segment_id'] = segment_id
            dic_single_left_scenario['scenario_id'] = label
            dic_single_left_scenario['turn_left_veh_num'] = left_veh_count
            dic_single_left_scenario['left_veh_id'] = left_veh_list
            all_left_scenario.append(dic_single_left_scenario)

    return all_left_scenario

def scenario_extract_v3(df_trj,df_obj,all_left_scenario,file_index,test_state,test_scenario):  #左转+直行的冲突场景筛选(真交互）

    scenario_label_list = pd.unique(df_trj['scenario_label'].tolist())
    # print(scenario_label_list)
    interest_od_dict = get_interest_od(df_obj)
    dic_heading = {}
    for i in range(len(scenario_label_list)):
        try:
            dic_single_left_scenario = {}
            flag_dis = 0
            flag_angle = 0
            label = scenario_label_list[i]
            if test_state == 1:
                if label < test_scenario:
                    continue
                if label > test_scenario:
                    break
            print("scenario is {}".format(label))
            interest_od = interest_od_dict[label]
            df = df_trj[(df_trj['scenario_label'] == label) & (df_trj['valid'] == True) & (df_trj['obj_type']==1)]

            df_veh1, df_veh2 = process_data(df, interest_od)
            flag_dis, dis_each_min, time_stamp_min_dis, time_range_interactive, loc_veh_1_min_dis, loc_veh_2_min_dis = judge_dis(df_veh1, df_veh2)
            flag_angle, left_flag, angle_index_list, angle_veh_1, angle_veh_2, left_veh_id,straight_veh_id = judge_angle(df_veh1, df_veh2,df)
            left_veh_count, left_veh_list, right_veh_count, right_veh_list, straight_veh_count, straight_veh_list = get_left_veh_single_scenario(
                df)
            #print('straight_id {}'.format(straight_veh_id))
            straight_veh_list_direction_1,straight_veh_list_direction_2 = get_diff_direction_veh(straight_veh_id,straight_veh_list,df)  #直行车辆在十字交叉口有四种流向，根据车道斜率分为两大类,straight_veh_list_direction_1为interest veh所在的方向
            #print(left_veh_count, left_veh_list, right_veh_count, right_veh_list, straight_veh_count, straight_veh_list)
            #print('aa {}'.format(straight_veh_list_direction_1))
            if flag_dis == 1 and flag_angle == 1 and left_flag == 1:
                real_left_veh_list = get_real_type_veh(left_veh_id,left_veh_list,time_stamp_min_dis,df)  #得到该交叉口、该冲突事件中的所有左转车辆
                real_straight_veh_list = get_real_type_veh(straight_veh_id,straight_veh_list_direction_1,time_stamp_min_dis,df) #得到该交叉口、该冲突事件中的所有直行车辆
                print("real_veh:{},{},{},{}".format(len(real_left_veh_list), real_left_veh_list,len(real_straight_veh_list), real_straight_veh_list))
                print('该场景下目标交叉口左转车辆共{}辆，直行车辆共{}辆'.format(len(real_left_veh_list),len(real_straight_veh_list)))

                conflict_point = get_conflict_point(left_veh_id,straight_veh_id,df)   #得到轨迹冲突点
                print(conflict_point)
                time_list_left_veh_passing,left_veh_real_passing,target_left_veh_time_stamp_passing = get_time_stamp_veh_passing_by(real_left_veh_list,conflict_point,df,left_veh_id)  #得到车辆经过冲突点的时刻,targrt_veh为主交互对象
                time_list_straight_veh_passing,straight_veh_real_passing,target_straight_veh_time_stamp_passing = get_time_stamp_veh_passing_by(real_straight_veh_list,conflict_point,df,straight_veh_id)
                last_straight_veh_passing_time = max(time_list_straight_veh_passing)
                flag_interactive_truely = 0
                for time in time_list_left_veh_passing:
                    if last_straight_veh_passing_time > time:  #即左转车先于直行车通过，说明真正存在交互
                        flag_interactive_truely = 1
                print('flag_interactive_truely is {}'.format(flag_interactive_truely))
                if flag_interactive_truely == 1:
                    dic_single_left_scenario['segment_id'] = file_index
                    dic_single_left_scenario['scenario_id'] = label
                    dic_single_left_scenario['interactive_veh_1_id'] = interest_od[0]
                    dic_single_left_scenario['interactive_veh_2_id'] = interest_od[1]
                    dic_single_left_scenario['turn_left_veh_id'] = left_veh_id
                    dic_single_left_scenario['min_dis'] = dis_each_min
                    dic_single_left_scenario['time_stamp_min_dis'] = time_stamp_min_dis
                    dic_single_left_scenario['time_range_interactive'] = time_range_interactive
                    dic_single_left_scenario['loc_veh_1_interactive'] = loc_veh_1_min_dis
                    dic_single_left_scenario['loc_veh_2_interactive'] = loc_veh_2_min_dis
                    dic_single_left_scenario['intersection_center_loc'] = (
                        (loc_veh_1_min_dis[0] + loc_veh_2_min_dis[0]) / 2,
                        (loc_veh_1_min_dis[1] + loc_veh_2_min_dis[1]) / 2)  # 用两车的位置的平均值近似代替交叉口的中心位置
                    dic_single_left_scenario['angle_veh_1'] = angle_veh_1
                    dic_single_left_scenario['angle_veh_2'] = angle_veh_2
                    all_left_scenario.append(dic_single_left_scenario)
                    print(
                        '存在左转场景第{}个,两车OD为{}、{},最小距离为{:.2f}m,两车是航向角变化分别为{:.2f}°,{:.2f}°,'
                        'left_veh_id {}'.format(label,interest_od[0],interest_od[1],dis_each_min,angle_veh_1,angle_veh_2,left_veh_id))
        except:
            continue

    return all_left_scenario

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

def scenario_extract_straight_turn_left(data_file_list, data_file_index_list,interest_od_file_list):
    test_state = 1
    test_seg = 46
    test_scenario = 22
    all_left_scenario = []
    for i in tqdm(range(len(data_file_list))):
        file_index = data_file_index_list[i]
        seg_id = file_index_to_segment_id(file_index)
        if test_state == 1 :
            if seg_id< test_seg :
                continue
            if seg_id > test_seg :
                break
        print(test_seg)
        df_trj = pd.read_csv(data_file_list[i])
        df_obj = pd.read_csv(interest_od_file_list[i])
        #all_left_scenario = scenario_extract_v1(df_trj,df_obj,all_left_scenario,file_index)
        all_left_scenario = scenario_extract_v3(df_trj, df_obj, all_left_scenario, file_index, test_state,
                                                test_scenario)  #寻找真交互事件

    outpath = 'E:/Result_save/data_save/'
    outpath = 'data/Turn_left_scenario_test_2.csv'
    df_out = pd.DataFrame(all_left_scenario, index=None)  # 解决每列长度不一致的问题
    df_out.to_csv(outpath)

def scenario_extract_only_left(data_file_list, data_file_index_list,interest_od_file_list):  #只筛选存在左转车辆的

    all_left_scenario = []
    for i in tqdm(range(len(data_file_list))):
        segment_id = data_file_index_list[i]
        df_trj = pd.read_csv(data_file_list[i])
        df_obj = pd.read_csv(interest_od_file_list[i])
        all_left_scenario = scenario_extract_v2(df_trj,df_obj,all_left_scenario,segment_id)
    outpath = 'E:/Result_save/data_save/'
    outpath = 'data/Only_turn_left_scenario_test.csv'
    df_out = pd.DataFrame(all_left_scenario, index=None)  # 解决每列长度不一致的问题
    df_out.to_csv(outpath)
    print('Result has been printed')

def indiator_cal_all_seg(data_file_list, data_file_index_list,interest_od_file_list,df_turn_left_scenario,test_state,test_seg, test_scenario):

    indiator_all_seg_all_left_scenario = []
    for i in tqdm(range(len(data_file_list))):
        file_index = data_file_index_list[i]
        seg_id = file_index_to_segment_id(file_index)
        print('Cal indiator {}'.format(seg_id))
        if test_state == 1 :
            if seg_id< test_seg :
                continue
            if seg_id > test_seg :
                break
        #print(test_seg)
        scenario_label_list = df_turn_left_scenario[df_turn_left_scenario['segment_id']==seg_id]['scenario_id'].tolist()
        df_trj = pd.read_csv(data_file_list[i])
        df_obj = pd.read_csv(interest_od_file_list[i])

        indiator_result_single_seg = cal_indiator_single_scanario(df_trj, df_obj, scenario_label_list, seg_id, test_state,
                                     test_scenario)
        indiator_all_seg_all_left_scenario += indiator_result_single_seg
        # if indiator_result != {}:
        #     indiator_all_seg_all_left_scenario.append(indiator_result)
        # else:
        #     continue
    # print(indiator_all_seg_all_left_scenario)
    outpath = 'E:/Result_save/data_save/Indiator_cal_all.csv'
    # outpath = 'data/Indiator_cal_all_test.csv'
    df_out = pd.DataFrame(indiator_all_seg_all_left_scenario, index=None)  # 解决每列长度不一致的问题
    df_out.to_csv(outpath)

def get_passing_seq(left_veh_id, straight_veh_id,conflict_point,df):  #将车辆看做一维点计算PET，粗略计算
    passing_seq = -1

    _, _, left_veh_time_stamp_passing = get_time_stamp_veh_passing_by(
        [left_veh_id], conflict_point, df, left_veh_id)  # 得到左转车经过冲突点时的时刻
    _, _, straight_veh_time_stamp_passing = get_time_stamp_veh_passing_by([straight_veh_id], conflict_point, df, straight_veh_id) #直行车经过冲突点时的时刻
    if left_veh_time_stamp_passing < straight_veh_time_stamp_passing: #左转车先通过冲突点
        passing_seq = 1
    elif left_veh_time_stamp_passing > straight_veh_time_stamp_passing:
        passing_seq = 0
    PET = left_veh_time_stamp_passing-straight_veh_time_stamp_passing
    return passing_seq,PET


def cal_indiator_single_scanario(df_trj,df_obj,scenario_label_list,segment_id,test_state,test_scenario):  #计算左转场景中的所有技术指标
    indiator_all_left_scenario = []

    interest_od_dict = get_interest_od(df_obj)
    for i in range(len(scenario_label_list)):
        indiator_result = {}
        label = scenario_label_list[i]
        if test_state == 1:
            if label < test_scenario:
                continue
            if label > test_scenario:
                break
        # print("scenario is {}".format(label))
        interest_od = interest_od_dict[label]
        df = df_trj[(df_trj['scenario_label'] == label) & (df_trj['valid'] == True) & (df_trj['obj_type']==1)]

        df_veh1, df_veh2 = process_data(df, interest_od)  #得到目标交互车辆
        flag_angle, left_flag, angle_index_list, angle_veh_1, angle_veh_2, left_veh_id, straight_veh_id = judge_angle(
            df_veh1, df_veh2, df)
        df_left = df[df['obj_id'] == left_veh_id]
        df_stra = df[df['obj_id'] == straight_veh_id]
        # 车辆尺寸信息
        l_left, w_left = df_left['length'].iloc[0], df_left['width'].iloc[0]
        l_stra, w_stra = df_stra['length'].iloc[0], df_stra['length'].iloc[0]
        conflict_point = get_conflict_point(left_veh_id, straight_veh_id, df)  # 得到轨迹冲突点
        left_info, straight_info = get_lane_num_diff_direction(segment_id, label)
        if left_info != -1 and straight_info != -1 and conflict_point !=():
            passing_seq, PET_1dem = get_passing_seq(left_veh_id, straight_veh_id, conflict_point, df)  # 记录左转车和直行车的通行顺序，左转车先行则为1，直行车先行为0
            # PET, (l_left, w_left), (l_stra, w_stra) = cal_PET(left_veh_id, straight_veh_id,passing_seq,df)  #由于航向角数据有误，坐标系旋转误差较大，故暂时采用一维关系计算PET
            indiator_result['segment_id'] = segment_id
            indiator_result['scenario_label'] = label
            indiator_result['PET'] = round(PET_1dem,4)
            indiator_result['left_veh_first'] = passing_seq
            indiator_result['left_veh_start_in_place'] = judge_veh_start_in_place(left_veh_id,df) #判断车辆进入交叉口时是否为原地起步
            indiator_result['left_veh_length'],indiator_result['left_veh_width'] = l_left,w_left
            indiator_result['straight_veh_length'],indiator_result['straight_veh_width'] = l_stra,w_stra
            indiator_result['lane_num_left_in'],indiator_result['lane_width_left_in'],indiator_result['lane_id_left_in'] = left_info[0],left_info[1],left_info[2]
            indiator_result['lane_num_left_out'],indiator_result['lane_width_left_out'],interest_od_dict['lane_id_left_out'] = left_info[3],left_info[4],left_info[5]
            indiator_result['lane_num_straight_in'],indiator_result['lane_width_straight_in'],indiator_result['lane_id_straight_in'] = straight_info[0],straight_info[1],straight_info[2]
            indiator_result['lane_num_straight_out'],indiator_result['lane_width_straight_out'],indiator_result['lane_id_straight_out'] = straight_info[3],straight_info[4],straight_info[5]
            indiator_all_left_scenario.append(indiator_result)

    return indiator_all_left_scenario

def get_veh_tranforms(df):  #得到车辆基于航向角旋转之后的坐标信息

    df = df.copy()
    df['heading_180_new'] = df['heading'] * 180 / np.pi  # 航向角信息
    # x_A' = x_0 +(x_A-x_0) cosb - (y_A-y_0) sin b
    # y_A' = y_0 +(y_A-y_0) cosb + (x_A-y_0) sin b
    df['x_conflict'] = df['center_x'] + (df['x_conflict_ori'] - df['center_x']) * np.cos(
        df['heading_180_new']) - \
                       (df['y_conflict_ori'] - df['center_y']) * np.sin(df['heading_180_new'])
    df['y_conflict'] = df['center_y'] + (df['y_conflict_ori'] - df['center_y']) * np.cos(
        df['heading_180_new']) - \
                       (df['x_conflict_ori'] - df['center_x']) * np.sin(df['heading_180_new'])

    return df
def get_time_stamp_veh_passing_v2(df_veh,conflict_point):

    x_con, y_con = conflict_point[0], conflict_point[1]
    dis_min = 999
    index_min = -1
    for i in range(len(df_veh)):
        x, y = df_veh['x_conflict'].iloc[i], df_veh['y_conflict'].iloc[i]
        dis = np.sqrt((x - x_con) ** 2 + (y - y_con) ** 2)
        # print(dis)
        if dis < dis_min:
            dis_min = dis
            index_min = i
    time_passing = df_veh['time_stamp'].iloc[index_min]
    return time_passing

def output_file(df,path2):  #输出文件测试
    path1 = 'data/test_file/'
    path = path1 + path2 + '.csv'
    df.to_csv(path)
    print('test_file has been printed')

def cal_PET(left_veh_id, straight_veh_id,passing_seq,df):
    df = df.copy()
    PET = -1
    df_left = df[df['obj_id']==left_veh_id]
    df_stra = df[df['obj_id']==straight_veh_id]
    #车辆尺寸信息
    l_left,w_left = df_left['length'].iloc[0],df_left['width'].iloc[0]
    l_stra,w_stra = df_stra['length'].iloc[0],df_stra['length'].iloc[0]
    #定义A点为先通过冲突点的车辆的位置点，B点为后通过车辆的
    if passing_seq == 0: #直行车先行
        df_stra['x_conflict_ori'] = df_stra['center_x']-0.5*l_stra   #未旋转之前的坐标信息 x_A
        df_stra['y_conflict_ori'] = df_stra['center_y']+0.5*w_stra
        df_left['x_conflict_ori'] = df_left['center_x'] + 0.5 * l_left  # 未旋转之前的坐标信息 x_B
        df_left['y_conflict_ori'] = df_left['center_y'] + 0.5 * w_left
        df_stra = get_veh_tranforms(df_stra)
        df_left = get_veh_tranforms(df_left)
    elif passing_seq == 1 : #左转车先行
        df_left['x_conflict_ori'] = df_left['center_x'] - 0.5 * l_left  # 未旋转之前的坐标信息 x_A
        df_left['y_conflict_ori'] = df_left['center_y'] - 0.5 * w_left
        df_stra['x_conflict_ori'] = df_stra['center_x'] + 0.5 * l_stra  # 未旋转之前的坐标信息 x_B
        df_stra['y_conflict_ori'] = df_stra['center_y'] - 0.5 * w_stra
        df_stra = get_veh_tranforms(df_stra)
        df_left = get_veh_tranforms(df_left)

    output_file(df_left,'df_left')
    output_file(df_stra,'df_stra')
    veh_left_trj = np.column_stack((df_left['x_conflict'], df_left['y_conflict']))  # 得到车辆的轨迹信息
    veh_straight_trj = np.column_stack((df_stra['x_conflict'], df_stra['y_conflict']))
    line_left = LineString(veh_left_trj)  # 将车辆轨迹转化为shapely对象
    line_straight = LineString(veh_straight_trj)
    interscetion = line_left.intersection(line_straight)  # 得到轨迹交点，即冲突点
    print('intersection {}'.format(interscetion))
    conflict_point = (interscetion.xy[0][0], interscetion.xy[1][0])
    time_left = get_time_stamp_veh_passing_v2(df_left,conflict_point)
    time_stra = get_time_stamp_veh_passing_v2(df_stra,conflict_point)
    PET = time_left - time_stra #大于0为直行先走，小于0为左转先走

    return PET,(l_left,w_left),(l_stra,w_stra)

def judge_veh_start_in_place(left_veh_id,df):
    flag = 0
    df_veh = df[df['obj_id']==left_veh_id]
    v_x = pd.unique(df_veh['velocity_x'].tolist())
    v_y = pd.unique(df_veh['velocity_y'].tolist())
    if (0 in v_x) and (0 in v_y):
        flag = 1
    return flag

def get_lane_num_diff_direction(segment_id,label):
    lane_num_in,lane_num_out = -1,-1
    file_path_inter_info = 'data/veh_lane_info_test.csv'
    file_path_inter_info = 'E:/Result_save/data_save/veh_lane_info_all.csv'
    df = pd.read_csv(file_path_inter_info)
    df_1 = df[(df['segment_id']==segment_id)&(df['scenario_id']==label)]
    if len(df_1) == 0:
        return -1,-1

    lane_num_left_in, lane_width_left_in, lane_id_left_in, lane_num_left_out, lane_width_left_out, \
    lane_id_left_out = float(df_1['lane_num_left_in']),float(df_1['lane_width_left_in']),float(df_1['lane_id_left_in']),float(df_1['lane_num_left_out']),float(df_1['lane_width_left_out']),float(df_1['lane_id_left_out'])

    left_info = [lane_num_left_in, lane_width_left_in, lane_id_left_in, lane_num_left_out, lane_width_left_out, lane_id_left_out]
    lane_num_straight_in, lane_width_straight_in, lane_id_straight_in, lane_num_straight_out, lane_width_straight_out, lane_id_straight_out\
        = float(df_1['lane_num_straight_in']),float(df_1['lane_width_straight_in']),float(df_1['lane_id_straight_in']),float(df_1['lane_num_straight_out']),float(df_1['lane_width_straight_out']),float(df_1['lane_id_straight_out'])
    straight_info = [lane_num_straight_in, lane_width_straight_in, lane_id_straight_in, lane_num_straight_out, lane_width_straight_out, lane_id_straight_out]
    return left_info,straight_info


