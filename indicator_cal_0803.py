'''
20220803  计算仙霞剑河数据集、waymo数据集左转和直行的PET
'''
import numpy as np
import pandas as pd
from tqdm import  tqdm
from shapely.geometry import LineString
import datetime

'''
基于轨迹数据计算指标：PET、加速度
'''
def get_time_stamp_veh_passing(df_veh,conflict_point):

    x_con, y_con = conflict_point[0], conflict_point[1]
    dis_min = 999
    index_min = -1
    for i in range(len(df_veh)):
        x,y = df_veh['center_x'].iloc[i],df_veh['center_y'].iloc[i]
        dis = np.sqrt((x - x_con) ** 2 + (y - y_con) ** 2)
        # print(dis_plan)
        if dis < dis_min:
            dis_min = dis
            index_min = i
    if index_min != -1:
        time_passing = df_veh['time_stamp'].iloc[index_min]
    else:
        time_passing = 0
        print('经过时间点出现异常')
    return time_passing

def get_conflict_point(left_veh_id,straight_veh_id,df):  #基于左转和直行交互车辆的轨迹得到冲突点坐标

    try:
        df_veh_left = df[df['obj_id']==left_veh_id]
        veh_left_trj = np.column_stack((df_veh_left['center_x'],df_veh_left['center_y']))  #得到车辆的轨迹信息
        df_veh_straight = df[df['obj_id']==straight_veh_id]
        veh_straight_trj = np.column_stack((df_veh_straight['center_x'],df_veh_straight['center_y']))
        line_left = LineString(veh_left_trj)  #将车辆轨迹转化为shapely对象
        line_straight = LineString(veh_straight_trj)
        interscetion = line_left.intersection(line_straight)  #得到轨迹交点，即冲突点
    # print('intesection {}'.format(interscetion))
        conflict_point = (interscetion.xy[0][0], interscetion.xy[1][0])
    except:
        conflict_point = ()
    return conflict_point

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

def acceleration_cal(df_veh):  #计算加速度指标,df_veh是车辆的轨迹数据集 a=x/t
    # 计算左车、直行车下一时刻的加速度ax,ay
    for i in range(len(df_veh) - 1):
        label = df_veh['frame_label'].iloc[i]
        # print(label)
        df_veh.loc[df_veh['frame_label'] == label, 'ax_next'] = (df_veh['velocity_x'].iloc[i + 1] - df_veh['velocity_x'].iloc[
            i]) \
                                                                / (df_veh['time_stamp'].iloc[i + 1] - df_veh['time_stamp'].iloc[
            i])
        df_veh.loc[df_veh['frame_label'] == label, 'ay_next'] = (df_veh['velocity_y'].iloc[i + 1] - df_veh['velocity_y'].iloc[
            i]) \
                                                                / (df_veh['time_stamp'].iloc[i + 1] - df_veh['time_stamp'].iloc[
            i])


def PET_cal_waymo_old():
    fpath= r'F:\Result_save\data_save\Turn_left_scenario_info_0419.xlsx'
    df_1= pd.read_excel(fpath)
    df_2 = df_1.copy()
    df_left = df_2.copy()

    df_left.insert(5, 'PET_strai_front_to_left',-1 )
    df_left.insert(6, 'PET_left_to_strai_later', -1)
    df_left.insert(7, 'Gap_straight', -1)  #直行车流的穿行间距
    for i in tqdm(range(len(df_left))):

        seg_id = df_left['segment_id'].iloc[i]
        scenario_id = df_left['scenario_id'].iloc[i]
        file_index = segment_id_to_file_index(seg_id)
        filepath_trj = 'F:/Result_savedata_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
        seg_trj = pd.read_csv(filepath_trj)
        seg_trj = seg_trj[(seg_trj['valid'] == True) &(seg_trj['scenario_label'] == scenario_id)]  #目标场景的所有轨迹

        veh_left_id = int(df_left['turn_left_veh_id'].iloc[i])
        veh_straight_front_id = int(df_left['veh_straight_front'].iloc[i])
        veh_straight_later_id = int(df_left['veh_straight_later'].iloc[i])
        print(f'segment_id:{seg_id},scenario_id:{scenario_id},左转车id:{veh_left_id},直行前车id:{veh_straight_front_id},直行后车id:{veh_straight_later_id}')
        conflict_point = get_conflict_point(veh_left_id,veh_straight_front_id,seg_trj)  #得到轨迹冲突点

        time_left_veh_passing = get_time_stamp_veh_passing(seg_trj[seg_trj['obj_id']==veh_left_id],conflict_point)
        time_straight_veh_front_passing = get_time_stamp_veh_passing(seg_trj[seg_trj['obj_id'] == veh_straight_front_id],
                                                                   conflict_point)
        time_straight_veh_later_passing = get_time_stamp_veh_passing(
            seg_trj[seg_trj['obj_id'] == veh_straight_later_id],conflict_point)

        df_left['PET_strai_front_to_left'].iloc[i] = time_left_veh_passing- time_straight_veh_front_passing  #计算PET（直行车先行的情况）
        df_left['PET_left_to_strai_later'].iloc[i] = time_straight_veh_later_passing - time_left_veh_passing #计算PET（左转车先行的情况）
        df_left['Gap_straight'].iloc[i] = time_straight_veh_later_passing-time_straight_veh_front_passing  #直行车通过间隙


    tpath = '/data_save/all_real_left_scenario_result.csv'
    df_left.to_csv(tpath)

def PET_cal_waymo():
    start = datetime.datetime.now()
    #进行数据筛选（因为有一部分scenario的数据是计算博弈的时候没法用的，根据博弈的计算结果，将这部分数据剔除掉
    file_path = './result/waymo_scoring_result_1_cal_type1_range_0_194_u_type_risk.csv'
    df_ori = pd.read_csv(file_path)
    filter_or_not = 1  # 1 为筛选，0为不筛选

    fpath = r'F:\Result_save\data_save\Turn_left_scenario_info_0419.xlsx'
    df_left = pd.read_excel(fpath)
    list_result = []
    for i in tqdm(range(len(df_left))):
        result_set = {}
        seg_id = df_left['segment_id'].iloc[i]
        scenario_id = df_left['scenario_id'].iloc[i]
        t = df_ori[(df_ori['segment_id']==seg_id) & (df_ori['scenario_id']==scenario_id)]
        if filter_or_not==0 or len(t)>0:
            print('满足条件')
            file_index = segment_id_to_file_index(seg_id)
            filepath_trj = 'F:/Result_save/data_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
            seg_trj = pd.read_csv(filepath_trj)
            seg_trj = seg_trj[(seg_trj['valid'] == True) & (seg_trj['scenario_label'] == scenario_id)]  # 目标场景的所有轨迹

            veh_left_id = int(df_left['turn_left_veh_id'].iloc[i])
            veh_straight_front_id = int(df_left['veh_straight_front'].iloc[i])
            veh_straight_later_id = int(df_left['veh_straight_later'].iloc[i])
            # print(f'segment_id:{seg_id},scenario_id:{scenario_id},左转车id:{veh_left_id},直行前车id:{veh_straight_front_id},直行后车id:{veh_straight_later_id}')
            conflict_point = get_conflict_point(veh_left_id, veh_straight_front_id, seg_trj)  # 得到轨迹冲突点
            if conflict_point == ():
                print('无交互冲突点')
            else:
                time_left_veh_passing = get_time_stamp_veh_passing(seg_trj[seg_trj['obj_id'] == veh_left_id], conflict_point)
                time_straight_veh_front_passing = get_time_stamp_veh_passing(
                    seg_trj[seg_trj['obj_id'] == veh_straight_front_id],
                    conflict_point)
                time_straight_veh_later_passing = get_time_stamp_veh_passing(
                    seg_trj[seg_trj['obj_id'] == veh_straight_later_id], conflict_point)

                result_set['segment_id'] = seg_id
                result_set['scenario_id'] = scenario_id
                result_set['direction_veh'] = '_'
                result_set['left_veh_id'] = veh_left_id
                result_set['veh_straight_front_id'] = veh_straight_front_id
                result_set['PET_straight_front'] = time_left_veh_passing - time_straight_veh_front_passing

                result_set['veh_straight_later_id'] = veh_straight_later_id
                result_set['PET_straight_later'] = time_straight_veh_later_passing - time_left_veh_passing
                list_result.append(result_set)

    df_result = pd.DataFrame(list_result)
    df_result.to_csv('./result/PET/PET_result_waymo.csv')

    end = datetime.datetime.now()
    print(f'程序计算用时{end - start}')

def PET_cal_xianxia():
    start = datetime.datetime.now()
    # 数据筛选 如果不需要，注释掉下面两行
    file_path = './result/xianxia_scoring_result_cal_type_1_all_u_type_PET.xlsx'
    df_ori = pd.read_excel(file_path)
    filter_or_not = 1  #1 为筛选，0为不筛选

    filepth2 = r'veh_all_info_tra_all.xlsx'  # 交叉口西进口左转车的轨迹数据
    df_all_info_all = pd.read_excel(filepth2)
    list_result = []
    for segment_id in [0, 1]:
        df_all_info = df_all_info_all[df_all_info_all['segment_index'] == segment_id]
        num_scenario = len(pd.unique(df_all_info['scenario_label']))  # 交互行为及场景数量
        for k in range(num_scenario):
            t = df_ori[(df_ori['segment_id']==segment_id) & (df_ori['scenario_id']==k)]
            if filter_or_not==0 or len(t)>0:
                print('可以计算')
                result_set = {}
                if segment_id == 0:
                    direction_veh = 'west'
                elif segment_id == 1:
                    direction_veh = 'south'
                print(f'segment:{segment_id},scenario_label:{k}')
                df_L_ori = df_all_info[(df_all_info['scenario_label'] == k) & (df_all_info['action_type'] == 'left')]
                df_S_ori = df_all_info[(df_all_info['scenario_label'] == k) & (df_all_info['action_type'] == 'straight')]
                seg_trj = df_all_info[df_all_info['scenario_label'] == k]
                veh_left_id = pd.unique(df_L_ori['obj_id'])[0]
                veh_stra_id = pd.unique(df_S_ori['obj_id'])[0]
                conflict_point = get_conflict_point(veh_left_id,veh_stra_id,seg_trj)  #得到轨迹冲突点
                if conflict_point == ():
                    print('无交互冲突点')
                else:
                    time_left_veh_passing = get_time_stamp_veh_passing(df_L_ori,conflict_point)
                    time_straight_veh_passing = get_time_stamp_veh_passing(df_S_ori,conflict_point)
                    PET = time_straight_veh_passing - time_left_veh_passing  #左转先过PET为正，左转后过为负
                    PET_abs = abs(PET)
                    result_set['segment_id'] = 0
                    result_set['scenario_id'] = k
                    result_set['direction_veh'] = direction_veh
                    result_set['left_veh_id'] = veh_left_id
                    result_set['stra_veh_id'] = veh_stra_id
                    result_set['PET'] = PET
                    result_set['PET_abs'] = PET_abs
                    list_result.append(result_set)
    df_result = pd.DataFrame(list_result)
    df_result.to_csv('./result/PET/PET_result_xianxia.csv')

    end = datetime.datetime.now()
    print(f'程序计算用时{end - start}')

if __name__ == '__main__':
    # PET_cal_xianxia()
    PET_cal_waymo()