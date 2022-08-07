import numpy as np
import pandas as pd
from tqdm import  tqdm
from shapely.geometry import LineString


'''
基于轨迹数据计算指标：PET、加速度
To 唐揽月师姐
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
    time_passing = df_veh['time_stamp'].iloc[index_min]
    return time_passing

def get_conflict_point(left_veh_id,straight_veh_id,df):  #基于左转和直行交互车辆的轨迹得到冲突点坐标

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


fpath='data_save\Real_Turn_left_scenario_0307V2.xlsx'
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
    filepath_trj = 'data_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
    seg_trj = pd.read_csv(filepath_trj)
    seg_trj = seg_trj[(seg_trj['valid'] == True) &(seg_trj['scenario_label'] == scenario_id)]  #目标场景的所有轨迹

    veh_left_id = int(df_left['turn_left_veh_id'].iloc[i])
    veh_straight_front_id = int(df_left['veh_straight_front'].iloc[i])
    veh_straight_later_id = int(df_left['veh_straight_later'].iloc[i])

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






