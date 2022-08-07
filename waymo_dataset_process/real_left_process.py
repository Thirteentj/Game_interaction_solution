import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import Extract01 as Ext
import Scenario_extract as SExt
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import  tqdm

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

fpath='E:\Result_save\data_save\Real_Turn_left_scenario_0307V2.xlsx'
df_1= pd.read_excel(fpath)
df_2 = df_1.copy()
df_left = df_2.copy()
df_left.insert(5, 'PET_strai_front_to_left',-1 )
df_left.insert(6, 'PET_left_to_strai_later', -1)
df_left.insert(7, 'Gap_straight', -1)  #直行车流的穿行间距
df_left.insert(8, 'time_left_veh_passing', -1)
df_left.insert(9, 'time_straight_veh_front_passing', -1)
df_left.insert(10, 'time_straight_veh_later_passing', -1)
for i in tqdm(range(len(df_left))):
    try:
        seg_id = df_left['segment_id'].iloc[i]
        scenario_id = df_left['scenario_id'].iloc[i]
        file_index = Ext.segment_id_to_file_index(seg_id)
        filepath_trj = 'E:/Result_save/data_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
        seg_trj = pd.read_csv(filepath_trj)
        seg_trj = seg_trj[(seg_trj['valid'] == True) &(seg_trj['scenario_label'] == scenario_id)]  #目标场景的所有轨迹

        veh_left_id = int(df_left['turn_left_veh_id'].iloc[i])
        veh_straight_front_id = int(df_left['veh_straight_front'].iloc[i])
        veh_straight_later_id = int(df_left['veh_straight_later'].iloc[i])

        conflict_point = SExt.get_conflict_point(veh_left_id,veh_straight_front_id,seg_trj)  #得到轨迹冲突点

        time_left_veh_passing = get_time_stamp_veh_passing(seg_trj[seg_trj['obj_id']==veh_left_id],conflict_point)
        time_straight_veh_front_passing = get_time_stamp_veh_passing(seg_trj[seg_trj['obj_id'] == veh_straight_front_id],
                                                                   conflict_point)
        time_straight_veh_later_passing = get_time_stamp_veh_passing(
            seg_trj[seg_trj['obj_id'] == veh_straight_later_id],conflict_point)

        df_left['PET_strai_front_to_left'].iloc[i] = time_left_veh_passing- time_straight_veh_front_passing  #计算PET（直行车先行的情况）
        df_left['PET_left_to_strai_later'].iloc[i] = time_straight_veh_later_passing - time_left_veh_passing #计算PET（左转车先行的情况）
        df_left['Gap_straight'].iloc[i] = time_straight_veh_later_passing-time_straight_veh_front_passing  #直行车通过间隙

        df_left['time_left_veh_passing'].iloc[i] = time_left_veh_passing
        df_left['time_straight_veh_front_passing'].iloc[i] = time_straight_veh_front_passing
        df_left['time_straight_veh_later_passing'].iloc[i] = time_straight_veh_later_passing
    except:
        continue


print(df_left)
tpath = 'E:/Result_save/data_save/all_real_left_scenario_result.csv'
df_left.to_csv(tpath)





