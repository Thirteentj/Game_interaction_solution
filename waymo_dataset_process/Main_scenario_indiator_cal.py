import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import Scenario_extract as SExt
import Extract01 as Ext

# print(Ext.file_index_to_segment_id('00213'))

if __name__ == '__main__':

    filepath1 = 'D:/Data/左转视频筛选/all_scenario_all_objects_info/' + '*_all_scenario_all_object_info_1.csv'
    filepath2 = 'D:/Data/左转视频筛选/objects_of_interest_info/' + '*_all_scenario_objects_of_interest_info.csv'
    filepath3 = 'D:/Data/Git/waymo-od/data/all_lane_topo_info/*_all_seg_all_lane_topo_info.csv'
    data_file_list, data_file_index_list = SExt.get_file_list(filepath1)
    interest_od_file_list, interest_od_index_list = SExt.get_file_list(filepath2)
    lane_topo_file_list,lane_topo_file_index_list = SExt.get_file_list(filepath3)
    filepath_turn_left_scenario = 'D:/Data/Git/waymo-od/data/Turn_left_scenario_test.csv'
    df_turn_left_scenario = pd.read_csv(filepath_turn_left_scenario)
    filepath_all_intersection_info = 'D:/Data/Git/waymo-od/data/all_lane_topo_info/0_50_all_intersection_info.csv'
    df_all_intersection_info = pd.read_csv(filepath_all_intersection_info)

    filepath_oridata = 'D:/Data/WaymoData_motion_1/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = Ext.get_file_list(filepath_oridata)
    test_state = 0
    test_seg, test_scenario = 46,22

    if data_file_index_list != interest_od_index_list:
        print("The file isn't match")
    else:
        print("Data has been loaded")
    # Ext.get_lane_num_diff_direction_all_seg(df_turn_left_scenario,df_all_intersection_info, interest_od_file_list,data_file_list, data_file_index_list, all_file_list, file_index_list,test_state,test_seg,test_scenario)
    SExt.indiator_cal_all_seg(data_file_list, data_file_index_list,interest_od_file_list,df_turn_left_scenario,test_state,test_seg, test_scenario)
    #SExt.scenario_extract_only_left(data_file_list, data_file_index_list, interest_od_file_list)

