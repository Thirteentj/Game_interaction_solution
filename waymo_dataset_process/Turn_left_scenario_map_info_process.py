from tqdm import tqdm
import pandas as pd


def seg_id_2_fileindex(segment_id):
    file_index = -1
    if segment_id == 0:
        file_index = '00000'
    elif 0< segment_id < 9:
        file_index = '0000' + str(segment_id)
    elif 10<=segment_id < 99:
        file_index = '000' + str(segment_id)
    elif 100<=segment_id<999:
        file_index = '00' + str(segment_id)

    return file_index
def read_map_file(seg_id,scenario_id):
    file_index = seg_id_2_fileindex(seg_id)
    filepath = f'G:/Map_info/static_map_point_info/{file_index}_all_scenario_static_map_info.csv'
    df_map = pd.read_csv(filepath)
    # print('文件已读取')
    df_map = df_map[(df_map['file_index']==seg_id)&(df_map['scenario_label']==scenario_id)]
    return df_map

#读取左转事件及其场景
filepath2 = r'F:\Result_save\data_save\Turn_left_scenario_info_0419.xlsx'  # 共194个事件
df_all = pd.read_excel(filepath2)

for i in tqdm(range(len(df_all))):
    seg_id = df_all['segment_id'].iloc[i]
    scenario_id = df_all['scenario_id'].iloc[i]
    print(f'seg_id:{seg_id},scenario_id:{scenario_id}')
    df_map_i = read_map_file(seg_id,scenario_id)
    #一个scenario保存一个csv文件
    out_path = f'G:/Map_info/Turn_left_map_info/Turn_left_scenario_map_info_seg_{seg_id}_scenario_{scenario_id}.csv'
    df_map_i.to_csv(out_path)
    print(f'seg_id:{seg_id},scenario_id:{scenario_id}的地图数据已经提取处理完毕')
    