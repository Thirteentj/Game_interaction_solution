import os
import cv2
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#该版本为测试版本，计算使用的所有参数仅供调试

class veh():
    def __init__(self,id,loc,v_xy,G=0.01,M=4000,D_r1=0.2,D_r2=0.6,k1=0.2,k2=0.05,R1=1,R2=1):
        self.G = G
        self.id = id
        self.M = M
        self.loc = loc
        self.D_r1,self.D_r2 = D_r1,D_r2
        self.k1,self.k2 = k1,k2
        self.R1,self.R2 = R1,R2
        self.vx = v_xy[0]
        self.vy = v_xy[1]
        self.v = np.sqrt(self.vx**2+self.vy**2)

class para_temp_init():
    def __init__(self):
        self.para_dict = {}
        self.G = 0.01
        self.M2 = 4000
        self.k1, self.k2 = 0.2, 0.05
        self.R1, self.R2 = 1, 1
        self.D_r1, self.D_r2 = 0.2, 0.6

class map_para_init():
    def __init__(self,x_start,x_end,y_start,y_end):
        #地图范围
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        #x、y方向的分割粒度
        self.x_step = 2*int(abs((self.x_end-self.x_start)))
        self.y_step = 2*int(abs((self.y_end-self.y_start)))
        self.xx = np.linspace(self.x_start, self.x_end, self.x_step)
        self.yy = np.linspace(self.y_start, self.y_end, self.y_step)
    def map_size(self):
        #绘图尺寸
        self.width = 20*(self.x_end-self.x_start)
        self.height = 20*(self.y_end-self.y_start)
        return self.height,self.width
    def map_meshgrid(self):

        self.X ,self.Y = np.meshgrid(self.xx,self.yy)
        return self.X,self.Y

def cal_E_V_single(veh_i,map_para):
    x_i,y_i = veh_i.loc
    X, Y = map_para.map_meshgrid()
    r_v = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
    G,R2,M2,k1,k2 = para_temp.G,para_temp.R2,para_temp.M2,para_temp.k1,para_temp.k2
    v_i = veh_i.v
    E_V = G * R2 * M2 / pow(r_v,k1) * np.exp(k2*v_i*((X-x_i)/(np.sqrt((X-x_i)**2+(Y-y_i)**2))))
    return E_V

def cal_E_R_single(veh_i,map_para):
    x_i,y_i = veh_i.loc
    X,Y = map_para.map_meshgrid()
    r = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
    G, R2, M2, k1, = para_temp.G,para_temp.R2,para_temp.M2,para_temp.k1
    E_R = G * R2 * M2 / pow(r,k1)
    return E_R

def cal_E_R_all(veh_list,map_para):  #计算所有车辆的势能场
    E_R = np.zeros((map_para.y_step, map_para.x_step))
    for veh_i in veh_list:
        E_R += cal_E_R_single(veh_i,map_para)
    return E_R

def cal_E_V_all(veh_list,map_para):  #计算所有车辆的运动场
    E_V = np.zeros((map_para.y_step, map_para.x_step))
    # print('first shape {}'.format(E_V.shape))
    for veh_i in veh_list:
        E_V += cal_E_V_single(veh_i,map_para)
    return E_V


def get_data():
    veh_list = []  # 记录全局所有车辆信息
    veh_1 = veh(1, (10, 5), (20,0))  # 初始化车辆对象
    veh_2 = veh(2, (15, 10), (18,0))  # 初始化车辆对象
    veh_3 = veh(3, (20, 5), (25,0))  # 初始化车辆对象
    veh_list.append(veh_1)
    veh_list.append(veh_2)
    veh_list.append(veh_3)
    map_para = map_para_init(0,50,0,15)
    return veh_list,map_para

def process_data(top_view_trj, frame_id):
    veh_list = []
    seg_trj_single = top_view_trj
    time_stamp = pd.unique(seg_trj_single['time_stamp'])
    all_veh_info = seg_trj_single[seg_trj_single['time_stamp'] == time_stamp[frame_id-1]]

    for i in range(len(all_veh_info)):
        veh_info = all_veh_info.iloc[i]
        v_id = veh_info.obj_id
        loc_x = veh_info.center_x
        loc_y = veh_info.center_y
        v_x = veh_info.velocity_x
        v_y = veh_info.velocity_y
        veh_i = veh(v_id, (loc_x, loc_y), (v_x, v_y))
        veh_list.append(veh_i)

    map_x_min, map_x_max = seg_trj_single['center_x'].min(), seg_trj_single['center_x'].max()
    map_y_min, map_y_max = seg_trj_single['center_y'].min(), seg_trj_single['center_y'].max()
    map_para = map_para_init(map_x_min, map_x_max, map_y_min, map_y_max)
    return veh_list, map_para


