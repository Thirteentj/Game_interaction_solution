import math
import datetime
from shapely.geometry import LineString
import numpy as np
import pandas as pd
import Potential_Field_V2 as PF  #V2 和V3版本应该分开
import bezier_path as bezier
# import game_video_generate as vidoe_gen
#V2版本重新规划交互对象的轨迹，并计算出交互对象的实际状态，并对比实际动力学状态和规划的运动状态的差距
#V1版本不对实际轨迹进行规划，只是在车辆的实际轨迹（每个时间步的实际状态）的基础上计算出下一时刻的加速度等信息，并不对车辆状态进行改变
class veh():
    def __init__(self):
        self.id = -1  #车辆的ID信息
        self.size = ()  #车辆的尺寸l：长度，w：宽度
        self.index = []
        self.time_stamp = -1
        #车辆实际过程参数
        self.time_interactive = -1 #交互对象的总交互时长
        self.time_stamp_list_real = []  #交互对象实际的时间序列
        #xy坐标系中的数据
        self.start_point = () #车辆的交互过程起始点坐标
        self.end_point = ()  #车辆交互过程的终点坐标
        self.start_yaw = np.radians(180)  #开始时刻的航向角   需要单独计算
        self.end_yaw = np.radians(270)  #结束时刻的航向角
        self.tra_real_xy = []  #交互车辆每个时间步长的轨迹坐标
        self.tra_real_s = []  #自然坐标系下的轨迹坐标
        self.tra_real_s_sum = []
        self.vx_real = []  #交互车辆每个时间步长的x方向速度
        self.vy_real = []
        self.ax_real = []  # 交互车辆每个时间步长的x方向加速度
        self.ay_real = []
        #自然坐标系中的数据
        self.v_real = []  # 交互车辆每个时间步长的实际速度（也是自然坐标系中车辆的实际速度）
        self.a_real = []  # 交互车辆每个时间步长的实际加速度（也是自然坐标系中车辆的实际加速度）
        self.theta_real = []  # 所有时间步加速度的角度值

        self.dis_real = ()  #交互车辆每个时间步长和冲突点之间的距离
        self.conflict_point_real = ()
        self.conflict_point_real_index = -1
        #博弈过程记录
        self.time_stamp_list_game = []
        self.a_plan = []  #所有时间步的基于博弈得到的加速度集合(自然坐标系）
        self.theta_plan = []  # 所有时间步加速度的角度值
        self.dis_plan = []  #所有时间步距离冲突点的距离集合
        self.loc_x_plan = []  #所有时间步的x坐标信息
        self.loc_Y_plan = []  #所有时间步的y坐标信息
        self.v = [] #所有时间步的速度集合
        self.vx_paln = []  # 所有时间步的x速度集合
        self.vy_paln = []  # 所有时间步的y速度集合
        self.s = [] #所有时间步的策略集合
        self.s_all = ['s1','s2','s3'] #车辆的博弈策略集合
        self.s_set = {'s1':{'a_plan':0.5},'s2':{'a_plan':0},'s3':{'a_plan':-0.5}}  #博弈策略的具体参数
        self.risk_level = ()
        self.tra_plan_xy = []  #博弈车辆根据起终点初步规划出来的预期轨迹（xy坐标系）
        self.tra_plan_s = []  #博弈车辆根据终点规划的轨迹（自然坐标系）
        self.tra_plan_s_sum = []  #博弈车辆自然坐标系下的累积距离
        self.payoff = []  #博弈每个时间步，车辆自身的综合收益
        self.conflict_point_plan = ()
        self.conflict_point_plan_index = -1



def get_payoff_coefficient(P,risk_level):  #得到风险感知系数
    a,b = risk_level
    # get m
    aa = 1-0.5*P/a
    # print(P,a,b)
    if P<a:
        m = 1-0.5*P/a
    elif a<=P<=b:
        m = 0.5
    elif b<P:
        m = 0.5*(1-P)*(1-b)
    #get n
    n = 1 - m
    return m,n

def get_motion_state(dis_now,v_now,s,dt,s_set):  #得到每个时间步的车辆运动状态  车辆运动状态可以用列表或者集合存储

    # s_set = {'s1':{'a_next':1.5},'s2':{'a_next':0},'s3':{'a_next':-1.5}}  #策略集合
    a_next = s_set[s]['a_plan']
    v_next = v_now + a_next * dt
    l_next = dis_now - v_now * dt - 0.5 * a_next * dt ** 2
    dv = a_next * dt

    return l_next,v_next,a_next,dv

def get_plan_trajectory_veh(veh_i,type = 'left'):  #对于左转车辆，使用贝塞尔曲线进行路径拟合，对于直行车辆，尽量使用直线
    #这里的轨迹，可以考虑使用左转车道中心线将问题简化
    start_point = veh_i.start_point
    end_point = veh_i.end_point
    start_yaw = veh_i.start_yaw
    end_yaw = veh_i.end_yaw
    path_new = []
    if type == 'left':
        path = bezier.get_planning_path_bezier(start_point,start_yaw,end_point,end_yaw)
        #坐标存储格式转换

        path_x = [point[0]  for point in path]
        path_y = [point[1] for point in path]
        path_new.append(path_x)
        path_new.append(path_y)
    elif type == 'straight':
        path_x = np.linspace(start_point[0],end_point[0],100)
        path_y = np.linspace(start_point[1],end_point[1],100)
        path_new.append(path_x)
        path_new.append(path_y)

    #路径存储记录
    veh_i.tra_plan_xy = path_new   #存储形式[[x_list],[y_list]]
    return veh_i



def location_xy2_s(path_xy):  #xy坐标系转换到自然坐标系
    #已知二维坐标，转换为自然坐标

    path_len = [0]  #自然坐标系下的距离存储
    path_len_sum = [0]  # 自然坐标系下的累积距离存储
    path_dire = [0] #角度存储
    path_s = []
    path_s_sum = []
    for i in range(len(path_xy[0]) - 1):
        point_x,point_y = path_xy[0][i],path_xy[1][i]
        point_x_next, point_y_next = path_xy[0][i+1], path_xy[1][i+1]
        dx = point_x_next-point_x
        dy = point_y_next-point_y
        dis_interval = np.sqrt((dx)**2+(dy)**2)

        direction = lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx)  #计算转向角
        len_sum = dis_interval + path_len_sum[-1]  #计算累积长度
        # path_len.append(len_sum)   #这里暂时不记录累积距离，记录每个分段的距离
        path_len.append(dis_interval)
        path_dire.append(direction[1])
        path_len_sum.append(len_sum)

    path_s.append(path_len)
    path_s.append(path_dire)
    path_s_sum.append(path_len_sum)
    path_s_sum.append(path_dire)

    return path_s,path_s_sum

def location_s2_xy(start_point,path_s):   #自然坐标系转换到xy坐标系 已知自然坐标，转换到二维坐标
    path_len = path_s[0]  #自然坐标系下的距离存储
    path_dire = path_s[1]  #角度存储
    dx = path_len * np.cos(path_dire)
    dy = path_len * np.sin(path_dire)
    print('dx',dx,dy)
    x_point = []
    y_point = []
    xy_point = []
    for i in range(len(path_len)):
        if i == 0:
            dx_s = start_point[0] + dx[i]
            dy_s = start_point[1] + dy[i]
        else:
            dx_s = x_point[-1]+dx[i]
            dy_s = y_point[-1]+dy[i]
        x_point.append(dx_s)
        y_point.append(dy_s)
    xy_point.append(x_point)
    xy_point.append(y_point)
    return xy_point
def get_risk_point(Risk_matrix,conflict_point,map_para):
    risk = 0
    map_x,map_y = map_para.map_meshgrid()
    x_conflict,y_conflict = conflict_point[0],conflict_point[1]
    x_index,y_index = -1,-1 #寻找风险矩阵中冲突点位置坐标对应的索引
    dis_min = 9999
    for i in range(map_para.y_step):
        for j in range(map_para.x_step):
            dis_temp = np.sqrt((map_x[i][j]-x_conflict)**2 + (map_y[i][j]-y_conflict)**2)
            if dis_temp < dis_min:
                dis_x_temp = dis_min
                x_index,y_index = j,i
    risk = Risk_matrix[y_index,x_index]
    if risk > 1:
        risk = 1
    return risk

def get_motion_state_V2(dis_now,v_now,a_now,s,dt,s_set,risk):  #得到每个时间步的车辆运动状态  车辆运动状态可以用列表或者集合存储

    a_next = abs(((1-risk) + a_now)+np.random.uniform(0,0.2))
    if s == 's1':  #加速运动
        if a_next >1.5:
            a_next = 1.5
    elif s == 's2': #匀速运动
        a_next = 0
    elif s == 's3':  #减速运动
        a_next = - a_next
        if a_next <-1.5:
            a_next = -1.5

    v_next = v_now + a_next * dt
    l_next = dis_now - v_now * dt - 0.5 * a_next * dt ** 2
    dv = a_next * dt
    # print(f'策略为{s},加速度{a_next}')
    return l_next,v_next,a_next,dv

def get_Nash_equilibrium(index,veh_L,veh_S,df_all_veh,seg_trj_single,type=1):  #求解当前时间步的纳什均衡最优解
    time_now = veh_L.time_stamp_list_real[index]
    df_all_veh_now = df_all_veh[df_all_veh['time_stamp']==time_now]
    dis_L_now, v_L_now, a_L_now = veh_L.dis_plan[-1], veh_L.v_real[index], veh_L.a_real[index]
    dis_S_now, v_S_now, a_S_now = veh_S.dis_plan[-1], veh_S.v_real[index], veh_S.a_real[index]
    S_L_all = S_S_all = veh_L.s_all
    L_best = S_best = []
    payoff_max = payoff_left_best = payoff_stra_best = -100
    df_payoff_L = pd.DataFrame(columns=S_L_all, index=S_S_all)
    df_payoff_S = pd.DataFrame(columns=S_L_all, index=S_S_all)
    df_payoff_sum = pd.DataFrame(columns=S_L_all, index=S_S_all)
    # df_re = pd.DataFrame(columns=S_L_all, index=S_S_all)
    # df_re2 = pd.DataFrame(columns=S_L_all, index=S_S_all)
    for i in range(len(S_L_all)):
        for j in range(len(S_S_all)):
            s_L_temp, s_S_temp = S_L_all[i], S_S_all[j]
            #计算左车和直行车的状态和收益
            #使用-0.5,0,0.5对状态进行估计，实际的a需要再计算
            dis_L_next, v_L_next, a_L_next, dv_L = get_motion_state(dis_L_now, v_L_now, s_L_temp,veh_L.time_stamp,veh_L.s_set)
            dis_S_next, v_S_next, a_S_next, dv_S = get_motion_state(dis_S_now, v_S_now, s_S_temp,veh_S.time_stamp,veh_S.s_set)

            Risk_L, Risk_S, map_para = PF.get_risk_interactive_all(index,time_now, veh_L, veh_S, v_L_next, a_L_next,
                                                                                 v_S_next, a_S_next,seg_trj_single)  #得到感知风险
            # print(Risk_L)
            # Risk_L = get_risk_point(Risk_L_matrix, veh_L.conflict_point_real, map_para)
            # Risk_S = get_risk_point(Risk_S_matrix, veh_S.conflict_point_real, map_para)
            # print(f'左车的风险值为{Risk_L},直行车的风险值为{Risk_S}')
            #计算实际的状态
            dis_L_next, v_L_next, a_L_next, dv_L = get_motion_state_V2(dis_L_now, v_L_now,a_L_now, s_L_temp, veh_L.time_stamp,veh_L.s_set,Risk_L)
            dis_S_next, v_S_next, a_S_next, dv_S = get_motion_state_V2(dis_S_now, v_S_now,a_L_now, s_S_temp, veh_S.time_stamp,veh_S.s_set,Risk_S)


            m_L,n_L = get_payoff_coefficient(Risk_L,veh_L.risk_level)
            m_S, n_S = get_payoff_coefficient(Risk_S, veh_S.risk_level)
            if type == 1:  #综合考虑安全和效益
                payoff_L = m_L * dv_L * 0.001 + n_L * (1 - Risk_L)  #这里考虑将效率的效益缩小，以适应风险波动的小范围特征
                payoff_S = m_S * dv_S * 0.001 + n_S * (1 - Risk_S)
            elif type == 2:#方案二：只考虑安全
                payoff_L = (1 - Risk_L)
                payoff_S = (1 - Risk_S)
            elif type == 3:  #方案三：只考虑效率
                payoff_L = dv_L
                payoff_S = dv_S
            payoff_sum = payoff_L + payoff_S
            # print('左车：Risk {}，m {},n {},距离 {}，下一时刻距离 {}'.format(Risk_L,m_L,n_L,dis_L_now,dis_L_next))
            # print('直行车：Risk {}，m {},n {},，距离 {}，下一时刻距离 {}'.format(Risk_S, m_S, n_S,dis_S_now,dis_S_next))
            # print('当前策略,左车{}，直行车{}，左车收益为{}，直行车收益为{}，总收益为{}'.format(S_L_all[i],S_S_all[j],
            #                                                        payoff_L,payoff_S,payoff_sum))
            df_payoff_L.loc[s_L_temp, s_S_temp] = payoff_L
            df_payoff_S.loc[s_L_temp, s_S_temp] = payoff_S
            df_payoff_sum.loc[s_L_temp, s_S_temp] = payoff_sum
            # df_re.loc[s_L_temp, s_S_temp] = (R_interact_feature_L,R_interact_feature_S)
            # df_re2.loc[s_L_temp, s_S_temp] = (Risk_L, Risk_S)
            # print(df_payoff_L,df_payoff_S,df_payoff_sum)
            if payoff_sum >= payoff_max:
                payoff_max = payoff_sum
                payoff_left_best = payoff_L
                payoff_stra_best = payoff_S
                L_best = [S_L_all[i],dis_L_next,v_L_next,a_L_next]
                S_best = [S_S_all[j], dis_S_next, v_S_next, a_S_next]


    # a = df_re
    relative_dis_L = cal_relative_dis(index, veh_L.conflict_point_real_index, veh_L.tra_real_s_sum[0])
    relative_dis_S = cal_relative_dis(index, veh_S.conflict_point_real_index, veh_S.tra_real_s_sum[0])
    # print(f's_L_all:{S_L_all},这个时间步的最优策略为{L_best}')
    veh_L.index.append(index);
    veh_L.time_stamp_list_game.append(time_now);
    veh_L.s.append(L_best[0])
    veh_L.dis_plan.append(relative_dis_L);veh_L.v.append(L_best[2]);veh_L.a_plan.append(L_best[3])
    veh_L.payoff.append(payoff_left_best)


    veh_S.index.append(index);veh_S.time_stamp_list_game.append(time_now);veh_S.s.append(S_best[0])
    veh_S.dis_plan.append(relative_dis_S);veh_S.v.append(S_best[2]);veh_S.a_plan.append(S_best[3])
    veh_S.payoff.append(payoff_stra_best)
    return veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum

def get_Nash_equilibrium_PET(index,veh_L,veh_S,df_all_veh,seg_trj_single,type=1):  #求解当前时间步的纳什均衡最优解,使用PET作为安全收益
    time_now = veh_L.time_stamp_list_real[index]
    df_all_veh_now = df_all_veh[df_all_veh['time_stamp']==time_now]
    dis_L_now, v_L_now, a_L_now = veh_L.dis_plan[-1], veh_L.v_real[index], veh_L.a_real[index]  #左车当下时刻距离冲突点的距离，当下时刻的速度，当下时刻加速度
    dis_S_now, v_S_now, a_S_now = veh_S.dis_plan[-1], veh_S.v_real[index], veh_S.a_real[index]
    S_L_all = S_S_all = veh_L.s_all
    L_best = S_best = []
    payoff_max = payoff_left_best = payoff_stra_best = -100
    df_payoff_L = pd.DataFrame(columns=S_L_all, index=S_S_all)
    df_payoff_S = pd.DataFrame(columns=S_L_all, index=S_S_all)
    df_payoff_sum = pd.DataFrame(columns=S_L_all, index=S_S_all)
    # df_re = pd.DataFrame(columns=S_L_all, index=S_S_all)
    # df_re2 = pd.DataFrame(columns=S_L_all, index=S_S_all)
    for i in range(len(S_L_all)):
        for j in range(len(S_S_all)):
            s_L_temp, s_S_temp = S_L_all[i], S_S_all[j]
            #计算左车和直行车的状态和收益
            #使用-0.5,0,0.5对状态进行估计，实际的a需要再计算
            dis_L_next, v_L_next, a_L_next, dv_L = get_motion_state(dis_L_now, v_L_now, s_L_temp,veh_L.time_stamp,veh_L.s_set)
            dis_S_next, v_S_next, a_S_next, dv_S = get_motion_state(dis_S_now, v_S_now, s_S_temp,veh_S.time_stamp,veh_S.s_set)

            #计算实际的状态
            # dis_L_next, v_L_next, a_L_next, dv_L = get_motion_state_V2(dis_L_now, v_L_now,a_L_now, s_L_temp, veh_L.time_stamp,veh_L.s_set,Risk_L)
            # dis_S_next, v_S_next, a_S_next, dv_S = get_motion_state_V2(dis_S_now, v_S_now,a_L_now, s_S_temp, veh_S.time_stamp,veh_S.s_set,Risk_S)

            #以当下状态，预测每个时间步二车的PET
            time_residul_L = dis_L_next / (v_L_next + 1e-5) #加一个小量，避免v=0（停车）的情况出现
            time_residul_S = dis_S_next / (v_S_next + 1e-5)
            PET = abs(time_residul_L-time_residul_S)


            m_L=m_S=n_L=n_S = 0.5
            if type == 1:  #综合考虑安全和效益
                payoff_L = m_L * dv_L  + n_L * PET  #这里考虑将效率的效益缩小，以适应风险波动的小范围特征
                payoff_S = m_S * dv_S  + n_S * PET
            elif type == 2:#方案二：只考虑安全
                payoff_L = PET
                payoff_S = PET
            elif type == 3:  #方案三：只考虑效率
                payoff_L = dv_L
                payoff_S = dv_S
            payoff_sum = payoff_L + payoff_S
            # print('左车：Risk {}，m {},n {},距离 {}，下一时刻距离 {}'.format(Risk_L,m_L,n_L,dis_L_now,dis_L_next))
            # print('直行车：Risk {}，m {},n {},，距离 {}，下一时刻距离 {}'.format(Risk_S, m_S, n_S,dis_S_now,dis_S_next))
            # print('当前策略,左车{}，直行车{}，左车收益为{}，直行车收益为{}，总收益为{}'.format(S_L_all[i],S_S_all[j],
            #                                                        payoff_L,payoff_S,payoff_sum))
            df_payoff_L.loc[s_L_temp, s_S_temp] = payoff_L
            df_payoff_S.loc[s_L_temp, s_S_temp] = payoff_S
            df_payoff_sum.loc[s_L_temp, s_S_temp] = payoff_sum
            # df_re.loc[s_L_temp, s_S_temp] = (R_interact_feature_L,R_interact_feature_S)
            # df_re2.loc[s_L_temp, s_S_temp] = (Risk_L, Risk_S)
            # print(df_payoff_L,df_payoff_S,df_payoff_sum)
            if payoff_sum >= payoff_max:
                payoff_max = payoff_sum
                payoff_left_best = payoff_L
                payoff_stra_best = payoff_S
                L_best = [S_L_all[i],dis_L_next,v_L_next,a_L_next]
                S_best = [S_S_all[j], dis_S_next, v_S_next, a_S_next]


    # a = df_re
    relative_dis_L = cal_relative_dis(index, veh_L.conflict_point_real_index, veh_L.tra_real_s_sum[0])
    relative_dis_S = cal_relative_dis(index, veh_S.conflict_point_real_index, veh_S.tra_real_s_sum[0])
    # print(f's_L_all:{S_L_all},这个时间步的最优策略为{L_best}')
    veh_L.index.append(index)
    veh_L.time_stamp_list_game.append(time_now)
    veh_L.s.append(L_best[0])
    veh_L.dis_plan.append(relative_dis_L);veh_L.v.append(L_best[2]);veh_L.a_plan.append(L_best[3])
    veh_L.payoff.append(payoff_left_best)

    veh_S.index.append(index);veh_S.time_stamp_list_game.append(time_now);veh_S.s.append(S_best[0])
    veh_S.dis_plan.append(relative_dis_S);veh_S.v.append(S_best[2]);veh_S.a_plan.append(S_best[3])
    veh_S.payoff.append(payoff_stra_best)
    return veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum

def get_conflict_point(tra_L_xy,tra_S_xy):  #基于左转和直行交互车辆的轨迹得到冲突点坐标

    veh_left_trj = np.column_stack((tra_L_xy[0],tra_L_xy[1]))
    veh_straight_trj = np.column_stack((tra_S_xy[0],tra_S_xy[1]))
    line_left = LineString(veh_left_trj)  #将车辆轨迹转化为shapely对象
    line_straight = LineString(veh_straight_trj)
    interscetion = line_left.intersection(line_straight)  #得到轨迹交点，即冲突点
    a = interscetion
    # print(len(interscetion.xy[0]))
    try:
        if (len(interscetion.xy[0])== 0 ):
            return -1
        else:
            conflict_point = (interscetion.xy[0][0], interscetion.xy[1][0])
            return conflict_point
    except:
        return -1

def cal_relative_dis(now_index,conflict_point_plan_index,tra_plan_s_sum):  #计算当前位置与冲突点的相对距离
    # print('index')
    # print(conflict_point_real_index)
    # print(tra_plan_s_sum)
    # print(len(tra_plan_s_sum))
    dis_relative = tra_plan_s_sum[conflict_point_plan_index] - tra_plan_s_sum[now_index]
    return dis_relative

def veh_para_init_waymo(veh_L,veh_S,df_L,df_S):  #对场景中交互车辆的所有参数进行初始化
    # 总交互时长，交互开始时间戳、时间步长
    time_interval = df_L['time_stamp'].max() - df_L['time_stamp'].min()
    time_begin = df_L['time_stamp'].min()
    time_stamp = 0.1
    veh_L.id = df_L['obj_id'].iloc[0]
    veh_S.id = df_S['obj_id'].iloc[0]
    veh_L.size = (df_L['length'].iloc[0],df_L['width'].iloc[0])
    veh_S.size = (df_S['length'].iloc[0],df_S['width'].iloc[0])
    veh_L.time_interactive = veh_S.time_interactive = time_interval
    veh_L.time_stamp = veh_S.time_stamp = time_stamp
    veh_L.time_stamp_list_real = veh_S.time_stamp_list_real = df_L['time_stamp'].tolist()
    df_L.loc[:,'ax_next'],df_L.loc[:,'ay_next'] = -1,-1
    df_S.loc[:,'ax_next'], df_S.loc[:,'ay_next'] = -1, -1
    # print(df_L)

    #计算左车、直行车下一时刻的加速度ax,ay,再计算a
    for i in range(len(df_L)-1):
        label = df_L['frame_label'].iloc[i]
        # print(label)

        df_L.loc[df_L['frame_label']==label,'ax_next'] = (df_L['velocity_x'].iloc[i+1]-df_L['velocity_x'].iloc[i])\
                                                         /(df_L['time_stamp'].iloc[i+1]-df_L['time_stamp'].iloc[i])
        df_L.loc[df_L['frame_label'] == label, 'ay_next'] = (df_L['velocity_y'].iloc[i + 1] - df_L['velocity_y'].iloc[i]) \
                                                            / (df_L['time_stamp'].iloc[i + 1] - df_L['time_stamp'].iloc[i])
    for i in range(len(df_S)-1):
        label = df_S['frame_label'].iloc[i]
        df_S.loc[df_S['frame_label']==label,'ax_next'] = (df_S['velocity_x'].iloc[i+1]-df_S['velocity_x'].iloc[i])\
                                                         /(df_S['time_stamp'].iloc[i+1]-df_S['time_stamp'].iloc[i])
        df_S.loc[df_S['frame_label'] == label, 'ay_next'] = (df_S['velocity_y'].iloc[i + 1] - df_S['velocity_y'].iloc[i]) \
                                                            / (df_S['time_stamp'].iloc[i + 1] - df_S['time_stamp'].iloc[i])
    # print(df_L)

    #起点、终点位置信息
    veh_L.start_point,veh_L.end_point = (df_L['center_x'].iloc[0],df_L['center_y'].iloc[0]),(df_L['center_x'].iloc[-1],df_L['center_y'].iloc[-1])
    veh_S.start_point,veh_S.end_point = (df_S['center_x'].iloc[0],df_S['center_y'].iloc[0]),(df_S['center_x'].iloc[-1],df_S['center_y'].iloc[-1])

    # print('起终点信息，左车，起点{}，终点{},距离{}；直行车，起点{}，终点{}，距离{}'.format(veh_L.start_point,veh_L.end_point,
    #                                                           np.linalg.norm(np.array(veh_L.start_point) - np.array(veh_L.end_point)),
    #                                                           veh_S.start_point,veh_S.end_point,
    #                                                           np.linalg.norm(np.array(veh_S.start_point) - np.array(veh_S.end_point))))
    #真实轨迹、速度、加速度信息
    #左车
    print(f'左车序列长度{len(df_L)}，直行车序列长度{len(df_S)}')
    L_trj_real = np.array([df_L['center_x'].tolist(),df_L['center_y'].tolist()])
    L_vx_real,L_vy_real = np.array(df_L['velocity_x'].tolist()),np.array(df_L['velocity_y'].tolist())
    L_ax_real,L_ay_real = np.array(df_L['ax_next'].tolist()),np.array(df_L['ay_next'].tolist())
    # print(L_trj_real_all)
    # print(L_vx_real)
    veh_L.tra_real_xy, veh_L.vx_real, veh_L.vy_real = L_trj_real, L_vx_real, L_vy_real
    veh_L.ax_real, veh_L.ay_real = L_ax_real,L_ay_real

    #直行车
    S_trj_real = np.array([df_S['center_x'].tolist(), df_S['center_y'].tolist()])
    S_vx_real, S_vy_real = np.array(df_S['velocity_x'].tolist()), np.array(df_S['velocity_y'].tolist())
    S_ax_real, S_ay_real = np.array(df_S['ax_next'].tolist()), np.array(df_S['ay_next'].tolist())
    veh_S.tra_real_xy, veh_S.vx_real, veh_S.vy_real = S_trj_real, S_vx_real, S_vy_real
    veh_S.ax_real, veh_S.ay_real = S_ax_real, S_ay_real


    # 规划两辆车的行驶轨迹，左车使用贝塞尔曲线进行拟合，直行车使用直线
    # veh_L = get_plan_trajectory_veh(veh_L, 'left')
    # veh_S = get_plan_trajectory_veh(veh_S, 'straight')
    # 将xy坐标系信息转换为自然坐标系
    veh_L.tra_real_s, veh_L.tra_real_s_sum = location_xy2_s(veh_L.tra_real_xy)
    veh_S.tra_real_s, veh_S.tra_real_s_sum = location_xy2_s(veh_S.tra_real_xy)

    angle_frenet_list_L, angle_frenet_list_S = np.array(veh_L.tra_real_s[1]), np.array(veh_S.tra_real_s[1])
    veh_L.v_real = veh_L.vx_real * angle_frenet_list_L + veh_L.vy_real * angle_frenet_list_L
    veh_L.a_real = veh_L.ax_real *  angle_frenet_list_L + veh_L.ay_real * angle_frenet_list_L
    veh_S.v_real = veh_S.vx_real * angle_frenet_list_S + veh_S.vy_real * angle_frenet_list_S # !注意不能求平方根，要转换到frenet坐标系中，要先求出每次的转向角
    veh_S.a_real = veh_S.ax_real * angle_frenet_list_S + veh_S.ay_real * angle_frenet_list_S

    # 冲突点信息计算与记录
    conflict_point = get_conflict_point(veh_L.tra_real_xy,veh_S.tra_real_xy)
    if conflict_point == -1:
        return 'error','error'
    veh_L.conflict_point_real = veh_S.conflict_point_real = conflict_point
    index_conflict_point_L, index_conflict_point_S = find_point_index(veh_L.tra_real_xy,veh_S.tra_real_xy,
                                                                      conflict_point)  # 找到冲突点在轨迹点集合中的序列
    veh_L.conflict_point_real_index, veh_S.conflict_point_real_index = index_conflict_point_L, index_conflict_point_S
    # print('冲突点 index 左车{}，直行车{}'.format(index_conflict_point_L, index_conflict_point_S))

    #当前时刻距离冲突点的相对距离计算
    relative_dis_L = cal_relative_dis(0, veh_L.conflict_point_real_index, veh_L.tra_real_s_sum[0])
    relative_dis_S = cal_relative_dis(0, veh_S.conflict_point_real_index, veh_S.tra_real_s_sum[0])
    # print('初始时刻的相对距离计算，左车{}m，直行车{}m'.format(relative_dis_L,relative_dis_S))

    # 初始时刻的速度加速度信息  这一段的参数需要之后进行更新
    veh_L.index.append(0)
    veh_L.time_stamp_list_game.append(time_begin)
    veh_L.s.append('s2')  #需要修改
    veh_L.v.append(veh_L.v_real[0])
    veh_L.dis_plan.append(relative_dis_L)  #需要重新计算!!
    veh_L.a_plan.append(0.01)
    veh_S.index.append(0)
    veh_S.time_stamp_list_game.append(time_begin)
    veh_S.s.append('s2')  #需要修改
    veh_S.v.append(veh_S.v_real[0])
    veh_S.dis_plan.append(relative_dis_S)   #需要重新计算!!
    veh_S.a_plan.append(0.01)

    return veh_L, veh_S

def veh_para_init_xianxia(veh_L,veh_S,df_L,df_S,new_conflict_point=''):  #对场景中交互车辆的所有参数进行初始化
    # 总交互时长，交互开始时间戳、时间步长
    time_interval = df_L['time_stamp'].max() - df_L['time_stamp'].min()
    time_begin = df_L['time_stamp'].min()
    time_stamp = 0.1
    veh_L.id = df_L['obj_id'].iloc[0]
    veh_S.id = df_S['obj_id'].iloc[0]
    veh_L.size = (df_L['length'].iloc[0],df_L['width'].iloc[0])
    veh_S.size = (df_S['length'].iloc[0],df_S['width'].iloc[0])
    veh_L.time_interactive = veh_S.time_interactive = time_interval
    veh_L.time_stamp = veh_S.time_stamp = time_stamp
    veh_L.time_stamp_list_real = veh_S.time_stamp_list_real = df_L['time_stamp'].tolist()

    #风险感知等级，需要标定
    veh_L.risk_level = (0.44, 0.61)
    veh_S.risk_level = (0.59, 0.64)
    #起点、终点位置信息
    veh_L.start_point,veh_L.end_point = (df_L['center_x'].iloc[0],df_L['center_y'].iloc[0]),(df_L['center_x'].iloc[-1],df_L['center_y'].iloc[-1])
    veh_S.start_point,veh_S.end_point = (df_S['center_x'].iloc[0],df_S['center_y'].iloc[0]),(df_S['center_x'].iloc[-1],df_S['center_y'].iloc[-1])

    print('起终点信息，左车，起点{}，终点{},距离{}；直行车，起点{}，终点{}，距离{}'.format(veh_L.start_point,veh_L.end_point,
                                                              np.linalg.norm(np.array(veh_L.start_point) - np.array(veh_L.end_point)),
                                                              veh_S.start_point,veh_S.end_point,
                                                              np.linalg.norm(np.array(veh_S.start_point) - np.array(veh_S.end_point))))
    #真实轨迹、速度、加速度信息
    #左车
    print(f'左车序列长度{len(df_L)}，直行车序列长度{len(df_S)}')
    L_trj_real = np.array([df_L['center_x'].tolist(),df_L['center_y'].tolist()])
    L_vx_real,L_vy_real = np.array(df_L['velocity_x'].tolist()),np.array(df_L['velocity_y'].tolist())
    L_ax_real,L_ay_real = np.array(df_L['ax_next'].tolist()),np.array(df_L['ay_next'].tolist())
    # print(L_trj_real_all)
    # print(L_vx_real)
    veh_L.tra_real_xy, veh_L.vx_real, veh_L.vy_real = L_trj_real, L_vx_real, L_vy_real
    veh_L.ax_real, veh_L.ay_real = L_ax_real,L_ay_real
    veh_L.v_real = np.array(df_L['Speed[km/h]'].tolist())/3.6
    veh_L.a_real = np.array(df_L['Acceleration[m/s2]'].tolist())
    #直行车
    S_trj_real = np.array([df_S['center_x'].tolist(), df_S['center_y'].tolist()])
    S_vx_real, S_vy_real = np.array(df_S['velocity_x'].tolist()), np.array(df_S['velocity_y'].tolist())
    S_ax_real, S_ay_real = np.array(df_S['ax_next'].tolist()), np.array(df_S['ay_next'].tolist())
    veh_S.tra_real_xy, veh_S.vx_real, veh_S.vy_real = S_trj_real, S_vx_real, S_vy_real
    veh_S.ax_real, veh_S.ay_real = S_ax_real, S_ay_real
    veh_S.v_real = np.array(df_L['Speed[km/h]'].tolist())/3.6
    veh_S.a_real = np.array(df_L['Acceleration[m/s2]'].tolist())

    # 规划两辆车的行驶轨迹，左车使用贝塞尔曲线进行拟合，直行车使用直线
    # veh_L = get_plan_trajectory_veh(veh_L, 'left')
    # veh_S = get_plan_trajectory_veh(veh_S, 'straight')
    # 将xy坐标系信息转换为自然坐标系
    veh_L.tra_real_s, veh_L.tra_real_s_sum = location_xy2_s(veh_L.tra_real_xy)
    veh_S.tra_real_s, veh_S.tra_real_s_sum = location_xy2_s(veh_S.tra_real_xy)

    # 冲突点信息计算与记录

    conflict_point = get_conflict_point(veh_L.tra_real_xy,veh_S.tra_real_xy)

    veh_L.conflict_point_real = veh_S.conflict_point_real = conflict_point
    index_conflict_point_L, index_conflict_point_S = find_point_index(veh_L.tra_real_xy,veh_S.tra_real_xy,
                                                                      conflict_point)  # 找到冲突点在轨迹点集合中的序列
    veh_L.conflict_point_real_index, veh_S.conflict_point_real_index = index_conflict_point_L, index_conflict_point_S
    print('冲突点 index 左车{}，直行车{}'.format(index_conflict_point_L, index_conflict_point_S))

    #当前时刻距离冲突点的相对距离计算
    relative_dis_L = cal_relative_dis(0, veh_L.conflict_point_real_index, veh_L.tra_real_s_sum[0])
    relative_dis_S = cal_relative_dis(0, veh_S.conflict_point_real_index, veh_S.tra_real_s_sum[0])
    print('初始时刻的相对距离计算，左车{}m，直行车{}m'.format(relative_dis_L,relative_dis_S))
    # else:
    #     conflict_point = new_conflict_point

    # 初始时刻的速度加速度信息  这一段的参数需要之后进行更新
    veh_L.index.append(0)
    veh_L.time_stamp_list_game.append(time_begin)
    veh_L.s.append('s2')  #需要修改
    veh_L.v.append(veh_L.v_real[0])
    veh_L.dis_plan.append(relative_dis_L)  #需要重新计算!!
    veh_L.a_plan.append(0.01)
    veh_S.index.append(0)
    veh_S.time_stamp_list_game.append(time_begin)
    veh_S.s.append('s2')  #需要修改
    veh_S.v.append(veh_S.v_real[0])
    veh_S.dis_plan.append(relative_dis_S)   #需要重新计算!!
    veh_S.a_plan.append(0.01)

    return veh_L, veh_S

def find_point_index(L_tra_plan_xy,S_tra_plan_xy,conflict_point):
    index_L,index_S = -1,-1
    min_dis_L,min_dis_S = 999,999
    #寻找左转车轨迹冲突点序列
    for i in range(len(L_tra_plan_xy[0])):
        dis_temp_L = np.sqrt((L_tra_plan_xy[0][i] - conflict_point[0]) ** 2 + (L_tra_plan_xy[1][i] - conflict_point[1]) ** 2)
        if dis_temp_L < min_dis_L:
            min_dis_L = dis_temp_L
            index_L = i
    #寻找直行车轨迹冲突点序列
    for i in range(len(S_tra_plan_xy[0])):
        dis_temp_S = np.sqrt((S_tra_plan_xy[0][i] - conflict_point[0]) ** 2 + (S_tra_plan_xy[1][i] - conflict_point[1]) ** 2)
        if dis_temp_S < min_dis_S:
            min_dis_S= dis_temp_S
            index_S = i

    return index_L,index_S

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def interactive_scoring(veh_L,veh_S):   #基于求解结果给交互过程打分
    scoring_L = scoring_S = 1
    fenzi_L = fenmu_L = fenzi_S = fenmu_S = 0

    #规划值
    a_L_plan = veh_L.a_plan
    # theta_L_plan = veh_L.theta_plan
    a_S_plan = veh_S.a_plan
    # theta_S_plan = veh_S.theta_plan
    #实际值
    # theta_L_real = veh_L.theta_real
    a_S_real = veh_S.a_real
    # theta_S_real = veh_S.theta_real
    a_L_real = veh_L.a_real
    if np.isnan(a_L_real[-1]):
        a_L_real = np.delete(a_L_real,-1)
    if np.isnan(a_S_real[-1]):
        a_S_real = np.delete(a_S_real,-1)


    max_L = min(len(a_L_real),len(a_L_plan))
    max_S = min(len(a_S_real),len(a_S_plan))
    a_L_real_2,a_L_plan_2 = np.array(a_L_real[:max_L]),np.array(a_L_plan[:max_L])
    a_S_real_2,a_S_plan_2 = np.array(a_S_real[:max_S]), np.array(a_S_plan[:max_S])
    a_L_real_2[np.isnan(a_L_real_2)] = 0
    a_L_plan_2[np.isnan(a_L_plan_2)] = 0
    a_S_real_2[np.isnan(a_S_real_2)] = 0
    a_S_plan_2[np.isnan(a_S_plan_2)] = 0
    print(a_L_real_2, a_L_plan_2)
    # print( a_L_real_2,a_L_plan_2)
    scoring_L = cos_sim(a_L_real_2, a_L_plan_2)
    scoring_S = cos_sim(a_S_plan_2, a_S_real_2)
    # import scipy.stats
    # KL_L = scipy.stats.entropy(a_L_real_2,a_L_plan_2)
    # KL_S = scipy.stats.entropy(a_S_real_2, a_S_plan_2)
    # print(KL_L,KL_S)
    # from dtw import dtw
    # manhattan_distance = lambda x, y: np.abs(x - y)
    # # d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
    # scoring_L, cost_matrix_L, acc_cost_matrix_L, path_L = dtw(a_L_real_2, a_L_plan_2, dist=manhattan_distance)
    # scoring_S, cost_matrix_S, acc_cost_matrix_S, path_S = dtw(a_S_plan_2, a_S_real_2, dist=manhattan_distance)


    return scoring_L,scoring_S

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

def get_time_range_index(time_range,seg_trj2):  #得到时间范围对应的frame序列
    time_range_index1 = ()
    time_range = eval(time_range)
    print(f'总时间长度{time_range[1]-time_range[0]},时间{time_range}')
    df = seg_trj2[(seg_trj2['time_stamp']>=time_range[0]) & (seg_trj2['time_stamp']<=time_range[1])]
    time_range_index1 = (df['frame_label'].min(),df['frame_label'].max())
    print('时间范围为{}'.format(time_range_index1))
    return time_range_index1

def get_real_time_index(time_range,seg_trj_single, veh_left_id, veh_strai_later_id):  #seg_trj_single 为一个场景20s内的所有轨迹
    time_range = eval(time_range)
    # print(f'总时间长度{time_range[1] - time_range[0]},时间{time_range}')
    df = seg_trj_single[(seg_trj_single['time_stamp'] >= time_range[0]) & (seg_trj_single['time_stamp'] <= time_range[1])]
    time_range_index1 = (df['frame_label'].min(), df['frame_label'].max())
    # print('原始的时间范围为{}'.format(time_range_index1))

    df_L, df_S = seg_trj_single[seg_trj_single['obj_id'] == veh_left_id], seg_trj_single[
        seg_trj_single['obj_id'] == veh_strai_later_id]
    #求冲突点
    trj_L = [df_L['center_x'].tolist(),df_L['center_y'].tolist()]
    trj_S = [df_S['center_x'].tolist(),df_S['center_y'].tolist()]
    # print(trj_L)
    # print(trj_S)
    try:
        conflict_point = get_conflict_point(trj_L,trj_S)
        if conflict_point == -1:  # 表示不存在冲突点
            print('不存在冲突点')
            return 'error', 'error'
    except:
        print('不存在冲突点')
        return 'error','error'

    index_L,index_S = find_point_index(trj_L, trj_S,conflict_point)  # 找到冲突点在轨迹点集合中的序列

    a,b = max(df_L['frame_label'].min(), df_S['frame_label'].min()),min(df_L['frame_label'].max(), df_S['frame_label'].max())
    # print(f'最初的最小frame{a},最大为{b}')
    # 对左车和直行车的时间范围筛选一遍
    min_frame_label = min(max(df_L['frame_label'].min(), df_S['frame_label'].min()),index_L,index_S)
    max_frame_label = max(min(df_L['frame_label'].max(), df_S['frame_label'].max()),index_L,index_S)  #要确保交互范围在在进入冲突点之前就存在
    # print(f'修正后的最小frame{min_frame_label}，最大frame{max_frame_label}')
    #第二遍筛选，确保时间起点左转车在冲突点之前，直行车在冲突点之后
    #求冲突点
    min_frame_label = max(df_L['frame_label'].min(), df_S['frame_label'].min())
    max_frame_label = min(df_L['frame_label'].max(), df_S['frame_label'].max())

    return min_frame_label,max_frame_label

def re_check_data(df_L, df_S,min_frame_label,max_frame_label):
    range_ori = max_frame_label - min_frame_label
    flag = -1
    df_S_new = df_S
    df_L_new = df_L
    if len(df_L)>len(df_S):  #需要更正直行车信息
        frame_list_big = df_L['frame_label'].tolist()
        for i in range(len(frame_list_big)):
            tar_frame = frame_list_big[i]
            if tar_frame not in df_S['frame_label'].tolist() :
                if frame_list_big[i-1] in df_S['frame_label'].tolist(): #用上一帧数据取代
                    a = df_S[df_S['frame_label']==frame_list_big[i-1]]
                    a['frame_label'] = tar_frame
                elif frame_list_big[i+1] in df_S['frame_label'].tolist(): #否则用下一帧数据取代
                    a = df_S[df_S['frame_label']==frame_list_big[i+1]]
                    a['frame_label'] = tar_frame
                #插入行
                # 拆分
                pd_arr1 = df_S_new[:i]
                pd_arr2 = df_S_new[i:]
                # 参数：添加数据，是否无视行索引
                df_S_new = pd_arr1.append(a, ignore_index=True).append(pd_arr2, ignore_index=True)
                df_L_new = df_L
                # df_S = pd.concat([df_S,a])

    elif len(df_L)<len(df_S):  #需要更正左转车信息
        frame_list_big = df_S['frame_label'].tolist()
        for i in range(len(frame_list_big)):
            tar_frame = frame_list_big[i]
            if tar_frame not in df_L['frame_label'].tolist():
                if frame_list_big[i - 1] in df_L['frame_label'].tolist():  # 用上一帧数据取代
                    a = df_L[df_L['frame_label'] == frame_list_big[i - 1]]
                    a['frame_label'] = tar_frame
                elif frame_list_big[i + 1] in df_L['frame_label'].tolist():  # #否则用下一帧数据取代
                    a = df_L[df_L['frame_label'] == frame_list_big[i + 1]]
                    a['frame_label'] = tar_frame
                # 插入行
                # 拆分
                pd_arr1 = df_L_new[:i]
                pd_arr2 = df_L_new[i:]
                # 参数：添加数据，是否无视行索引
                df_L_new = pd_arr1.append(a, ignore_index=True).append(pd_arr2, ignore_index=True)
                df_S_new = df_S

                # df_L = pd.concat([df_L, a])
    elif len(df_L)==len(df_S):  #不需要更正，直接返回（实际并不能完全保证，但bug的可能性很低）
        flag = 0


    return df_L_new, df_S_new





def main_waymo(a,b,index,cal_type,u_type,risk_level_list):
    start = datetime.datetime.now()
    # load data
    filepath2 = r'F:\Result_save\data_save\Turn_left_scenario_info_0419.xlsx'  #共194个事件
    df_all = pd.read_excel(filepath2)
    # print(f'共{len(df_all)}个文件')
    scoring_list = []
    file_count = 0
    error_seg_list = []
    # risk_level_list = [(0.3, 0.5), (0.4, 0.6), (0.6, 0.8)]  # 分析不同风险感知水平下的驾驶人的策略变化 分别依次对应三种类型：保守、稳定、激进

    for risk_down, risk_up in risk_level_list:
        for i in range(a, b):
            try:
                target_segment = df_all['segment_id'].iloc[i]
                target_scenario = df_all['scenario_id'].iloc[i]
                veh_left_id = df_all['turn_left_veh_id'].iloc[i]
                veh_strai_later_id = df_all['veh_straight_later'].iloc[i]

                time_range = df_all['time_range_interactive'].iloc[i]
                # time_range_index = (37,147)
                scoring_dict = {}
                a_result = {}
                file_index = seg_id_2_fileindex(target_segment)
                # print(file_index)
                print(f'范围是{a}-{b}，index为{index},这是第{i}个片段，file_index 为{target_segment},target_scenario为{target_scenario},'
                      f'左转车为{veh_left_id},直行车（后车）{veh_strai_later_id},驾驶人风险水平为:{risk_down}:{risk_up},计算类型{cal_type}')
                filepath_trj = f'F:/Result_save/data_save/all_scenario_all_objects_info/{file_index}_all_scenario_all_object_info_1.csv'

                seg_trj = pd.read_csv(filepath_trj)
                seg_trj2 = seg_trj[seg_trj['scenario_label'] == target_scenario]


                seg_trj = seg_trj[seg_trj['valid'] == True]
                seg_trj_single = seg_trj[seg_trj['scenario_label'] == target_scenario]  # 一个20s场景中的所有轨迹数据
                min_frame_label, max_frame_label = get_real_time_index(time_range, seg_trj_single, veh_left_id, veh_strai_later_id)
                if min_frame_label == 'error':
                    print(f'第{i}个片段，file_index 为{target_segment},target_scenario为{target_scenario}，未找到冲突点，')
                    error_path = f'F:/Result_save/left/error/error_{target_segment}_{target_scenario}.txt'
                    with open(error_path, "w") as f:
                        f.write(f"未找到冲突点")
                    continue

                # 生成轨迹移动视频
                # vidoe_gen.generate(seg_trj2,target_segment,target_scenario)
                # df_L_temp, df_S_temp = seg_trj_single[seg_trj_single['obj_id'] == veh_left_id], seg_trj_single[
                #     seg_trj_single['obj_id'] == veh_strai_later_id]

                seg_trj_single_2 = seg_trj_single[
                    (min_frame_label <= seg_trj_single['frame_label']) & (seg_trj_single['frame_label'] <= max_frame_label)]
                # seg_trj_single_2 = re_check_data(seg_trj_single_2,min_frame_label,max_frame_label)  #解决极端情况：时间范围内极少数帧缺失
                df_L_1, df_S_1 = seg_trj_single_2[seg_trj_single_2['obj_id'] == veh_left_id], seg_trj_single_2[
                    seg_trj_single_2['obj_id'] == veh_strai_later_id]
                df_all_veh = seg_trj_single_2  # 所有车辆的轨迹数据
                df_L, df_S = re_check_data(df_L_1, df_S_1, min_frame_label, max_frame_label)  # 解决极端情况：时间范围内极少数帧缺失
                if len(df_L) != len(df_S): #暂时先不处理
                    continue
                # 初始化交互车辆
                veh_L, veh_S = veh(), veh()
                veh_L, veh_S = veh_para_init_waymo(veh_L, veh_S, df_L, df_S)
                if veh_L == 'error':
                    print(f'第{i}个片段，file_index 为{target_segment},target_scenario为{target_scenario}，未找到冲突点，')
                    error_path = f'F:/Result_save/left/error/error_{target_segment}_{target_scenario}.txt'
                    with open(error_path, "w") as f:
                        f.write(f"未找到冲突点")
                    continue
                # 风险感知等级，需要标定
                veh_L.risk_level = (risk_down, risk_up)
                veh_S.risk_level = (risk_down, risk_up)

                payoff_L = []  # 记录所有时间步的所有策略的收益
                payoff_S = []
                payoff_sum = []  # 记录所有时间步的所有策略的收益总和
                print(f'共{len(df_L)}个时间步')
                for k in range(len(df_L)):
                    # print('第%d个时间步'%i)
                    if u_type=='risk':
                        veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum = get_Nash_equilibrium(i, veh_L, veh_S,
                                                                                                     df_all_veh, seg_trj_single,type=cal_type)
                    else:
                        veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum = get_Nash_equilibrium_PET(i, veh_L, veh_S,
                                                                                                     df_all_veh,seg_trj_single,type=cal_type)

                    payoff_L.append(df_payoff_L)
                    payoff_S.append(df_payoff_S)
                    payoff_sum.append(df_payoff_sum)
                    if k % 20 == 0:
                        print('第%d个时间步，收益矩阵为' % k)
                        print(df_payoff_sum)

                # print(veh_L.s)
                # print(veh_S.s)

                scoring_L, scoring_S = interactive_scoring(veh_L, veh_S)
                print(f'最终得分，左车{scoring_L},右车{scoring_S}')
                scoring_dict['segment_id'] = target_segment
                scoring_dict['scenario_id'] = target_scenario
                scoring_dict['risk_down'] = risk_down
                scoring_dict['risk_up'] = risk_up
                scoring_dict['scoring_left'] = scoring_L
                scoring_dict['scoring_straight'] = scoring_S
                a_result['left_a_real'] = veh_L.a_real
                a_result['left_a_plan'] = veh_L.a_plan
                a_result['straight_a_real'] = veh_S.a_real
                a_result['straight_a_plan'] = veh_S.a_plan
                scoring_list.append(scoring_dict)
                a_result_path = f'F:/Result_save/left/detail/a_result_{target_segment}_{target_scenario}.txt'
                with open(a_result_path, "w") as f:
                    f.write(f"left_a_real:{veh_L.a_real.tolist()}\n")
                    f.write(f"left_a_plan:{veh_L.a_plan}\n")
                    f.write(f"straight_a_real:{veh_S.a_real.tolist()}\n")
                    f.write(f"straight_a_plan:{veh_S.a_plan}\n")
            except:
                continue


    scoring_df = pd.DataFrame(scoring_list)
    scoring_output = f'result/waymo_scoring_result_{index}_cal_type{cal_type}_range_{a}_{b}_u_type_{u_type}.csv'
    scoring_df.to_csv(scoring_output)
    end = datetime.datetime.now()
    print(f'程序计算用时{end - start}')

def main_xianxia(cal_type,u_type,risk_level_list):  #对仙霞剑河交叉口进行评价
    start = datetime.datetime.now()
    filepth2 = r'veh_all_info_tra_all.xlsx'  # 交叉口西进口左转车的轨迹数据
    df_all_info_all = pd.read_excel(filepth2)
    scoring_list = []
    # risk_level_list = [(0.4,0.7),(0.3,0.6),(0.5,0.8),(0.6,0.9)]  #分析不同风险感知水平下的驾驶人的策略变化
    # cal_type = 1
    for segment_id in [0,1]:
        df_all_info = df_all_info_all[df_all_info_all['segment_index']==segment_id]
        num_scenario = len(pd.unique(df_all_info['scenario_label']))  # 交互场景数量
        for risk_down,risk_up in risk_level_list:
            for k in range(num_scenario):

                if segment_id == 0:
                    direction_veh = 'west'
                elif segment_id == 1:
                    direction_veh = 'south'
                print(f'segment:{segment_id},scenario_label:{k}')
                df_L_ori = df_all_info[(df_all_info['scenario_label'] == k) & (df_all_info['action_type'] == 'left')]
                df_S_ori = df_all_info[(df_all_info['scenario_label'] == k) & (df_all_info['action_type'] == 'straight')]
                veh_left_id = pd.unique(df_L_ori['obj_id'])[0]
                veh_stra_id = pd.unique(df_S_ori['obj_id'])[0]

                #确定交互时间范围
                seg_trj_single = df_all_info[df_all_info['scenario_label'] == k]  # 一个20s场景中的所有轨迹数据
                min_frame_label = max(df_L_ori['frame_label'].iloc[0],df_S_ori['frame_label'].iloc[0])
                max_frame_label = min(df_L_ori['frame_label'].iloc[-1],df_S_ori['frame_label'].iloc[-1])
                # min_frame_label, max_frame_label = get_real_time_index(time_range, seg_trj_single,veh_left_id,veh_stra_id)
                seg_trj_single_2 = seg_trj_single[
                    (min_frame_label <= seg_trj_single['frame_label']) & (seg_trj_single['frame_label'] <= max_frame_label)]
                df_L, df_S = seg_trj_single_2[seg_trj_single_2['obj_id'] == veh_left_id], seg_trj_single_2[
                    seg_trj_single_2['obj_id'] == veh_stra_id]
                df_all_veh = seg_trj_single_2  # 所有车辆的轨迹数据

                # 初始化交互车辆
                veh_L, veh_S = veh(), veh()
                # #做一个预判，判断交互区间内是否存在冲突点
                trj_L = [df_L['center_x'].tolist(), df_L['center_y'].tolist()]
                trj_S = [df_S['center_x'].tolist(), df_S['center_y'].tolist()]
                try:
                    conflict_point = get_conflict_point(trj_L, trj_S)
                    veh_L, veh_S = veh_para_init_xianxia(veh_L, veh_S, df_L, df_S)
                except:
                    conflict_point = -1
                    print(f'这是第{k}个事件,交互范围内的轨迹没有冲突点')
                    continue

                # 风险感知等级，需要标定
                veh_L.risk_level = (risk_down,risk_up)
                veh_S.risk_level = (risk_down,risk_up)



                # if conflict_point != -1:
                #     veh_L, veh_S = veh_para_init_xianxia(veh_L, veh_S, df_L, df_S)
                # else:
                #
                #     trj_L = [df_L_ori['center_x'].tolist(), df_L_ori['center_y'].tolist()]
                #     trj_S = [df_S_ori['center_x'].tolist(), df_S_ori['center_y'].tolist()]
                #     try:
                #         new_conflict_point = get_conflict_point(trj_L, trj_S)
                #         veh_L, veh_S = veh_para_init_xianxia(veh_L, veh_S, df_L, df_S,new_conflict_point)
                #     except:
                #         new_conflict_point = -1
                #         print(f'这是第{k}个事件,完全没有冲突点')
                #         continue

                #博弈交互过程计算
                payoff_L = []  # 记录所有时间步的所有策略的收益
                payoff_S = []
                payoff_sum = []  # 记录所有时间步的所有策略的收益总和
                print(f'这是第{k}个事件，共{len(df_L)}个时间步,驾驶人风险水平为:{risk_down}:{risk_up},计算类型{cal_type}')
                for i in range(len(df_L)):

                    if u_type=='risk':
                        veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum = get_Nash_equilibrium(i, veh_L, veh_S,
                                                                                                     df_all_veh, seg_trj_single,type=cal_type)
                    else:
                        veh_L, veh_S, df_payoff_L, df_payoff_S, df_payoff_sum = get_Nash_equilibrium_PET(i, veh_L, veh_S,
                                                                                                     df_all_veh,seg_trj_single,type=cal_type)

                    payoff_L.append(df_payoff_L)
                    payoff_S.append(df_payoff_S)
                    payoff_sum.append(df_payoff_sum)
                    # print(df_payoff_L)
                    # print(df_payoff_S)
                    if i % 20 == 0:
                        print('第%d个时间步，收益矩阵为' % i)
                        print(df_payoff_sum)
                print(veh_L.s)
                print(veh_S.s)
                scoring_L, scoring_S = interactive_scoring(veh_L, veh_S)
                print(f'最终得分，左车{scoring_L},直行车{scoring_S}')
                scoring_dict = {}
                scoring_dict['segment_id'] = segment_id
                scoring_dict['direction'] = direction_veh
                scoring_dict['scenario_id'] = k
                scoring_dict['risk_down'] = risk_down
                scoring_dict['risk_up'] = risk_up
                scoring_dict['scoring_left'] = scoring_L
                scoring_dict['scoring_straight'] = scoring_S
                scoring_list.append(scoring_dict)

    scoring_df = pd.DataFrame(scoring_list)
    scoring_output = f'result/xianxia_scoring_result_cal_type_{cal_type}_all_u_type_{u_type}.xlsx'
    scoring_df.to_excel(scoring_output,index=None)
    end = datetime.datetime.now()
    print(f'程序计算用时{end - start}')


if __name__ == '__main__':
    # risk_level_list = [(0.3, 0.5), (0.4, 0.6), (0.6, 0.8)]  # 分析不同风险感知水平下的驾驶人的策略变化 分别依次对应三种类型：保守、稳定、激进
    risk_level_list = [(0.3, 0.5)]
    cal_type = 1  #1：综合考虑风险和效率；2：考虑风险；3：考虑效率
    u_type = 'risk'  #安全风险的计算方式  risk：使用风险场计算，PET：使用PET作为安全收益进行计算

    # main_xianxia(cal_type,u_type,risk_level_list)
    # #共194个文件
    a,b,index = 0,194,1
    main_waymo(a,b,index,cal_type,u_type,risk_level_list)

