import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.patheffects as path_effects
import matplotlib as mpl
import cv2
import glob
from tqdm import tqdm

# Generate visualization images.
def create_figure_and_axes(size_pixels):
    """Initializes one_state unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1,1)

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax

def plot_top_view_single_pic_map(trj_in, scenario_id_in, frame_id_in):
    plt.figure(figsize=(10 ,7))
    plt.figure()
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(
        np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 20),
        fontsize=5)
    plt.yticks(
        np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 20),
        fontsize=5)
    ax = plt.gca()


    # trj_in['center_x'] = trj_in['center_x'] - trj_in['center_x'].min()
    # trj_in['center_y'] = trj_in['center_y'] - trj_in['center_y'].min()
    unique_veh_id = pd.unique(trj_in['obj_id'])
    for single_veh_id in unique_veh_id:
        single_veh_trj = trj_in[trj_in['obj_id'] == single_veh_id]
        single_veh_trj = single_veh_trj[single_veh_trj['frame_label'] == frame_id_in]
        # print(single_veh_trj)
        if len(single_veh_trj) > 0 and single_veh_trj['valid'].iloc[0] == True:
            ts = ax.transData
            coords = [single_veh_trj['center_x'].iloc[0], single_veh_trj['center_y'].iloc[0]]
            if single_veh_trj['is_AV'].iloc[0] == 1:
                temp_facecolor = 'black'
                temp_alpha = 0.99
                heading_angle = single_veh_trj['heading'].iloc[0] * 180 / np.pi
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)
            else:
                if single_veh_trj['is_interest'].iloc[0] == 1:
                    temp_facecolor = 'red'  # 有交互行为的车辆变为红色
                else:
                    if single_veh_trj['obj_type'].iloc[0] == 1:
                        temp_facecolor = 'blue'
                    elif single_veh_trj['obj_type'].iloc[0] == 2:
                        temp_facecolor = 'green'
                    else:
                        temp_facecolor = 'magenta'
                temp_alpha = 0.5
                heading_angle = single_veh_trj['heading'].iloc[0] * 180 / np.pi
                # transform for other vehicles, note that the ego global heading should be added to current local heading
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)
            t = tr + ts
            # note that exact xy needs to to calculated
            veh_length = single_veh_trj['length'].iloc[0]
            veh_width = single_veh_trj['width'].iloc[0]
            ax.add_patch(patches.Rectangle(
                xy=(single_veh_trj['center_x'].iloc[0] - 0.5 * veh_length,
                    single_veh_trj['center_y'].iloc[0] - 0.5 * veh_width),
                width=veh_length,
                height=veh_width,
                linewidth=0.1,
                facecolor=temp_facecolor,
                edgecolor='black',
                alpha=temp_alpha,
                transform=t))
            # add vehicle local id for only vehicle object
            if single_veh_trj['obj_type'].iloc[0] == 1:
                temp_text = plt.text(single_veh_trj['center_x'].iloc[0],
                                     single_veh_trj['center_y'].iloc[0], str(single_veh_id), style='italic',
                                     weight='heavy', ha='center', va='center', color='white', rotation=heading_angle,
                                     size=2.5)
                temp_text.set_path_effects \
                    ([path_effects.Stroke(linewidth=0.7, foreground='black'), path_effects.Normal()])

    # plt.show()
    fig_save_name = 'figure_save/temp_top_view_figure/top_view_segment_' + '__' + 'scenario_' + str(
        scenario_id_in) + '_frame_' + str(frame_id_in) + '_trajectory.jpg'
    plt.savefig(fig_save_name, dpi=300)
    plt.close('all')

def top_view_video_generation(path_2, scenario_id_in,video_save_name):
    # this function generates one top view video based on top view figures from one segment
    img_array = []
    for num in range(1, len(os.listdir('figure_save/temp_top_view_figure/')) + 1):
        image_filename = 'figure_save/temp_top_view_figure/' + 'top_view_segment_' + '__' + 'scenario_' + str(
            scenario_id_in) + '_frame_' + str(num) + '_trajectory.jpg'
        img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    # video_save_name = 'figure_save/top_view_video/' + path_2 +  '_scenario_' + str(scenario_id_in) + '.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('No %d top view video made success' % scenario_id_in)
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/', "*.jpg"))
    for f in filelist:
        os.remove(f)



def generate(seg_trj_single,target_segment,target_scenario):
    file_index = '00' + str(target_segment)
    video_name = f'D:/Data/Git/waymo-od/figure_save/top_view_video/{file_index}_scenario_{target_scenario}.avi'
    if os.path.isdir(video_name):
        print('视频已存在')
        return 0
    else:
        test_state = 0
        filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/', '*.jpg'))
        for f in filelist:
            os.remove(f)
        print(f'segment {target_segment}, scenario: {target_scenario}')
        total_frame_num = seg_trj_single['frame_label'].max()
        print(f'共{total_frame_num}帧')
        for frame_id in range(1, total_frame_num + 1):
            if test_state == 1:
                if frame_id == 5:
                    break
            plot_top_view_single_pic_map(seg_trj_single, target_scenario, frame_id)
        print('No.%d scenario fig has been made,now begin to generate top view viedo.' % target_scenario)
        # ----------video generation------------
        top_view_video_generation(target_segment, target_scenario,video_name)
