from os import listdir
from os.path import isfile, join
from random import shuffle, randint
from PIL import Image
import numpy as np
from math import floor, ceil

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_batch(video_names, param_dict, phase="train",repeats=10):
    sec1_frame = str2bool(param_dict['sec1_frame'])
    
    if phase=="test" and not sec1_frame:
        video_names, all_frame_numbers = get_test_frame_numbers(video_names, param_dict,repeats)
    else:
        all_frame_numbers = get_train_frame_numbers(video_names, param_dict)

    network_type = param_dict['network_type']
    if network_type=="optical_flow":
        batch = get_batch_1stream_flows(video_names, param_dict, all_frame_numbers)
    elif network_type in ["spatial","grey_frame", "single_grey", "multi_grey","single_grey_diff", "multi_grey_diff"]:
        batch = get_batch_1stream_spatial(video_names, param_dict, all_frame_numbers) 
    elif network_type in ["multiplier",
                            "multi_grey_single_rgb",
                            "multi_and_single_grey",
                            "multi_diff_and_single_grey",
                            "multi_grey_diff_single_rgb"]:
        batch = get_batch_2streams(video_names, param_dict, all_frame_numbers)   
    elif network_type=="late_fusion_two_stream_resnet":
        batch = get_batch_2streams(video_names, param_dict, all_frame_numbers)   
    
    return batch
    
def get_batch_1stream_flows(X, param_dict, all_frame_numbers):
    base_path = param_dict['dataset_name'] + "/tvl1_flow/"
    sec1_frame = str2bool(param_dict['sec1_frame'])
    channels = int(param_dict['channels'])
    jumps = str2bool(param_dict['jumps'])

    batch_flows = []
    for idx, video_name in enumerate(X):
        flows = get_flow_frames(video_name, base_path, all_frame_numbers[idx,:])
        batch_flows.append(flows)
    return batch_flows


def get_batch_1stream_spatial(X, param_dict,all_frame_numbers):
    base_path = param_dict['dataset_name']+"/jpegs_256/"

    batch_single_frame = []

    network_type = param_dict['network_type']

    for idx, video_name in enumerate(X):
        vid_frame_nums = all_frame_numbers[idx]

        if network_type == "spatial":
            frame = get_spatial_frames(video_name, base_path, vid_frame_nums,False)
        elif "diff" in network_type:
            frame = get_spatial_frames(video_name, base_path, vid_frame_nums,True)
            prev_frame = get_spatial_frames(video_name, base_path, vid_frame_nums-1,True)
            frame = frame - prev_frame
        else:
            frame = get_spatial_frames(video_name, base_path, vid_frame_nums,True)
        
        batch_single_frame.append(frame)
    return batch_single_frame

def get_batch_2streams(X, param_dict, all_frame_numbers):
    dataset_name = param_dict['dataset_name']
    base_path1 = dataset_name + "/tvl1_flow/"
    base_path2 = dataset_name + "/jpegs_256/"
    channels = [int(c) for c in param_dict['channels'].split(",")]
    network_type = param_dict['network_type']

    frame_numbers1, frame_numbers2 = all_frame_numbers
    batch1 = []
    batch2 = []

    for idx, video_name in enumerate(X):
        vid_frame_nums1 = frame_numbers1[idx]
        vid_frame_nums2 = frame_numbers2[idx]
        if network_type == "multiplier":
            frame1 = get_flow_frames(video_name, base_path1, vid_frame_nums1)
        elif "diff" in network_type:
            frame1 = get_spatial_frames(video_name, base_path2, vid_frame_nums1,True)
            prev_frame1 = get_spatial_frames(video_name, base_path2, vid_frame_nums1-1,True)
            frame1 = frame1 - prev_frame1
        else:
            frame1 = get_spatial_frames(video_name, base_path2, vid_frame_nums1,True)

        if channels[1] == 3:
            frame2 = get_spatial_frames(video_name, base_path2, vid_frame_nums2,False)
        else:
            frame2 = get_spatial_frames(video_name, base_path2, vid_frame_nums2,True)

        batch1.append(frame1)
        batch2.append(frame2)

        

    return batch1, batch2  


def get_flow_frames(video_name, base_path, flow_numbers, flow_stack="uvuv"):
    xdim, ydim = get_frame_shape(base_path + "u/" + video_name)
    len_flow_numbers = len(flow_numbers)
    channels = len_flow_numbers*2
    flows = np.zeros((xdim,ydim,channels))
    for j in range(0,int((channels/2))-1):
        frame_name = get_frame_name(video_name, flow_numbers[j])
        frame_name_u = base_path + "u/" + video_name + frame_name
        frame_name_v = base_path + "v/" + video_name + frame_name
        if flow_stack=="uvuv":
            u = (j*2)
            v = (j*2)+1
        elif flow_stack=="uuvv":
            u = j
            v = j+len_flow_numbers
        flows[:,:,u] = Image.open(frame_name_u, 'r')
        flows[:,:,v] = Image.open(frame_name_v, 'r')
    return flows

def get_spatial_frames(video_name, base_path, frame_numbers, grey):
    xdim, ydim, zdim = get_frame_shape(base_path + video_name)
    num_frames = len(frame_numbers)
    
    if not grey:
        frame_name = get_frame_name(video_name, frame_numbers[0])
        return np.asarray(Image.open(base_path + video_name + frame_name, 'r'))
    else:
        frames = np.zeros((xdim, ydim, num_frames))
        for i in range(0,num_frames):
            frame_name = get_frame_name(video_name, frame_numbers[i])
            frame = rgb2gray(np.asarray(Image.open(base_path + video_name + frame_name, 'r')))
            frames[:,:,i] = frame
        return frames
         
def get_test_frame_numbers(X, param_dict,repeats):
    channels = [int(c) for c in param_dict['channels'].split(",")]
    

    #channels = int(param_dict['channels'])

    dataset_name = param_dict['dataset_name']
    network_type = param_dict['network_type']
    total_frames = get_total_frames(dataset_name + "/jpegs_256/" + X)

    if network_type in ["multi_grey_diff_single_rgb", "multiplier", "late_fusion_two_stream_resnet"]:
        channels_temp = channels[0]
        channels_spat = channels[1]
        #blablabla
        if network_type == "multiplier":
            channels_temp = int(channels_temp/2)

        all_frame_numbers_temp = np.zeros((repeats, channels_temp))
        #all_frame_numbers_spat = np.asarray([[i] for i in range(1,total_frames-1,max(int(total_frames/repeats),1))][:repeats])

        max_channels = max([channels_temp, channels_spat])
        mid_frame_numbers = [int(i) for i in range(max(int(max_channels/2)+1,3),
                                                  total_frames-int(max_channels/2),
                                                  max(int((total_frames-max_channels)/repeats),1))]

        all_frame_numbers_spat = np.asarray([[i] for i in mid_frame_numbers[:repeats]])
        #Temp
        for i,mid_num in enumerate(mid_frame_numbers[:repeats]):
            frame_range = range(-(ceil(channels_temp/2)-1), floor(channels_temp/2)+1)
            frame_nums_to_add = [int(mid_num + i) for i in frame_range]
            all_frame_numbers_temp[i,:] = frame_nums_to_add
        all_frame_numbers_temp = all_frame_numbers_temp[:len(mid_frame_numbers),:].astype(int)



        
        print(all_frame_numbers_temp)        
        print(all_frame_numbers_spat)

        #all_frame_numbers_spat = np.zeros((repeats, channels_spat))
        repeated_X = [X for i in range(0,len(all_frame_numbers_temp))] 
        all_frame_numbers = [all_frame_numbers_temp, all_frame_numbers_spat]
        return repeated_X, all_frame_numbers
    else:

        if network_type in ["temporal","optical_flow"]:
            channels = int(channels/2)

        all_frame_numbers = np.zeros((repeats, channels))

        if network_type == "spatial" and channels==3:
            all_frame_numbers = np.asarray([[i] for i in range(1,total_frames-1,max(int(total_frames/repeats),1))][:repeats])
        else:
            mid_frame_numbers = [i for i in range(max(int(channels/2)+1,3),
                                                  total_frames-int(channels/2),
                                                  max(int((total_frames-channels)/repeats),1))]
            for i,mid_num in enumerate(mid_frame_numbers[:repeats]):
                frame_range = range(-(ceil(channels/2)-1), floor(channels/2)+1)
                frame_nums_to_add = [mid_num + i for i in frame_range]
                all_frame_numbers[i,:] = frame_nums_to_add
            print("num times repeated:", len(mid_frame_numbers))
            all_frame_numbers = all_frame_numbers[:len(mid_frame_numbers),:]

        repeated_X = [X for i in range(0,len(all_frame_numbers))]
        return repeated_X, all_frame_numbers.astype(int)

def get_train_frame_numbers(X, param_dict):
    network_type = param_dict['network_type']
    dataset_name = param_dict['dataset_name']
    sec1_frame = str2bool(param_dict['sec1_frame'])
    
    channels = [int(c) for c in param_dict['channels'].split(",")]
    if network_type=="spatial":
        channels1 = 1
    else:
        channels1 = channels[0]

    jumps = str2bool(param_dict['jumps'])

    if network_type in ["optical_flow","temporal","multiplier"]:
        channels1 = int(channels1 / 2)

    all_frame_numbers1 = np.zeros((len(X), channels1))
    for idx, video_name in enumerate(X):
        total_frames_in_vid = get_total_frames(dataset_name + "/jpegs_256/" + video_name)
        
        if sec1_frame:
            sec1_frame_num = int(total_frames_in_vid / 4.0)# This is the frame at 1 second
            if channels1 == 1:
                all_frame_numbers1[idx,0] = sec1_frame_num
            else:
                frame_range = range(-(ceil(channels1/2)-1), floor(channels1/2)+1)
                all_frame_numbers1[idx,:] = [sec1_frame_num + i for i in frame_range]
        elif channels1 == 1:
            earliest_start, latest_finish = use_only_middle_50_percent(total_frames_in_vid)
            all_frame_numbers1[idx,0] = randint(earliest_start, latest_finish)
        else:
            start, jump = get_starting_point_and_jumps(total_frames_in_vid, channels=channels1, jumps=jumps)
            frame_nums_to_add = [start+(jump*i) for i in range(0,int(channels1))]
            all_frame_numbers1[idx,:] = frame_nums_to_add
    
    all_frame_numbers1 = all_frame_numbers1.astype(int)

    if len(channels) > 1:
        mid_frame_num = ceil(channels1 / 2) - 1
        all_frame_numbers2 = [[num] for num in all_frame_numbers1[:, mid_frame_num]]

        return all_frame_numbers1, all_frame_numbers2

    return all_frame_numbers1

def get_frame_name(video_name, frame_number):
    if video_name[0] == "v":
        frame_name = "/frame" + str(frame_number).zfill(6) + ".jpg"
    else:
        frame_name = "/" + video_name + "_" + str(frame_number) + ".png"
    return frame_name

def get_frame_shape(path):
    if "b3sd" in path:
        parts = path.split("/")
        full_path = path + "/" + parts[-1] + "_0.png"
    else:
        full_path = path + "/frame000001.jpg"
    return np.asarray(Image.open(full_path, 'r')).shape

def get_total_frames(path):
    return len([f for f in listdir(path) if isfile(join(path, f))])


# Generate frame numbers to select
def get_starting_point_and_jumps(num_frames, channels, jumps=True):
    if jumps:
        if (num_frames < 50):
            jump = int(num_frames / channels)
            start = 1
        else:
            max_jump = int((num_frames-1) / channels)
            jump = randint(1,min(max_jump,15))
            max_start = (num_frames-1) - ((channels -1) * jump)
            start = randint(1,max_start)
        return start, jump
    else:
        finish = num_frames-channels
        if 1 >= finish:
            start = 2
        else:
            start = randint(2,finish)
        return start, 1 

# Use only middle 50 percent of frames
def use_only_middle_50_percent(num_frames):
  earliest_start = int(num_frames/4.0)
  latest_finish = num_frames - earliest_start
  return earliest_start, latest_finish

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])