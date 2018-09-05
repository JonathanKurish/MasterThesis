import os
import csv
import subprocess
import time
import cv2
import numpy as np
from pytube import YouTube
from glob import glob

# Utility functions module for B3SD

# Dataset Classes
classes = ["layup", "freethrow", "2-point", "3-point", "no-shot", "garbage"]

# PREDEFINED STRINGS
# static string used when downloading specific urls
youtube_static = "https://www.youtube.com/watch?v="

# file paths to annotation files
validation_file_path = "validation.csv"
videos_file_path = "videos.csv"

# Folder top level setup
base_path = "D:/b3sd/"
videos_path_original = "videos/original/"
videos_path_clips = "videos/clips/"
frames_path = "jpegs_256/"
flows_path_u = "tvl1_flow/u/"
flows_path_v = "tvl1_flow/v/"

# set format and resolution for download
subtype = 'mp4'
resolution = '240p'

# a single clip duration
duration = "00:00:04.000"

# Loading data definitions are similar.
# By having different functions we can easily
# distinguish them, when used
def load_youtube_urls(file_path):
    youtube_url_list = []
    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            video_url = youtube_static + row[0]
            youtube_url_list.append(video_url)
    return youtube_url_list

def load_stat_ids(file_path):
    stat_ids = []
    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            stat_id = row[0]    
            stat_ids.append(stat_id)
    print("All stat ids loaded")
    return stat_ids

def load_all_urls(file_path):
    urls = []
    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            url = row[1]    
            urls.append(url)
    print("All urls loaded")
    return urls

def load_shot_times(file_path):
    shot_times = []
    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            shot_time = row[3]    
            shot_times.append(shot_time)
    print("All shot times loaded")
    return shot_times

# type must be an int between 1 and 5
# types (1-5) : Layup, free throw, 2-point, 3-point, no_shot
def load_shot_type(file_path, type_chosen):
    shots = []
    offset = 5
    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            shot = int(row[offset + type_chosen])
            shots.append(shot)
    return shots


def load_shots(file_path, video_url):
    # holds all shots of the video
    shot_times = []
    stat_ids = []
    vid_id = get_id(video_url)

    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            video_id = row[1]
            stat_id = row[0]
            # only add shots for the video with the correct id
            if video_id == vid_id:
                time_of_shot = float(row[3])
                shot_times.append(time_of_shot)
                stat_id = int(stat_id)
                stat_ids.append(stat_id)
    return [shot_times, stat_ids]

# return a video id from a url
def get_id(url):
    temp = url.split('=')
    video_id = temp[1]
    return video_id

# download an save a video
# if the video already exists at the specified save_path
# a file is made with the original youtube title as the name
def download_youtube_video(video_url, save_path) :
    #returns an object
    yt = YouTube(video_url)
    
    # rename the downloaded file to the video id later
    video_id = get_id(video_url)

    # retrieve a youtube files list matching the filters, this doesn't download the sound.
    temp = yt.streams.filter(subtype=subtype)\
             .filter(resolution=resolution)\
             .all()
    # pick the first video with resolution at 480p if it exists, otherwise download it at 360p below
    if len(temp) == 0:
        try:
            # get video_title
            filename = yt.streams.filter(subtype=subtype)\
                                 .filter(resolution='360p')\
                                 .first().default_filename
            
            print("video title: ", filename)
            # no resolution at 480p, try lower resolution at 360p
            print("480p was not avialable. downloading 360p video for ", video_url)

            yt.streams.filter(subtype=subtype)\
              .filter(resolution='360p')\
              .first()\
              .download(save_path)
            print("download finished for ", video_url)
            os.rename(save_path + "/" + filename, save_path + "/" + video_id + '.mp4')
            print("saved as ", video_id)
        except FileExistsError:    
            print("\n\n ERROR SAVING FILE \n file " + filename + " already exists. \n")
            os.remove(save_path + "/" + filename)
            os.rename(save_path + "/" + filename, save_path + "/" + video_id + '.mp4')
            print(" removed old file and added new\n\n")
            return 0
    else:

        try:
            # pick the first (and only video at 480p in the list)
            print("downloading 480p video for ", video_url)
            
            # get video_title
            filename = yt.streams.filter(subtype=subtype)\
                                 .filter(resolution=resolution)\
                                 .first().default_filename

            print("video title: ", filename)
            # download video
            yt.streams.filter(subtype=subtype)\
              .filter(resolution=resolution)\
              .first()\
              .download(save_path)

            print("download finished for ", video_url)
            os.rename(save_path + "/" + filename, save_path + "/" + video_id + '.mp4')
            print("saved as ", video_id)       
        except FileExistsError:
            print("\n\n ERROR SAVING FILE \n file " + filename + " already exists. \n")
            os.remove(save_path + "/" + filename)
            os.rename(save_path + "/" + filename, save_path + "/" + video_id + '.mp4')
            print(" removed old file and added new\n\n")
            return 0
    return 0


# crop and save a particular sequence in a video
# both filemane_in and filename_out must contain extension
# time is specified with HH:mm:ss format
# the third input is a duration and NOT ending time !
# e.g. crop_video(file_in, 00:01:00, 00:01:30, file_out)
# result in 1 minute and 30 seconds long clip starting from 00:01:30 in the original file
# the equivalent in the terminal manually is:
# ffmpeg -ss 00:01:00 -i filen_in.mp4 -to 00:01:30 -c copy -pix_fmt yuv420p file_out.mp4
def crop_video(filename_in, start_time, duration, filename_out):
    # colorspace encoding for video-players and very fast clipping
    start = "ffmpeg -n -ss "
    temp = " -async 1 -pix_fmt yuv420p "
    # spawns a subprocess which performs the command to a terminal/cmd
    subprocess.call(start + start_time + " -i " + filename_in + " -t " + duration + temp + filename_out)
    print("\n\n saved clip at: " + filename_in + " to " + filename_out)    
    return 0


# Create a series of subfolders within home_folder based on the list provided
# home_folder is a string
# list items must be strings
def create_folders(home_folder, list_of_folders):
    directory = home_folder
    for x in range(0, len(list_of_folders)):
        if not os.path.exists(directory + list_of_folders[x]):
            os.makedirs(directory + list_of_folders[x])
        else:
            print(directory + list_of_folders[x] + "already existed, proceeding with next folder")
    print("finished creating folders")
    return 0


def get_video_names(path_to_folder):
    video_names = [f for f in os.listdir(path_to_folder) if f.endswith('.mp4')]
    print("current path:\n" + path_to_folder)
    print("video names in current path:\n", video_names)
    if len(video_names) == 0:
        return []
    else:
        return video_names


def organize_shots():
    all_stat_ids = load_stat_ids(validation_file_path)
    all_urls = load_all_urls(validation_file_path)
    all_shot_times = load_shot_times(validation_file_path)
    
    # count the number of occurrences of each class
    class_occurrences = [0,0,0,0,0]

    url_layup = []
    urls_freethrow = []
    url_point_2 = []
    url_point_3 = []
    url_no_shot = []

    t_layup = []
    t_freethrow = []
    t_point_2 = []
    t_point_3 = []
    t_no_shot = []

    layup = []
    freethrow = []
    point_2 = []
    point_3 = []
    no_shot = []
    t1 = load_shot_type(validation_file_path, 1)
    t2 = load_shot_type(validation_file_path, 2)
    t3 = load_shot_type(validation_file_path, 3)
    t4 = load_shot_type(validation_file_path, 4)
    t5 = load_shot_type(validation_file_path, 5)

    for i in range(0, len(all_stat_ids)):
        
        # occurrences
        votes = [t1[i], t2[i], t3[i], t4[i], t5[i]]
        max_value = max(votes)
        class_idx = votes.index(max_value)
        class_occurrences[class_idx] += 1
        
        # save url for that stat
        the_url = all_urls[i]

        # save shot_time for that stat
        the_time = all_shot_times[i]

        # save id in correct list
        the_id = all_stat_ids[i]
        if class_idx == 0:
            layup.append("layup_" + str(the_id))
            url_layup.append(the_url)
            t_layup.append(the_time)
        elif class_idx == 1:
            freethrow.append("freethrow_" + str(the_id))
            urls_freethrow.append(the_url)
            t_freethrow.append(the_time)
        elif class_idx == 2:
            point_2.append("point-2_" + str(the_id))
            url_point_2.append(the_url)
            t_point_2.append(the_time)
        elif class_idx == 3:
            point_3.append("point-3_" + str(the_id))        
            url_point_3.append(the_url)
            t_point_3.append(the_time)
        else:
            no_shot.append("no-shot_" + str(the_id))
            url_no_shot.append(the_url)
            t_no_shot.append(the_time)

    urls = url_layup + urls_freethrow + url_point_2 + url_point_3 + url_no_shot
    shot_times = t_layup + t_freethrow + t_point_2 + t_point_3 + t_no_shot

    return layup, freethrow, point_2, point_3, no_shot, class_occurrences, urls, shot_times

# format seonds to HH:mm:ss:msc
def to_gm(seconds):

    # split to obtain miliseconds for accuracy
    seconds = str(seconds)
    temp = seconds.split('.')
    if len(temp) == 1:
        res = time.strftime('%H:%M:%S', time.gmtime(int(temp[0])))
        res = res + ":000"
    else:
        part_time = temp[0]
        miliseconds = temp[1]
        trailing_zeros = "000"

        res = time.strftime('%H:%M:%S', time.gmtime(int(part_time)))
        res = res + "." + miliseconds + trailing_zeros[len(miliseconds):] 
    return res

# convert video to a sequance of frames
def video_to_images(in_path, video_title, out_path):
    # may fail soft if path is incorrect
    vidcap = cv2.VideoCapture(in_path + video_title)
    success,image = vidcap.read()

    count = 0
    success = True
    while success:
        temp = video_title.split('.')
        video_title = temp[0]
        cv2.imwrite(out_path + video_title + "/" + video_title + "_%d.png" % count, image)      
        success,image = vidcap.read()
        count += 1
    return 0


# FOR FLOWS UTILITIES
# returns a number corresponding to the total
# number of files ending with .png
def get_folder_size(path_to_folder):
    list_of_images = []
    all_frame_names = [f for f in os.listdir(path_to_folder) if f.endswith('.png')]
    return len(all_frame_names)

def read_image(path_to_folder, image_name):
    image = cv2.imread(path_to_folder + "/" + image_name, 0)
    return image

# takes the first two element of a list and returns them
# and a new list where the first element is removed
def get_two_images(the_list):
    if len(the_list) > 1:
        image1, the_list = the_list[-1], the_list[:-1]
        image2 = the_list[-1]
        return image1, image2, the_list
    else:
        return print("list is no longer")

def make_flow_farneback(frame_1, frame_2):
    # Calculate the flow using farnebacks algorithm
    # Parameters may be tuned
    flow = cv2.calcOpticalFlowFarneback(frame_1, frame_2, None, 0.1, 1, 3, 5, 5, 1, 2)
    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]

    # convert to [0,255] for useful information
    #flow_u /= np.max(np.abs(flow_u),axis=0)
    #flow_u *= (255.0/flow_u.max())
    #flow_v /= np.max(np.abs(flow_v),axis=0)
    #flow_v *= (255.0/flow_v.max())
    flow_u = cv2.normalize(flow_u,None,0,255,cv2.NORM_MINMAX)
    flow_v = cv2.normalize(flow_v,None,0,255,cv2.NORM_MINMAX)
    return flow_u, flow_v

def save_flow(image, path_to_folder, name):
    cv2.imwrite(path_to_folder + "/" + name + ".png", image)
    #print("saved flow at:" + path_to_folder + "/" + name + ".png")
    return 0




# OLD Not used
# creates a smaller list of times where
# no shots have been registered.
# it relies on no statistic for 10 seconds
# of the original shots
'''
def make_times_with_no_shot(list_of_shots):
    new_times = []
    for m in range(0, len(list_of_shots) - 1):
        diff_in_time = int(list_of_shots[m+1] - list_of_shots[m])
        if diff_in_time > 10:
            move = diff_in_time // 2
            converted_time = to_gm(list_of_shots[m] + move)
            new_times.append(converted_time)
    return new_times


# create a list from folder names
def get_folders(base_path):
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() ]
    return subfolders

def get_video_name(path):
    video_name = [f for f in os.listdir(path) if f.endswith('.mp4')]
    print("current path", path)
    print("current video_name", video_name)
    if len(video_name) == 0:
        return []
    else:
        return video_name[0]


# returns a list of images in a specific folder
def get_images_in_folder(path_to_folder):
    list_of_images = []
    all_frame_names = [f for f in os.listdir(path_to_folder) if f.endswith('.png')]
    print("current folder: ", path_to_folder)
    print("number of images in folder: ", len(all_frame_names))
    #print("the name of the first image: ", all_frame_names[0])
    if len(all_frame_names) == 0:
        return list_of_images
    else:
        for frame in range(0, len(all_frame_names)):
            image = cv2.imread(path_to_folder + "/" + str(all_frame_names[frame]), 0)
            list_of_images.append(image)
        return list_of_images
'''