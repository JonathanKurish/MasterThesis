import os
import csv
import subprocess
import time
import cv2
import numpy as np
from pytube import YouTube
from glob import glob

youtube_base_path = "https://www.youtube.com/watch?v="

# DONWLOAD VIDEO VARIABLES
# set format and resolution
# fails if the format cannot be obtained at 240p or 360p
subtype = 'mp4'
resolution = '240p' # change if desired

# format seonds to HH:mm:ss
def to_gm(seconds):
    res = time.strftime('%H:%M:%S', time.gmtime(seconds))
    return res

# return a video id from a url
def get_id(url):
    temp = url.split('=')
    video_id = temp[1]
    return video_id

# Returns a list of full youtube urls which cn be downloaded
def load_youtube_urls(file_path):
    youtube_url_list = []

    with open(file_path) as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            vid, title = row[0].split(",", 1)
            video_url = youtube_base_path + vid
            youtube_url_list.append(video_url)
    return youtube_url_list

# return two lists one with shot times in seconds and the stat_ids
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
            # pick the first (and only video at 240p in the list)
            print("downloading 240p video for ", video_url)
            
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
    temp = " -c copy -pix_fmt yuv420p "
    # spawns a subprocess which performs the command to a terminal/cmd
    subprocess.call(start + start_time + " -i " + filename_in + " -to " + duration + temp + filename_out)
    print("\n\n saved clip at: " + filename_in + " to " + filename_out)    
    return 0


# Create a series of subfolders within home_folder based on the list provided
# home_folder is a string
# list items must be strings
def create_folders(home_folder, list_of_video_titles):
    directory = home_folder
    for x in range(0, len(list_of_video_titles)):
        if not os.path.exists(directory + list_of_video_titles[x]):
            os.makedirs(directory + list_of_video_titles[x])
        #else:
            #print(directory + list_of_video_titles[x] + "already existed, proceeding with next folder")
    print("finished creating folders")
    return 0


# creates a smaller list of times where
# no shots have been registered.
# it relies on no statistic for 10 seconds
# of the original shots
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

# For each subfolder
# take the small clip, convert it
# to a series of individual frames
def video_to_images(path, video_title):

    # remove these prints later
    print("IM CURRENTLY ONLY SAVING EVERY OTHER FRAME")
    print("FIX ME IN video_cropper_utils.py - video_to_images() ")
    
    vidcap = cv2.VideoCapture(path +"/" + video_title)
    success,image = vidcap.read()

    count = 0
    success = True
    while success:
        temp = video_title.split('.')
        video_title = temp[0]
        cv2.imwrite(path + "/" + video_title + "_%d.png" % count, image)      

        success, image = vidcap.read()
        # Here we skip a frame
        # only every second frame is saved
        # space consumption
        success,image = vidcap.read()
        count += 1
    return 0


# OPTICAL FLOW HANDLING
def make_flow_farneback(frame_1, frame_2):
    # Calculate the flow using farnebacks algorithm
    # Parameters may be tuned
    flow = cv2.calcOpticalFlowFarneback(frame_1, frame_2, None, 0.5, 3, 30, 3, 5, 1.2, 0)
    flow_u = flow[:,:,0]
    flow_v = flow[:,:,1]

    # convert to [0,255] for extrapolating differences visually
    flow_u /= np.max(np.abs(flow_u),axis=0)
    flow_u *= (255.0/flow_u.max())
    flow_v /= np.max(np.abs(flow_v),axis=0)
    flow_v *= (255.0/flow_v.max())

    return flow_u, flow_v


# takes the first two element of a list and returns them
# and a new list where the first element is removed
def get_two_images(the_list):
    if len(the_list) > 1:
        image1, the_list = the_list[-1], the_list[:-1]
        image2 = the_list[-1]
        return image1, image2, the_list
    else:
        return print("list is no longer")

# returns a list of images in a specific folder
def get_images_in_folder(path_to_folder):
    list_of_images = []
    all_frame_names = [f for f in os.listdir(path_to_folder) if f.endswith('.png')]
    print("current folder: ", path_to_folder)
    print("number of images in folder: ", len(all_frame_names))
    if len(all_frame_names) == 0:
        return list_of_images
    else:
        for frame in range(0, len(all_frame_names)):
            image = cv2.imread(path_to_folder + "/" + str(all_frame_names[frame]), 0)
            list_of_images.append(image)
        return list_of_images

# Save an image to a specific path
def save_flow(image, path_to_folder, name):
    cv2.imwrite(path_to_folder + "/" + name + ".png", image)
    print("saved flow at:" + path_to_folder + "/" + name + ".png")
    return 0

def get_video_names(path):
    video_names = [f for f in os.listdir(path) if f.endswith('.mp4')]
    if len(video_names) == 0:
        return []
    else:
        return video_names