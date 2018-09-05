import os
from timeit import default_timer as timer
import video_cropper_utils as vid


# --------------------------------
# QUICK RECAP - Maybe show this text to Kim, to see if we should organize it in a different way 

# This file first creates a series of folders named after the video ids based on the file videos.csv
# It then downloads all original videos to the folder ~/videos/original with either 240p (first) or 360p
# Then, for each video a cropped section of 4 seconds (-1, +3) of the shot time is cropped and placed in
# the corresponding subdirectory with that video id in a shot_time directory. Each cropped file has a name of videoId_statId.mp4
# E.g ~/videos/cropped/video_id/shot_time/name.mp4

# if a cropped video is already created ffmpeg skips it. If the file doesn't exists ffmpeg throws an error and skips it.
# creating the entire video dataset only took 30 minutes. So it might be okay to delete all 4 second videos, for a fresh dataset

# For each video we also create a series of 4 second clips without any shots. These videos are made as a sequence starting
# in the middle of two shot times as: diff_in_time // 2, where the distance in seconds is at least 11 seconds (arbitrarily chosen).
# E.g. shot_1 = 10 sec and shot_2 = 21 sec.
# then 11 // 2 = 5  ->  10 + 5 = 15.
# then the no_shot start at 15 sec and ends at 19 sec

# for all 4 second videos each individual frame of that video is saved in its corresponding folder as a .png image. 
# CURRENTLY every other frame just comment out line 210 in video_cropper_utils.py
# This totals 60 images for each video placed in the same folder. Each image has a unique name of vidId_shotTime_frameNumber
# E.g. ~/videos/cropped/video_id/shot_time/name.png

# For every video a corresponding hierachical folder structure is present with optical flow
# optical flow is calculated between two adjacent frames and return a total of 59 * 2 flow images. One for each direction.
# E.g ~/flows/cropped/video_id/shot_time/name.png

# check video_cropper_utils.py for functions used

# Current folder structure

# basketball -
#   - download_videos.py
#   - video_cropper_utils.py
#   - videos.csv
#   - validation.csv
#   - flows
#        - cropped      # empty
#        - no_shot      # empty
#   - videos
#        - cropped      # empty
#        - no_shot      # empty
#        - original     # empty

# enough talking just run the damn code.
# --------------------------------

# timer for recording the running time
# this may be distorted by other processes running as well
# but it gives a reasonable result to time with
start = timer()

# These 7 paths must exist beforehand. All subdirectories depnd on these being available
# path to original file download
save_path = "D:/BASKETBALL/videos/original"

# path to sub directories for cropped videos
save_path_crops = "D:/BASKETBALL/videos/cropped/"

# path to sub directories for cropped videos without shots
save_path_crops_no_shot = "D:/BASKETBALL/videos/no_shots/"

# path to youtube urls in csv file
path_to_urls = "D:/BASKETBALL/videos.csv"

# path to shot times
path_to_shots = "D:/BASKETBALL/validation.csv"

# paths to optical flows
path_to_flows = "D:/BASKETBALL/flows/cropped/"
path_to_flows_no_shot = "D:/BASKETBALL/flows/no_shots/"


# set a fixed duration of the video to 4 seconds in total
duration = "00:00:04"

# placeholder youtube file urls
youTube_urls = []

# placeholder video_ids
video_ids = []

# retrieve a list of video urls
video_urls = vid.load_youtube_urls(path_to_urls)
print("\n\nall urls\n", video_urls, "\n\n")

already_downloaded = vid.get_video_names(save_path)
existing_urls = []
youtube_static = "https://www.youtube.com/watch?v="
for length in range(0, len(already_downloaded)):
    temp = already_downloaded[length].split('.')
    the_id = youtube_static + temp[0]
    existing_urls.append(the_id)

to_download_urls = list(set(video_urls) - set(existing_urls))
print("To download:")
print(to_download_urls)

# retrieve a list of video ids
for video_url in video_urls:
    video_id = vid.get_id(video_url)
    v_id = video_id.split(',')[0]
    video_ids.append(v_id)

print(video_ids)

# create folders for video ids in specified directory
vid.create_folders(save_path_crops, video_ids)
vid.create_folders(save_path_crops_no_shot, video_ids)
vid.create_folders(path_to_flows, video_ids)
vid.create_folders(path_to_flows_no_shot, video_ids)

# ________________________________________________________
# download all the videos from videos.csv
# if errors are thrown but the download skips the video and continues.
# If the files are downloaded this section can be commented out
# saves quite some time
for n in range(0, len(to_download_urls)):
    count = len(to_download_urls) - n
    print("videos remaining:", count)
    try:
        # this function calls a subprocess using the windows command prompt
        vid.download_youtube_video(to_download_urls[n], save_path)
    except Exception as e:
        print(e)
        print("HLLO")
        print(to_download_urls[n][0])
        print("couldnt download video: " + to_download_urls[n] + " continuing with next video")
        pass
print("all videos downloaded")
# ________________________________________________________




#_________________________________________________________
# LOOPING AND CREATING 4 SECOND VIDEOS
for j in range(0, len(video_urls)):
    
    # retrive all shots and stats from a single video url
    video_url = video_urls[j]
    video_shots = vid.load_shots(path_to_shots, video_url)
    
    video_stats = video_shots[1]
    video_shots = video_shots[0]
    
    # get the video vid_id
    vid_id = vid.get_id(video_url)

    # create a series of 4 second videos
    # based on the original videos shot times
    for x in range(0, len(video_shots)):
        
        # get the stat id 
        stat_id = int(video_stats[x])

        # convert shot_time to be 2 seconds before the shot
        shot_time = video_shots[x]
        the_shot = shot_time - 1
        

        # HHmmss instead of HH:mm:ss when saving
        time_to_string = float(shot_time)
        time_to_string = vid.to_gm(time_to_string)
        temp = time_to_string.split(':')
        # HHmmss instead of HH:mm:ss when saving
        time_to_string = temp[0] + temp[1] + temp[2]

        # convert from seconds to HH:mm:ss
        start_time = vid.to_gm(the_shot)

        # make directories on the fly
        print("ids:",len(video_ids))
        print("urls:",len(video_urls))
        haj = os.path.exists(save_path_crops + "/" + video_ids[j] + "/" + time_to_string)
        if (not haj):
            print("Making directory, since it doesnt exist yet")
            os.makedirs(save_path_crops + "/" + video_ids[j] + "/" + time_to_string)
            os.makedirs(path_to_flows + "/" + video_ids[j] + "/" + time_to_string)
        else:
            print("Directory already esists")
            
        # crop and save the video with the name "video_id_stat_id.mp4" in the folder with that video_id
        # save path is a directory for the original file named video_id.mp4
        apath1 = save_path + "/" + vid_id + ".mp4"
        apath2 = save_path_crops + vid_id + "/" + time_to_string + "/" + vid_id + "_" + time_to_string + ".mp4"
        print("The path1: ", apath1)
        print("The path2: ", apath2)
        vid.crop_video(apath1, start_time, duration, apath2)
    print("number of videos with shots cropped", len(video_shots))

    # do the same again but with no shots occuring in the cropped videos, it has different length
    # first obtain new list of times with no shots
    list_of_times = vid.make_times_with_no_shot(video_shots)

    count_no_shots = 0
    # loop over the new list and crop original video accordingly
    for y in range(0, len(list_of_times)):

        if len(list_of_times) == 0:
            print("not enough time between shots to make no_shot clips, continuing")
        else:
            time_to_string = list_of_times[y]
            temp = time_to_string.split(':')
            time_to_string = temp[0] + temp[1] + temp[2]
            
            # make directories on the fly
            if not os.path.exists(save_path_crops_no_shot + "/" + video_ids[j] + "/" + time_to_string):
                os.makedirs(save_path_crops_no_shot + "/" + video_ids[j] + "/" + time_to_string)
                os.makedirs(path_to_flows_no_shot + "/" + video_ids[j] + "/" + time_to_string)
                
            vid.crop_video(save_path + "/" + vid_id + ".mp4", time_to_string, duration, save_path_crops_no_shot + vid_id + "/" + time_to_string + "/" + vid_id + "_" + time_to_string + ".mp4")
            count_no_shots += 1
    print("number of videos without shots cropped", count_no_shots)


print("\n\n\n")
print("FINISHED CROPPING ALL VIDEOS")
print("\n\n\n")
print("STARTING TO MAKE IMAGES")


#__________________________________________________
# MAKE SOME IMAGES
for q in range(0, len(video_ids)):
    print("VIDEO ID", video_ids[q])
    # get the video shots
    video_shots = vid.load_shots(path_to_shots, video_urls[q])
    video_shots = video_shots[0]    

    # retrieve a list of path for each video
    list_of_folders = vid.get_folders(save_path_crops + video_ids[q])
    # same but without shots
    list_of_folders_no_shots = vid.get_folders(save_path_crops_no_shot + video_ids[q])

    # retrieve the video name for a path
    for k in range(0, len(list_of_folders)):
        if len(list_of_folders) == 0:
            print("cropped_folder: no video found, continuing")
            continue
        else:
            video_name = vid.get_video_name(list_of_folders[k])
            print("cropped_folder: converting - ", video_name, " to images")
            if video_name == []:
                print("no video in folder, continuing")
                continue
            else:
                vid.video_to_images(list_of_folders[k], video_name)

    # cant be in same loop since the lists differ in length currently
    for k1 in range(0, len(list_of_folders_no_shots)):
        if len(list_of_folders_no_shots) == 0:
            print("no_shots_folder: no video found, continuing")
            continue
        else:
            video_name_no_shots = vid.get_video_name(list_of_folders_no_shots[k1])
            print("no_shot_folder: converting - ", video_name_no_shots, " to images")
            if video_name == []:
                print("no video in folder, continuing")
                continue
            else:
                vid.video_to_images(list_of_folders_no_shots[k1], video_name_no_shots)

print("\n\n\n")
print("FINISHED MAKING ALL IMAGES")
print("\n\n\n")
print("STARTING TO MAKE OPTICAL FLOW")


# MAKE SOME FLOWS
# ______________________________________________________
print(len(video_ids))
for ids in range(0, len(video_ids)):
    
    # flow folders
    flow_folders = vid.get_folders(path_to_flows + video_ids[ids])
    flow_folders_no_shots = vid.get_folders(path_to_flows_no_shot + video_ids[ids])
    # image folders
    image_folders = vid.get_folders(save_path_crops + video_ids[ids])
    image_folders_no_shot = vid.get_folders(save_path_crops_no_shot + video_ids[ids])

    print("cropped", len(flow_folders))
    print("no_shot", len(flow_folders_no_shots))
    print("\n")

    # loop to create optical flows
    for vid1 in range(0, len(flow_folders)):
        path_to_save = flow_folders[vid1]
        temp = path_to_save.split('\\')
        time = temp[1]
        path_to_images = image_folders[vid1]
        images = vid.get_images_in_folder(path_to_images)
        
        count = len(images)
        if count == 0:
            print("folder: " + path_to_images + " had no images. Continuing")
            continue
        else:
            while count > 1:
                image1, image2, new_list = vid.get_two_images(images) # loop here to save flows
                flow_u, flow_v = vid.make_flow_farneback(image1, image2)
                vid.save_flow(flow_u, flow_folders[vid1], video_ids[ids] + "_" + time + "_" + str(60 - count) + "_U_")
                vid.save_flow(flow_v, flow_folders[vid1], video_ids[ids] + "_" + time + "_" + str(60 - count) + "_V_")
                images = new_list
                count -= 1
    
    for vid2 in range(0, len(flow_folders_no_shots)): 
        path_to_save_no_shot = flow_folders_no_shots[vid2]
        path_to_images_no_shot = image_folders_no_shot[vid2]
        images1 = vid.get_images_in_folder(path_to_images_no_shot)
        count1 = len(images1)
        if count1 == 0:
            print("folder: " + path_to_images + " had no images. Continuing")
            continue
        else:
            while count1 > 1:
                image1, image2, new_list1 = vid.get_two_images(images1) # loop here to save flows
                flow_u1, flow_v1 = vid.make_flow_farneback(image1, image2)
                vid.save_flow(flow_u1, flow_folders_no_shots[vid2], video_ids[ids] + "_" + time + "_" + str(60 - count1) + "_U_")
                vid.save_flow(flow_v1, flow_folders_no_shots[vid2], video_ids[ids] + "_" + time + "_" + str(60 - count1) + "_V_")
                images1 = new_list1
                count1 -= 1

end = timer() 
print("DONE JOHN")
print("Time taken:  ", (end - start)/60, " minutes")