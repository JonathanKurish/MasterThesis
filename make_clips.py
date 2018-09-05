import os
import B3SD_utils as B3SD
from timeit import default_timer as timer

#   Crops all original videos into series of 4 second clips 
#   using ffmpeg and stat_ids. Each new clip is placed in
#   /videos/clips/

#   Each clip is precisely cropped to miliseconds
#   based on the registered shot_time from validation.csv
#   for each individual statId

#   The time format for clip_duration when cropping is
#   HH:mm:ss:msc
#   Hours:minutes:seconds:miliseconds

#   Example of cropping a single video
#   B3SD.crop_video(filename_in_path, start_time, clip_duration, filename_out_path)

def make_clips_B3SD():

    start = timer()

    # top level entries when loading and saving videos
    base_in = B3SD.base_path + B3SD.videos_path_original
    base_out = B3SD.base_path + B3SD.videos_path_clips

    # retrieve all stat ids, organized in lists per class
    layup, freethrow, point_2, point_3, no_shot, class_occurrences, urls, all_shot_times = B3SD.organize_shots()

    # urls contain a list of urls.
    # Each url in urls has a 1-1 correspondance with all_stats_organized
    all_stats_organized = layup + freethrow + point_2 + point_3 + no_shot

    video_format = ".mp4"

    # find filename_in_path
    filename_in_full_paths = []
    for url in urls:
        vid_name = url + video_format
        full_in_path = base_in + vid_name
        filename_in_full_paths.append(full_in_path)

    # find filename_out_path
    filename_out_full_paths = []
    for stat in all_stats_organized:
        new_vid_name = stat + video_format
        full_out_path = base_out + new_vid_name
        filename_out_full_paths.append(full_out_path)



    # find the time of shot and prepare it
    # preparing means, that a conversion between
    # seconds and HH:mm:ss:msc takes place
    start_times = []
    end_times = []
    for shot in all_shot_times:
        # the clip starts 1 second before shot is registered
        new_time = float(shot) - 1
        
        new_end_time = new_time + 4

        # conversion from seconds
        start_time = B3SD.to_gm(new_time)
        end_time = B3SD.to_gm(new_end_time)
        start_times.append(start_time)
        end_times.append(end_time)

    # crop videos
    print("Starting to crop videos")
    for x in range(0, len(filename_in_full_paths)):
        print("current stat being processed in make_clips: " + filename_out_full_paths[x])
        try:
            B3SD.crop_video(filename_in_full_paths[x], start_times[x], B3SD.duration, filename_out_full_paths[x])
        except Exception:
            print("and error occured when cropping a video. Continuing")
            pass


    end = timer()
    print("make_clips.py: Time taken:  ", (end - start)/60, " minutes")

    '''
    # see how data is organized
    print(len(filename_in_full_paths))
    print(len(filename_out_full_paths))
    print(filename_in_full_paths[0])
    print(filename_out_full_paths[0])

    # Note the difference is 1 second between these two 
    print(all_shot_times[0])
    print(start_times[0])
    '''
    return 0