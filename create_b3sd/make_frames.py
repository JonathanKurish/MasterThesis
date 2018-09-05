import os
import B3SD_utils as B3SD
from timeit import default_timer as timer

#   Take each 4 second clip and turn it into frames 
#   Save it in /frames/class_statId/

def make_frames_B3SD():

    start = timer()

    print("makeing frames")
    # top level entries when loading and saving videos
    base_in = B3SD.base_path + B3SD.videos_path_clips
    base_out = B3SD.base_path + B3SD.frames_path

    # retrieve all stat ids, organized in lists per class
    layup, freethrow, point_2, point_3, no_shot, class_occurrences, urls, all_shot_times = B3SD.organize_shots()

    # urls contain a list of urls.
    # Each url in urls has a 1-1 correspondance with all_stats_organized
    all_stats_organized = layup + freethrow + point_2 + point_3 + no_shot

    video_format = ".mp4"

    # find filename_out_path
    filename_out_full_paths = []
    for i in range(0, len(all_stats_organized)):
        print("current stat being processed in make_frames: " + all_stats_organized[i])
        frame_base_name = all_stats_organized[i] + video_format
        try:
            # may fail soft and continue if the input path is incorrect
            B3SD.video_to_images(base_in, frame_base_name, base_out)
        except Exception:
            # hard failure
            print("failed to make frames for: " + base_in + frame_base_name +
                  " Output path: " + base_out + " might be incorrect")
            pass

    end = timer()
    print("make_frames.py: Time taken:  ", (end - start)/60, " minutes")

    return 0