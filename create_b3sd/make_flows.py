import os
import B3SD_utils as B3SD
from timeit import default_timer as timer

#   Take frames and turn it into flows 
#   Save it in /flows/u/class_statId/
#   Save it in /flows/v/class_statId/


def make_frames_B3SD():

    start = timer()

    print("Making flows")
    # top level entries when loading and saving videos
    base_in = B3SD.base_path + B3SD.frames_path
    base_out_u = B3SD.base_path + B3SD.flows_path_u
    base_out_v = B3SD.base_path + B3SD.flows_path_v

    # retrieve all stat ids, organized in lists per class
    layup, freethrow, point_2, point_3, no_shot, class_occurrences, urls, all_shot_times = B3SD.organize_shots()

    # urls contain a list of urls.
    # Each url in urls has a 1-1 correspondance with all_stats_organized
    all_stats_organized = layup + freethrow + point_2 + point_3 + no_shot

    image_format = ".png"

    for i in range(0, len(all_stats_organized)):
        print("current stat being processed in make_flows: " + all_stats_organized[i])
        try:
            num_frames = B3SD.get_folder_size(base_in + all_stats_organized[i])
            print(num_frames)
            
            list_of_images = []
            list_of_names = []
            for frame_number in range(0, num_frames):
                frame_base_name = all_stats_organized[i] + "_" + str(frame_number) + image_format
                image = B3SD.read_image(base_in + all_stats_organized[i] + "/", frame_base_name)
                
                list_of_names.append(frame_base_name)
                list_of_images.append(image)
                #print(base_in + all_stats_organized[i] + "/" + frame_base_name)
            
            #print(len(list_of_images))    
            count = len(list_of_images)
            if count == 0:
                print(base_in + all_stats_organized[i] + " had no images. Continuing")
                continue
            else:
                while count >= 1:
                    img1, img2, new_list = B3SD.get_two_images(list_of_images)
                    flow_u, flow_v = B3SD.make_flow_farneback(img1, img2)
                    B3SD.save_flow(flow_u, base_out_u + all_stats_organized[i], list_of_names[num_frames - count] + "_u")
                    B3SD.save_flow(flow_v, base_out_v + all_stats_organized[i], list_of_names[num_frames - count] + "_v")
                    images = new_list
                    count -= 1
        except Exception:
            print("And error occured when making flows. Continuing")
            pass


    end = timer()
    print("make_frames.py: Time taken:  ", (end - start)/60, " minutes")

    return 0