import B3SD_utils as B3SD
from timeit import default_timer as timer

#   1: downloads a series of youtube videos
#   see videos.csv for urls
#   Downloads are placed in B3SD/videos/original/

#   2: Creates the entire structure of
#   folders before cropping videos
#   see validation.csv for annotations

#   resulting folder organization:
#   - B3SD
#       - videos
#           - original
#               - youtubeId.mp4
#
#           - clips
#               - class_statId.mp4
#       
#       - frames
#           - class_statId
#                   - class_statId.png
#
#       - flows
#           - u
#               - class_statId
#                       - class_statId.png
#
#           - v
#               - class_statId
#                       - class_statId.png

def download_videos_B3SD():

    start = timer()

    # Create top level folder structure
    top_level_paths = [B3SD.videos_path_original,
                       B3SD.videos_path_clips,
                       B3SD.frames_path,
                       B3SD.flows_path_u,
                       B3SD.flows_path_v]

    B3SD.create_folders(B3SD.base_path, "")
    B3SD.create_folders(B3SD.base_path, top_level_paths)

    # retrieve lists of stat ids and urls
    all_stat_ids = B3SD.load_stat_ids(B3SD.base_path + B3SD.validation_file_path)
    video_urls = B3SD.load_youtube_urls(B3SD.base_path + B3SD.videos_file_path)

    # When downloading full length videos skip download
    # of videos already present in folder
    already_downloaded = B3SD.get_video_names(B3SD.base_path + B3SD.videos_path_original)
    existing_urls = []

    for url in already_downloaded:
        name = url.split('.')
        the_id = B3SD.youtube_static + name[0] # full url
        existing_urls.append(the_id)

    to_download_urls = list(set(video_urls) - set(existing_urls))
    print("\nvideos not downloaded yet: \n", to_download_urls)


    # Remove comment to actually perform download of missing videos
    # This part takes time depending on the number of videos 
    
    # download all the videos from videos.csv
    for n in range(0, len(to_download_urls)):
        count = len(to_download_urls) - n
        print("videos remaining:", count)
        try:
            # this function uses the subprocess module, spawning a new process
            B3SD.download_youtube_video(to_download_urls[n], B3SD.videos_path_original)
        except Exception as e:
            print(e)
            # Youtube declined the download request
            # The video url might not be available in certain countries
            # A timeout occured during connection to the server
            print("couldnt download video: " +
                   to_download_urls[n] +
                   " continuing with next video")
            # try the next url
            pass
    print("Finished trying to download all files")
    


    # Create sub level folder structure

    # first save stat_ids in their right class
    layup, freethrow, point_2, point_3, no_shot, class_occurences, urls, shot_times = B3SD.organize_shots()

    # /frames/class_statId/
    B3SD.create_folders(B3SD.base_path + B3SD.frames_path,
                        layup + freethrow + point_2 + point_3 + no_shot)

    # /flows/class_statId/
    B3SD.create_folders(B3SD.base_path + B3SD.flows_path_u,
                        layup + freethrow + point_2 + point_3 + no_shot)

    B3SD.create_folders(B3SD.base_path + B3SD.flows_path_v,
                        layup + freethrow + point_2 + point_3 + no_shot)


    end = timer()
    print("download_videos.py: Time taken:  ", (end - start)/60, " minutes")

    return 0