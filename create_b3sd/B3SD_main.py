import download_videos
import make_clips
import make_frames
import make_flows
from timeit import default_timer as timer

# main entry point for creating the B3SD dataset

# remember that files videos.csv and validation.csv must
# be at the same directory level as this file

# edit the following in B3SD_utils.py as needed:
# base_path = "D:/datasets/B3SD/"
# all other fixed paths are relative to this path
# if base_path does not exist beforehand it is created

start = timer()

# Order of excecution must be sequential in this case

print("\n\nStarted downloading videos\n\n")
download_videos.download_videos_B3SD()

print("\n\nStarted making clips\n\n")
make_clips.make_clips_B3SD()

print("\n\nStarted making frames\n\n")
make_frames.make_frames_B3SD()

#print("\n\nStarted making flows\n\n")
#make_flows.make_flows_B3SD()


end = timer()
print("\n\nFinished making the dataset\n\n")
print("Total time taken:  ", (end - start)/60, " minutes")