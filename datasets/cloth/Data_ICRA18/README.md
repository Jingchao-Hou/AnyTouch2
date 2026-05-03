
# README

## Data structure

-   train_metadata.json and test_metadata.json contains these key-value pairs for each sample: 
    -   cloth_index
    -   trial_index
    -   status: the status of this trial
        -   1: success
        -   -1: slipped after the grasp

-   cloth_metadata.json contains the ground truth of the clothes in the dataset. For each sample, these properties are listed in this file
    -   Fuzziness
    -   Thickness
    -   Smoothness
    -   Wool surfaced
    -   Stretchiness
    -   Endurance
    -   Softness
    -   Wind resistance
    -   Season
    -   Washing method
    -   Taxtile type
    
-   Data/
    -   Data of one trial is located at Data/<cloth_index>/<trial_index>
    -   GelSight data are stored as GelSight_video.mp4 and its background image is background.png. A possible method to extract frames is
    
			ffmpeg -i GelSight_video.mp4 -r 30 GelSight_frames/frame_%04d.png
			
    -   KinectFixed_<cloth_index>_<grasp_index>_<timestamp>.npz contains:
        -   kinect_color: Kinect BGR image
        -   kinect_depth: Kinect depth image
        -   Mat_K2W: Kinect-to-World transformation matrix (See example.py for its usage)
        -   fetch_points: The selected grasp position in world frame
        -   fetch_ori: The selected orientation in world frame

