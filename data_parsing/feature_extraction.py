import os
import sys
sys.path.append('..')
import fnmatch

from multimodal_movie_analysis.analyze_visual.analyze_visual import dir_process_video

def feature_extraction(videos_path):
    """
    Extracts features from videos and saves them
    to numpy arrrays (using dir_process_video() function 
    from "multimodal_movie_analysis" repo)

        Parameters: 
            videos_path (list): directories of videos
    """
    x = {}
    name_of_files = {}
    f_names = {}

    for folder in videos_path: 
        np_feature_files = fnmatch.filter(os.listdir(folder), '*_features.npy')
        np_fnames_files = fnmatch.filter(os.listdir(folder), '*_f_names.npy')
        
        # check if feature extraction 
        # is done for this class-folder
        if len(np_feature_files) <= 0 and len(np_fnames_files) <= 0:
            # calculate features for current folder

            x["x_{0}".format(folder)], \
            name_of_files["paths_{0}".format(folder)], \
            f_names['f_name_{0}'.format(folder)] = \
                dir_process_video(folder, 2, True, True, True)
