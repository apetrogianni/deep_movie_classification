import os
import sys
sys.path.append('..')
import fnmatch
import numpy as np
from multimodal_movie_analysis.analyze_visual.analyze_visual \
    import dir_process_video, process_video


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
    
    if os.path.isfile(videos_path[0]):
        # feature extraction if the input file
        # is only one .mp4 file
        features_stats, f_names_stats, feature_matrix, f_names, shot_change_t = \
            process_video(videos_path[0], 2, True, True, True)
        np.save(videos_path[0] + ".npy", feature_matrix)
        np.save(videos_path[0] + "_features_agg.npy", features_stats)
        # np.save(videos_path[0] + "_stats.npy", f_names)
        # np.save(videos_path[0] + "_f_names.npy", f_names_stats)
    else:
        for folder in videos_path: 
            np_feature_files = fnmatch.filter(os.listdir(folder), '*_features.npy')
            np_fnames_files = fnmatch.filter(os.listdir(folder), '*_f_names.npy')
            
            if len(np_feature_files) <= 0 and len(np_fnames_files) <= 0:
                # if feature extraction is not done for this class-folder, 
                # calculate features for this current folder
                x["x_{0}".format(folder)], \
                name_of_files["paths_{0}".format(folder)], \
                f_names['f_name_{0}'.format(folder)] = \
                    dir_process_video(folder, 2, True, True, True)
            