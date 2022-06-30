import os
import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import numpy as np
import sys
import pickle

# sys.argv
# 1 positive training npy files
# 2 negative training npy files
# 3 pickle file in/output
# 4 positive testing npy files
# 5 negative testing npy files
# 3 output path
if __name__ == "__main__":
    pos_training_path = sys.argv[1]
    neg_training_path = sys.argv[2]
    pickle_file_path = sys.argv[3]
    
    num_feature_in_stage = [2,1,1]
    # For performance reasons restricting feature size
    min_feature_height = 5
    max_feature_height = 9
    min_feature_width = 5
    max_feature_width = 9
    if not os.path.exists(pickle_file_path):
      print('Loading faces..') 
      faces_ii_training = utils.load_images(pos_training_path)
      print('..done. ' + str(len(faces_ii_training)) + ' faces loaded.\n\nLoading non faces..')
      non_faces_ii_training = utils.load_images(neg_training_path)
      print('..done. ' + str(len(non_faces_ii_training)) + ' non faces loaded.\n')

      # classifiers are haar like features
      classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_feature_in_stage, min_feature_height, max_feature_height, min_feature_width, max_feature_width)
      with open(pickle_file_path,"wb") as filehandler:
        pickle.dump(classifiers,filehandler)
    else:
      with open(pickle_file_path,"rb") as file:
        classifiers = pickle.load(file)