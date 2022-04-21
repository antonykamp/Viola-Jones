import os
import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import numpy as np
import sys

# sys.argv
# 1 positive training np files
# 2 negative training npy files
# 3 output path
if __name__ == "__main__":
    pos_training_path = sys.argv[1]
    neg_training_path = sys.argv[2]
    pos_testing_path = sys.argv[3]
    neg_testing_path = sys.argv[4]

    num_classifiers = 3
    # For performance reasons restricting feature size
    min_feature_height = 5
    max_feature_height = 9
    min_feature_width = 5
    max_feature_width = 9

    print('Loading faces..') 
    faces_ii_training = utils.load_images(pos_training_path)
    print('..done. ' + str(len(faces_ii_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_ii_training = utils.load_images(neg_training_path)
    print('..done. ' + str(len(non_faces_ii_training)) + ' non faces loaded.\n')

    # classifiers are haar like features
    classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)
    
    if len(sys.argv) < 3:
      exit()
      
    print('Loading test faces..')
    faces_testing = utils.load_images(pos_testing_path)
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = utils.load_images(neg_testing_path)
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(utils.ensemble_vote_all(faces_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_vote_all(non_faces_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')
