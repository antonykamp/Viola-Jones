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
    pickle_file_path = sys.argv[1]
    pos_testing_path = sys.argv[2]
    neg_testing_path = sys.argv[3]
    
    with open(pickle_file_path,"rb") as file:
        classifiers = pickle.load(file)

    print('Loading test faces..')
    faces_testing = utils.load_images(pos_testing_path)
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = utils.load_images(neg_testing_path)
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    essembled_positions_pos = utils.ensemble_all_positions_all(faces_testing, classifiers)
    essembled_positions_neg = utils.ensemble_all_positions_all(non_faces_testing, classifiers)
    
    correct_faces = sum([1 if len(positions) != 0 else 0 for positions in essembled_positions_pos])
    correct_non_faces = sum([1 if len(positions) == 0 else 0 for positions in essembled_positions_neg])

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

    print("---------------")
    print("faces")
    print(essembled_positions_pos)
    print("---------------")
    print("non faces")
    print(essembled_positions_neg)
