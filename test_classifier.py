import os
import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import sys
import pickle

# sys.argv
# 1 pickle file in/output
# 2 positive test npy files
# 3 negative test npy files

pickle_file_path = sys.argv[1]
pos_testing_path = sys.argv[4]
neg_testing_path = sys.argv[5]

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
correct_faces = sum(utils.ensemble_all_vote_all(faces_testing, classifiers))
correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_all_vote_all(non_faces_testing, classifiers))

print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
        + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
        + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
        + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')