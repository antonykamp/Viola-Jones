import violajones.IntegralImage as ii
import AdaBoost as ab
import violajones.ViolaJones as vj
import os
from PIL import Image
import numpy as np


def load_images(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            img_arr /= img_arr.max()
            images.append(img_arr)
    return images


if __name__ == "__main__":

    # TODO: select optimal threshold for each feature
    # TODO: attentional cascading

    print('Loading faces..')
    faces_training = load_images('trainingdata/faces')
    faces_ii_training = map(ii.to_integral_image, faces_training)
    print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_training = load_images('trainingdata/nonfaces')
    non_faces_ii_training = map(ii.to_integral_image, non_faces_training)
    print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

    num_classifiers = 20
    # classifiers are haar like features
    classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers=num_classifiers, min_feature_height=4, max_feature_height=10, min_feature_width=4, max_feature_width=10)

    print('Loading test faces..')
    faces_testing = load_images('trainingdata/faces/test')
    faces_ii_testing = map(ii.to_integral_image, faces_testing)
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = load_images('trainingdata/nonfaces/test')
    non_faces_ii_testing = map(ii.to_integral_image, non_faces_testing)
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(vj.classify_all(faces_ii_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(vj.classify_all(non_faces_ii_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

    # Just for fun: putting all haar-like features over each other generates a face-like image
    recon = vj.reconstruct(classifiers, faces_testing[0].shape)
    recon.save('reconstruction.png')