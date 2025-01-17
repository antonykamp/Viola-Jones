from functools import partial
import numpy as np
from violajones.Stage import Stage
from violajones.HaarLikeFeature import HaarLikeFeature
from violajones.HaarLikeFeature import FeatureTypes
import progressbar
from multiprocessing import Pool

LOADING_BAR_LENGTH = 50


# TODO: select optimal threshold for each feature
# TODO: attentional cascading

def learn(positive_iis, negative_iis, num_feature_in_stage=[], min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1, ):
    """
    Selects a set of classifiers. Iteratively takes the best classifiers based
    on a weighted error.
    :param positive_iis: List of positive integral image examples
    :type positive_iis: list[numpy.ndarray]
    :param negative_iis: List of negative integral image examples
    :type negative_iis: list[numpy.ndarray]
    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int

    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]
    """
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg

    img_height = min([pic.shape[1] for pic in positive_iis + negative_iis])-1
    img_width = min([pic.shape[0] for pic in positive_iis + negative_iis])-1

    # Maximum feature width and height default to image width and height
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    # Create initial labels
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

    images = np.array(positive_iis +negative_iis)

    # Create features for all sizes and locations
    features = _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = len(features)
    feature_indexes = np.arange(num_features)

    num_classifiers = [1]*num_features if len(num_feature_in_stage) == 0 else len(num_feature_in_stage)

    print('Calculating scores for images..')

    scores = np.zeros((num_imgs, num_features))
    bar = progressbar.ProgressBar()
    # Use as many workers as there are CPUs
    pool = Pool(processes=None)
    # Calculate score for each feature on each image
    for i in bar(range(num_imgs)):
        scores[i, :] = np.array(list(pool.map(partial(_get_feature_score, image=images[i]), features)))

    classification_errors = np.zeros(num_features)
    
    # select classifiers
    classifiers = []
    
    print('Selecting classifiers..')
    for i in range(num_classifiers):
        print('Classifier {}/{}'.format(i, num_classifiers))
        
        # Reset weights
        num_pos = int(sum(labels))
        num_neg = int(sum(not l for l in labels))
        pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
        neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
        weights = np.hstack((pos_weights, neg_weights))
        
        votes = np.zeros((num_features, num_imgs))
        features_in_stage = []
        
        for t in range(num_feature_in_stage[i]):
            print('Feature {}/{}'.format(t, num_feature_in_stage[i]))
            
            # normalize weights
            weights *= 1. / np.sum(weights)
            bar = progressbar.ProgressBar()
            for f in bar(range(len(feature_indexes))):
                f_idx = feature_indexes[f]
                f_scores = [scores[img_idx, f_idx] for img_idx in range(num_imgs)]
                feature = features[f_idx]
                # Train ababoost
                feature.fit(f_scores, labels)
                # predict with adaboost
                f_votes = feature.predict(f_scores)
                
                votes[f_idx, :] = np.array(f_votes)
                error = sum(map(lambda img_idx: weights[img_idx] * np.abs(votes[f_idx, img_idx] - labels[img_idx]), range(num_imgs)))
                classification_errors[f] = error
            
            # get best feature, i.e. with smallest error
            min_error_idx = np.argmin(classification_errors)
            best_error = classification_errors[min_error_idx]
            print(best_error)
            best_feature_idx = feature_indexes[min_error_idx]
            beta = best_error/(1-best_error)
            
            # set feature weight
            best_feature = features[best_feature_idx]
            feature_weight = np.log(1/beta)
            best_feature.weight = feature_weight
            features_in_stage.append(best_feature)
            
            # update image weights
            weights = np.array(list(map(lambda img_idx: weights[img_idx] * beta**(int(labels[img_idx] == votes[best_feature_idx, img_idx])), range(num_imgs))))

            # remove feature (a feature can't be selected twice)
            classification_errors = np.delete(classification_errors, min_error_idx)
            feature_indexes = np.delete(feature_indexes, best_feature_idx)
        
        stage = Stage(features_in_stage)
        classifiers.append(stage)
        
        # remove wrong classified images from labels, weight and scores
        wrong_classified_indexes = [img_idx for img_idx in range(num_imgs) if labels[img_idx] != stage.get_vote(images[img_idx])]
        print(wrong_classified_indexes)
        labels = np.delete(labels, wrong_classified_indexes)
        weights = np.delete(weights, wrong_classified_indexes)
        scores = np.delete(scores, wrong_classified_indexes, 0)
        images = np.delete(images, wrong_classified_indexes)
        num_imgs = len(images)

    return classifiers


def _get_feature_score(feature, image):
    return feature.get_score(image)


def _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    print('Creating haar-like features..')
    features = []
    for feature in FeatureTypes:
        # FeatureTypes are just tuples
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
    print('..done. ' + str(len(features)) + ' features created.\n')
    return features
