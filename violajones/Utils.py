from turtle import pos
import numpy as np
from PIL import Image
from violajones.HaarLikeFeature import FeatureType
from functools import partial
import violajones.IntegralImage as ii
import os
import progressbar


def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_weighted_vote(int_img) for c in classifiers]) >= 0.5*sum([c.weight for c in classifiers]) else 0


def ensemble_all_positions(int_img, classifiers, just_check=False):
    possible_pos = np.array(classifiers[0].get_all_possible_positions(int_img))
    possible_pos_idx = np.arange(len(possible_pos))
    num_positions = len(possible_pos)
    
    print("Collect positions by checking stages")
    bar = progressbar.ProgressBar()
    for stage in bar(classifiers):
        
        # If no position passed the last stage, no faces is in the image
        if len(possible_pos) == 0:
            break
        
        # Look if the next feature fit in the image at the relative position
        not_fit_images = [pos_idx for pos_idx in range(num_positions) if not stage.does_stage_fit_in_image(int_img, pos)]
        possible_pos = np.delete(possible_pos, not_fit_images)
        possible_pos_idx = np.delete(possible_pos_idx, not_fit_images)
        num_positions = len(possible_pos)
        
        # Get votes of the image at the relative positions
        votes = stage.get_votes_relative_positions(int_img, possible_pos)
        
        # update positions to use in next stage if passed stage
        not_passed_image_idx = [pos_idx for pos_idx in range(num_positions) if not votes[pos_idx]]
        possible_pos = np.delete(possible_pos, not_passed_image_idx)
        possible_pos_idx = np.delete(possible_pos_idx, not_passed_image_idx)
        num_positions = len(possible_pos)
        
    if just_check:
        return len(possible_pos) != 0
    return possible_pos
            
def ensemble_all_positions_all(int_imgs, classifiers):
    """From list containing images collect positions in images with faces"""
    collect_partial = partial(ensemble_all_positions, classifiers=classifiers)
    return list(map(collect_partial, int_imgs))


def ensemble_all_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1. Unlike ensemble_vote, the features are not only applied to one position.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return ensemble_all_positions(int_img, classifiers, just_check=True)


def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def ensemble_all_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1. Unlike ensemble_vote_all, the features are not only applied to one position.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_all_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def reconstruct(classifiers, img_size):
    """
    Creates an image by putting all given classifiers on top of each other
    producing an archetype of the learned class of object.
    :param classifiers: List of classifiers
    :type classifiers: list[violajones.HaarLikeFeature.HaarLikeFeature]
    :param img_size: Tuple of width and height
    :type img_size: (int, int)
    :return: Reconstructed image
    :rtype: PIL.Image
    """
    image = np.zeros(img_size)
    for c in classifiers:
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
    image -= image.min()
    image /= image.max()
    image *= 255
    result = Image.fromarray(image.astype(np.uint8))
    return result


def load_images(path):
    images = [] 
    bar = progressbar.ProgressBar()
    for _file in bar(os.listdir(path)):
        if _file.endswith('.npy'):
            img_arr = np.load(os.path.join(path, _file))
            # img_arr /= img_arr.max()
            int_arr = ii.to_integral_image(img_arr)
            images.append(int_arr)
    return images
