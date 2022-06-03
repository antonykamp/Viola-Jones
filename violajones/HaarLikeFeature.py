import violajones.IntegralImage as ii
from sklearn.ensemble import AdaBoostClassifier

def enum(**enums):
    return type('Enum', (), enums)

FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


class HaarLikeFeature(object):
    """
    Class representing a haar-like feature.
    """

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        """
        Creates a new haar-like feature.
        :param feature_type: Type of new feature, see FeatureType enum
        :type feature_type: violajonse.HaarLikeFeature.FeatureTypes
        :param position: Top left corner where the feature begins (x, y)
        :type position: (int, int)
        :param width: Width of the feature
        :type width: int
        :param height: Height of the feature
        :type height: int
        :param threshold: Feature threshold
        :type threshold: float
        :param polarity: polarity of the feature -1 or 1
        :type polarity: int
        """
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1
        self.adaboost = AdaBoostClassifier()
    
    def get_score(self, int_img):
        """
        Get score for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = ii.sum_region(int_img, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = ii.sum_region(int_img, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)), (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = ii.sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            # top right area
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]), (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            # bottom left area
            third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            # bottom right area
            fourth = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second - third + fourth
        return score
    
    def get_vote(self, int_img):
        """
        Get vote of this feature for given integral image.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 iff this feature votes positively, otherwise 0
        :rtype: int
        """
        score = self.get_score(int_img)
        return self.adaboost.predict([score])[0]
    
    def fit(self, scores, labels):
        """
        Trains a simple classifier given scores and labels
        :param scores: Array with the scores of images with this feature
        :type scores: np.array
        :param labels: Array with labels of the images
        """
        self.adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        self.adaboost.fit([[s] for s in scores], labels)
    
    def predict(self, scores):
        """
        Predicts the labels to an array of scores with this feature
        :param scores: Array with the scores of images
        :type scores: np.ndarray
        :return: Array with predictions of the scores
        :rtype: np.array
        """
        return self.adaboost.predict([[s] for s in scores])
        
    def get_weighted_vote(self, int_img, use_best_vote=False):
        """
        Get vote of this feature for given integral image multiplied by the weight.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :param use_best_vote: Wether it should use the best vote in the image or not
        :type use_best_vote: bool
        :return: 1 iff this feature votes positively, otherwise 0
        :rtype: int
        """
        if use_best_vote:
            return self.weight * self.get_best_vote(int_img)
        return self.weight * self.get_vote(int_img)
    
    def get_score_specified_position(self, int_img, top_left, bottom_right):
        """
        Returns the score of an image of this feature at a specific position. 
        For this, the function edits the top_left and bottom_right instance variable 
        temporarily
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :param top_left: New top left position of the feature
        :param top_left: Tuple with integer
        :param top_left: New bottom right position of the feature
        :param top_left: Tuple with integer
        :return: Score for given feature
        :rtype: int
        """
        old_top_left = self.top_left
        old_bottom_right = self.bottom_right
        
        self.top_left = top_left
        self.bottom_right = bottom_right
        
        vote = self.get_score(int_img)
        
        self.top_left = old_top_left
        self.bottom_right = old_bottom_right
        
        return vote
    
    def get_best_vote(self, int_img):
        """
        Get best vote of an integral image with this feature indepentend of the position.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 if the best position votes positively, otherwise 0
        :rtype: int
        """
        int_img_height, int_img_width = int_img.shape
        
        scores = []
        for left_border in range(int_img_width-self.width-1):
            for top_border in range(int_img_height-self.height-1):
                tmp_top_left = (left_border, top_border)
                tmp_bottom_right = (left_border+self.width, top_border+self.height)
                scores.append(self.get_score_specified_position(int_img, tmp_top_left, tmp_bottom_right))
        return max(self.predict(scores))

