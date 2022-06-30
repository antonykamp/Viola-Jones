class Stage(object):
    """
    Class representing a stage containing features.
    """

    def __init__(self, features):
        self.features = features

    def get_all_possible_positions(self, int_img):
        return list(set.intersection(*[set(f.get_all_possible_positions(int_img)) for f in self.features]))
    
    def get_votes_relative_positions(self, int_img, possible_pos):
        weighted_votes = [f.get_weighted_votes_relative_positions(int_img, possible_pos) for f in self.features]
        weights = [f.weight for f in self.features]
        return sum(weighted_votes) >= 0.5*sum(weights)

    def get_vote(self, int_img):
        return self.get_votes_relative_positions(int_img, [(0,0)])
    
    def does_stage_fit_in_image(self, int_img, pos):
        return all(f.does_feature_fit_in_image(int_img, pos) for f in self.features)
    