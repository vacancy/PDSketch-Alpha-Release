__all__ = ['MostPromisingTrajectoryTracker']


class MostPromisingTrajectoryTracker(object):
    """This is a tracker for tracking the most promising next action, used in reinforcement learning settings
    where we don't have a knowledge about the actual transition function or the goal.
    """

    def __init__(self, is_optimistic, threshold: float):
        self.is_optimistic = is_optimistic
        self.threshold = threshold

        self.best_score = float('-inf')
        self.solution = None

    def check(self, new_score):
        if new_score > self.best_score:
            return True
        return False

    def update(self, new_score, solution):
        assert new_score > self.best_score
        self.best_score = new_score
        self.solution = solution

