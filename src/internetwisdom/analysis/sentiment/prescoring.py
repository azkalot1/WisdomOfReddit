from abc import ABC, abstractmethod
from typing import Any
from .state import RedditContent
from dataclasses import asdict
from internetwisdom.analysis.prescoring import RelevancePredictor

class SentimentPrescorer(ABC):
    """
    Abstract base class for prescoring submissions before sentiment analysis.
    """

    @abstractmethod
    def predict(self, submission: RedditContent) -> bool:
        """
        Decide whether to keep (True) or skip (False) a submission.

        Args:
            submission: The loaded submission object.

        Returns:
            bool: True to process, False to skip.
        """
        pass


class AlwaysYesSentimentPrescorer(SentimentPrescorer):
    """
    A prescorer that never filters anything out:
    always returns True for every submission.
    """

    def predict(self, submission: RedditContent) -> bool:
        return True
    
class RelevancePrescorer(SentimentPrescorer):
    def __init__(self, predictor: RelevancePredictor, threshold: float = 0.5):
        self.predictor = predictor
        self.threshold = threshold

    def predict(self, submission: RedditContent) -> bool:
        submission_dict = asdict(submission)
        prediction = self.predictor.predict_single(submission_dict)
        return prediction['probability_relevant'] > self.threshold