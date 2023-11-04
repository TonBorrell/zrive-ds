class UserNotFoundException(Exception):
    """Exception raised for user not found in database when trying to make a prediction"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class PredictionException(Exception):
    """Exception raised for failed prediction"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
