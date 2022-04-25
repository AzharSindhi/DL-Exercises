class BaseLayer:
    def __init__(self, testing_phase=False):
        self.testing_phase = testing_phase
        self.trainable = False
        self.weights = None