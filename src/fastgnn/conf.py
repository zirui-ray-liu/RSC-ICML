class Config:
    def __init__(self):
        self.amp = False
        self.reorder = False
        self.use_approx_op = True
        self.sample_ratio = 1.0
        self.minimal_k = 10
        self.tune_layer_ratio = False

config = Config()