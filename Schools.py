from Place import Place
import numpy as np


class Schools(Place):

    def __init__(self, lmbd, location, place_id, place_len):
        super().__init__(lmbd, location, place_id, place_len)
        self.x_len=10000

    def prob(self, temp):
        return np.repeat(temp, 7) * self.lmbd
