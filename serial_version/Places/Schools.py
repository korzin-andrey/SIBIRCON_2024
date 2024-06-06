from . import Place
import numpy as np


class Schools(Place):

    def __init__(self, lmbd, location, place_id, place_len):
        super().__init__(lmbd, location, place_id, place_len)
        self.number_of_contacts = 7

    def prob(self, temp):
        return np.repeat(temp, self.number_of_contacts) * self.lmbd
