from Place import Place



class Schools(Place):

    def __init__(self, x_size, lmbd, location, place_id, place_len, vfunc):
        super(x_size, lmbd, location, place_id, place_len, vfunc)
        self.x_len=10000

    def prob(self, temp):
        return np.repeat(temp, 7) * self.lmbd
