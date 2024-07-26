import numpy as np
import scipy.stats as ss

def pick_incubation_period(number_of_people):
    return ss.lognorm(1.5, loc=4.5).rvs(size=number_of_people)


def pick_illness_period(number_of_people):
    return ss.lognorm(2, loc=8).rvs(size=number_of_people)
