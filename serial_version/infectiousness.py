import numpy as np


def pick_incubation_period(number_of_people):
    incubation_period_min = 2
    incubation_period_max = 11
    return np.random.randint(incubation_period_min, incubation_period_max, number_of_people)


def pick_illness_period(number_of_people):
    illness_period_min = 2
    illness_period_max = 14
    return np.random.randint(illness_period_min, illness_period_max, number_of_people)
