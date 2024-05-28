from collections import defaultdict
import numpy as np

def hh_pool(Agree, from_main_to_hh, hh_to_work, barrier, hh, days, hh_id, houses_class):
    
    for day in range(days):
        x_rand = np.random.rand(1000000)

        barrier.wait()   # ожидаем создание индекса
    
        if Agree.value == 1:
            hh_inf = defaultdict(list)

            # получаем индекс из главного
            index = from_main_to_hh.recv()

            # передаем индекс на работы
            hh_to_work.send(index)

            # получаем дни заболевания из главного
            ill_day = from_main_to_hh.recv()

            # передаем дни заболевания на работы
            hh_to_work.send(ill_day)

            curr_hh = hh_id[index]

            for day, HH in zip(ill_day, curr_hh):
                hh_inf[HH].append(day)

            houses_class.set_place_inf(hh_inf)

            # отправляем id
            hh.send(houses_class.infection(x_rand))

            barrier.wait()   # ожидаем заболевания других

            infected_people = from_main_to_hh.recv()
            hh_to_work.send(infected_people)

            houses_class.clean_place(zip(infected_people.sp_hh_id, infected_people.sp_id))
    return 0 
