from collections import defaultdict
import numpy as np

def school_pool(Agree, from_main_to_school, barrier, school, days, school_id, schools_class):
    
    for day in range(days):
        x_rand = np.random.rand(1000000)

        barrier.wait()   # ожидаем создание индекса
    
        if Agree.value == 1:
            school_inf = defaultdict(list)

            # получаем индекс из главного
            index = from_main_to_school.recv()

            # получаем дни заболевания из главного
            ill_day = from_main_to_school.recv()

            curr_school = school_id[index]

            for Day, hh_id in zip(ill_day, curr_school):
                if hh_id!=0:
                    school_inf[hh_id].append(Day)

            schools_class.set_place_inf(school_inf)

            # отправляем id
            school.send(schools_class.infection(x_rand))

            barrier.wait()   # ожидаем заболевания других

            infected_school = from_main_to_school.recv()

            schools_class.clean_place(zip(infected_school.work_id, infected_school.sp_id)) 
