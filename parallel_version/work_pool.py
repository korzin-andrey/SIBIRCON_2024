from collections import defaultdict
import numpy as np

def work_pool(Agree, from_hh_to_work, barrier, work, days, work_id, works_class):
    
    for day in range(days):
        x_rand = np.random.rand(1000000)

        barrier.wait()   # ожидаем создание индекса
    
        if Agree.value == 1:
            work_inf = defaultdict(list)

            # получаем индекс из домов
            index = from_hh_to_work.recv()

            # получаем дни заболевания из домов
            ill_day = from_hh_to_work.recv()

            curr_work = work_id[index]

            for Day, hh_id in zip(ill_day, curr_work):
                if hh_id!=0:
                    work_inf[hh_id].append(Day)

            works_class.set_place_inf(work_inf)

            # отправляем id
            work.send(works_class.infection(x_rand))

            barrier.wait()   # ожидаем заболевания других

            infected_people = from_hh_to_work.recv()
            infected_work = infected_people[(infected_people.work_id != 0) 
                                & (infected_people.age > 17)]

            works_class.clean_place(zip(infected_work.work_id, infected_work.sp_id)) 
