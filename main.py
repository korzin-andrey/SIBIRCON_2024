from data_load import load_and_preprocess_data, generate_dict
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import warnings
warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None

from Places import Households, Place, Schools, Workplaces
import time
from tqdm import tqdm

#import cProfile
from multiprocessing import Process, Pipe, Value, Barrier
from school_pool import school_pool
from main_pool import main_pool
from work_pool import work_pool
from hh_pool import hh_pool

def main(number_seed, output_folder):
    np.random.seed(number_seed)
    #tot = 0
    # выбор первоначально инфецированных
    I0 = np.random.choice(susceptible.sp_id, init_infected, replace=False)

    susceptible.loc[susceptible.sp_id.isin(I0), 
                    ['infected', 'susceptible', 'illness_day', 'illness_max']] = [1, 0, 3, 8]


    # для истории заражения
    #    
    #    id_susceptible_list, latitude_list, longitude_list, \ 
    #    type_list, id_place_list, days_inf, \
    infected, incidence_infected, incubation, incidence_incubation = [], [], [], []
    
    for i in susceptible[
                        (susceptible.infected == 1) & (susceptible.age > 17) & 
                        (susceptible.work_id != 0)].groupby('work_id').sp_id:
        [dict_work_id[i[0]].remove(j) for j in list(i[1])]     

    for i in susceptible[
                        (susceptible.infected == 1) & (susceptible.age <= 17) & 
                        (susceptible.work_id != 0)].groupby('work_id').sp_id:
        [dict_school_id[i[0]].remove(j) for j in list(i[1])]
        
    for i in susceptible[(susceptible.infected == 1)].groupby('sp_hh_id').sp_id:
        [dict_hh_id[i[0]].remove(j) for j in list(i[1])]


    # тесты, что все заболевшие удалены
    '''
    for i in dict_work_id.keys():
        if len(susceptible[(susceptible.work_id==i)&(susceptible.age>17)&(susceptible.infected==0)])!=len(dict_work_id[i]):
            raise ValueError("Заболевшие не удалились с рабочих мест")

    for i in dict_school_id.keys():
        if len(susceptible[(susceptible.work_id==i)&(susceptible.age<=17)&(susceptible.infected==0)])!=len(dict_school_id[i]):
            raise ValueError("Заболевшие не удалились из школ")
    '''
   
    # обьекты класса мест, где происходит заражение
    houses_class = Households(lmbd, households, dict_hh_id, dict_hh_len)
    works_class = Workplaces(lmbd, workplaces, dict_work_id, dict_work_len)
    schools_class = Schools(lmbd, schools, dict_school_id, dict_school_len)
    
    
    Agree = Value('d', 0)
    barrier = Barrier(4,)

    # 
    main_to_hh, from_main_to_hh = Pipe() # из главного цикла в дома
    hh_to_work, from_hh_to_work = Pipe() # из домов на работы
    main_to_school, from_main_to_school = Pipe() # из главного в школы

    hh, to_main_hh = Pipe() # из домов в главный
    work, to_main_work = Pipe() # с работы в главный
    school, to_main_school = Pipe() # из школы в главный

    main_process = Process(target = main_pool,
            args = (Agree, main_to_hh, main_to_school, 
            to_main_hh, to_main_work, to_main_school,
            barrier, days, susceptible, out_path, number_seed)
                        )

    hh_process = Process(target = hh_pool,
            args = (Agree, from_main_to_hh, hh_to_work, 
            barrier, hh, days, hh_id, houses_class)
                        )

    work_process = Process(target = work_pool,
            args = (Agree, from_hh_to_work, barrier, 
            work, days, work_id, works_class)
                        )

    school_process = Process(target = school_pool,
            args = (Agree, from_main_to_school, barrier,
            school, days, school_id, schools_class)
                            )

    main_process.start()
    hh_process.start()
    work_process.start()
    school_process.start()

    main_process.join()
    hh_process.join()
    work_process.join()
    school_process.join()

    '''
    for day in tqdm(range(days)):
        if len(susceptible[susceptible.illness_day > 2]) != 0:
            x_rand = np.random.rand( 1_000_000 )
            curr = susceptible[susceptible.infected == 1]
            hh_inf, work_inf, school_inf = defaultdict(list), defaultdict(list), defaultdict(list)
            #s_tot = time.perf_counter()
            index = curr.index.to_numpy()
            ill_day = curr.illness_day.to_numpy()
            curr_hh = hh_id[index]
            curr_work = work_id[index]
            curr_school = school_id[index]

            # добавление дня заболевания по id места, где есть больные
            # 8.693490803999339 - весь цикл длиться
            # 3.923998731042957 - длительность внутренности цикла
            #for _, row in curr.iterrows():
            #    s_tot = time.perf_counter()
            #    hh_inf[row.sp_hh_id].append(row.illness_day)
            #    if row.work_id != 0:
            #        if row.age > 17:
            #            work_inf[row.work_id].append(row.illness_day)
            #        else:
            #            school_inf[row.work_id].append(row.illness_day)

            for day, hh, work, school in zip(ill_day, curr_hh, curr_work, curr_school):
                hh_inf[hh].append(day)
                if work != 0:
                    work_inf[work].append(day)
                if school != 0:
                    school_inf[school].append(day)

            #tot += time.perf_counter() - s_tot


            houses_class.set_place_inf(hh_inf)
            works_class.set_place_inf(work_inf)
            schools_class.set_place_inf(school_inf)

            infected_id_hh = houses_class.infection(x_rand)
            infected_id_work = works_class.infection(x_rand)
            infected_id_school = schools_class.infection(x_rand)
        
            # реально заразившиеся
            infected_id = np.concatenate((infected_id_hh, infected_id_work, infected_id_school))
            infected_id = np.unique(infected_id.astype(int))
            infected_people = susceptible[(susceptible.sp_id.isin(infected_id))]
            infected_work = infected_people[(susceptible.work_id != 0) 
                                & (susceptible.age > 17)]
            infected_school = infected_people[(susceptible.work_id != 0) 
                                & (susceptible.age <= 17)]

            # задание параметров заразившимся
            susceptible.loc[            
                infected_people.index, 
                ['incubation', 'susceptible', 'incubation_max', 'illness_max']
                        ] = [1, 0, 0, 8]

            # удаление заразившихся из восприимчивых
            houses_class.clean_place(zip(infected_people.sp_hh_id, infected_people.sp_id)) 
            works_class.clean_place(zip(infected_work.work_id, infected_work.sp_id)) 
            schools_class.clean_place(zip(infected_school.work_id, infected_school.sp_id)) 

        #TODO: считать incidence до обновления счетчиков или после?
        newly_incubation = len(susceptible[(susceptible.incubation_day == 0) & (susceptible.incubation == 1)])
        curr_incubation = int(susceptible[['incubation']].sum())
        newly_infected = len(susceptible[(susceptible.illness_day == 1) & (susceptible.infected == 1)])
        curr_infected = int(susceptible[['infected']].sum())

        infected.append(curr_infected)
        incidence_infected.append(newly_infected)
        incubation.append(curr_incubation)
        incidence_incubation.append(newly_incubation)

        pd.DataFrame(infected).to_csv(out_path +f"prevalence_{number_seed}.csv")
        pd.DataFrame(incidence_infected).to_csv(out_path + f"incidence_{number_seed}.csv")
        pd.DataFrame(incubation).to_csv(out_path + f"prevalence_incubation_{number_seed}.csv")
        pd.DataFrame(incidence_incubation).to_csv(out_path + f"incidence_incubation_{number_seed}.csv")  

        # обновление параметров
        susceptible.loc[susceptible.infected == 1, 'illness_day'] += 1
        susceptible.loc[susceptible.illness_day > susceptible.illness_max, ['infected', 'illness_day']] = 0

        susceptible.loc[susceptible.incubation == 1, 'incubation_day'] += 1
        susceptible.loc[
                susceptible.incubation_day > susceptible.incubation_max, 
                ['infected', 'illness_day', 'incubation', 'incubation_day']
                        ] = [1, 1, 0, 0]

        #print(number_seed, day, curr_incubation, 
        #    newly_incubation, curr_infected, newly_infected, 
        #    datetime.datetime.now())
        #print()
    #print(tot)
    return infected, incubation
    '''


if __name__ == '__main__':
    np.random.seed(1)
    alpha =0.78
    lmbd = 0.17
    init_infected = 10
    days = 150    

    data_folder = 'chelyabinsk_15km/'
    data_path = './data/' + data_folder
    out_path = './results/' + data_folder

    # получаем данные
    people, households, workplaces, schools = load_and_preprocess_data(data_path)
    # задаем невосприимчивых и датафрэйм восприимчивых
    people.loc[
        np.random.choice(people.index, round(len(people) * alpha), replace=False), 
        'susceptible'] = 1
    susceptible = people[people.susceptible == 1]
    susceptible.index = range(len(susceptible))
    susceptible.index = susceptible.index.astype(np.int32)
    hh_id = susceptible.sp_hh_id.to_numpy()
    work_id = susceptible.work_id.to_numpy()
    age = susceptible.age.to_numpy()
    school_id = np.copy(work_id)
    work_id = np.copy(work_id)
    school_id[age>17] = 0
    school_id[age<7] = 0
    work_id[age<18] = 0


    # создаем словари восприимчивых
    dict_hh_id, dict_hh_len, dict_work_id, \
    dict_work_len, dict_school_id, dict_school_len = generate_dict(susceptible)

    # проверяем наличие output директории
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

    start_all = time.perf_counter()
    
    #cProfile.run("main(1, out_path)")
    main(1, out_path)
    print(time.perf_counter() - start_all)
    


    
