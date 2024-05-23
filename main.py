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

def main(number_seed, output_folder):
    np.random.seed(number_seed)

    # выбор первоначально инфецированных
    I0 = np.random.choice(susceptible.sp_id, init_infected, replace=False)

    susceptible.loc[susceptible.sp_id.isin(I0), 
                    ['infected', 'susceptible', 'illness_day', 'illness_max']] = [1, 0, 3, 9]


    # для истории заражения
    #    
    #    id_susceptible_list, latitude_list, longitude_list, \ 
    #    type_list, id_place_list, days_inf, \
    infected, incidence_infected, incubation, incidence_incubation = [], [], [], []
    
    for i in susceptible[
                        (susceptible.infected == 1) & (susceptible.age > 17) & 
                        (susceptible.work_id != 'X')].groupby('work_id').sp_id:
        [dict_work_id[i[0]].remove(j) for j in list(i[1])]     

    for i in susceptible[
                        (susceptible.infected == 1) & (susceptible.age <= 17) & 
                        (susceptible.work_id != 'X')].groupby('work_id').sp_id:
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
    
    
    for day in tqdm(range(days)):
        if len(susceptible[susceptible.illness_day > 2]) != 0:
            x_rand = np.random.rand( 1_000_000 )
            curr = susceptible[susceptible.infected == 1]
            hh_inf, work_inf, school_inf = defaultdict(list), defaultdict(list), defaultdict(list)


            # добавление дня заболевания по id места, где есть больные
            for _, row in curr.iterrows():
                hh_inf[row.sp_hh_id].append(row.illness_day)
                if row.work_id != 'X':
                    if row.age > 17:
                        work_inf[row.work_id].append(row.illness_day)
                    else:
                        school_inf[row.work_id].append(row.illness_day)



            houses_class.set_place_inf(hh_inf)
            works_class.set_place_inf(work_inf)
            schools_class.set_place_inf(school_inf)

            real_inf_hh = houses_class.real_inf(x_rand)
            real_inf_work = works_class.real_inf(x_rand)
            real_inf_school = schools_class.real_inf(x_rand)
        
            # реально заразившиеся
            real_inf = np.concatenate((real_inf_hh, real_inf_school, real_inf_work))
            real_inf = np.unique(real_inf.astype(int))
            real_inf = susceptible[(susceptible.sp_id.isin(real_inf))]
            inf_work = real_inf[(susceptible.work_id != 'X') 
                                & (susceptible.age > 17)]
            inf_school = real_inf[(susceptible.work_id != 'X') 
                                & (susceptible.age <= 17)]

            # задание параметров заразившимся
            susceptible.loc[            
                real_inf.index, 
                ['incubation', 'susceptible', 'incubation_max', 'illness_max']
                        ] = [1, 0, 0, 7]

            # удаление заразившихся из восприимчивых
            houses_class.clean_place(zip(real_inf.sp_hh_id, real_inf.sp_id)) 
            works_class.clean_place(zip(inf_work.work_id, inf_work.sp_id)) 
            schools_class.clean_place(zip(inf_school.work_id, inf_school.sp_id)) 

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
  
    return infected, incubation



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
    
    main(1, out_path)
    
    print(time.perf_counter() - start_all)
    


    
