import optuna
from tqdm import tqdm
import time
from Places import Households, Place, Schools, Workplaces
from data_load import load_and_preprocess_data, generate_dict
from infectiousness import pick_illness_period, pick_incubation_period
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
from functools import partial
import warnings
import multiprocessing as mp
from multiprocessing import Process, Queue, current_process, freeze_support
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None


EXIT_THRESHOLD = 0.9


def callback(study, trial):
    if trial.value > EXIT_THRESHOLD:
        study.stop()


def main(number_seed, out_path, susceptible, alpha, lmbd, init_infected, days,
         dict_work_id, dict_hh_id, dict_school_id, households, people, workplaces, schools,
         dict_hh_len, dict_work_len, dict_school_len, hh_id, work_id, school_id, age):
    # np.random.seed(number_seed)
    # choose initial infectious sp_id
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

    for day in tqdm(range(days)):
        if len(susceptible[susceptible.illness_day > 2]) != 0:
            x_rand = np.random.rand(10_000_000)
            curr = susceptible[susceptible.infected == 1]
            hh_inf, work_inf, school_inf = defaultdict(
                list), defaultdict(list), defaultdict(list)
            index = curr.index.to_numpy()
            ill_day = curr.illness_day.to_numpy()
            curr_hh = hh_id[index]
            curr_work = work_id[index]
            curr_school = school_id[index]

            for day, hh, work, school in zip(ill_day, curr_hh, curr_work, curr_school):
                hh_inf[hh].append(day)
                if work != 0:
                    work_inf[work].append(day)
                if school != 0:
                    school_inf[school].append(day)

            houses_class.set_place_inf(hh_inf)
            works_class.set_place_inf(work_inf)
            schools_class.set_place_inf(school_inf)

            infected_id_hh = houses_class.infection()
            infected_id_work = works_class.infection()
            infected_id_school = schools_class.infection()

            # реально заразившиеся
            infected_id = np.concatenate(
                (infected_id_hh, infected_id_work, infected_id_school))
            infected_id = np.unique(infected_id.astype(int))
            infected_people = susceptible[(
                susceptible.sp_id.isin(infected_id))]
            infected_work = infected_people[(susceptible.work_id != 0)
                                            & (susceptible.age > 17)]
            infected_school = infected_people[(susceptible.work_id != 0)
                                              & (susceptible.age <= 17)]

            # задание параметров заразившимся
            number_of_infected = len(infected_people)
            incubation_period_arr = pick_incubation_period(number_of_infected)
            illness_period_arr = pick_illness_period(number_of_infected)
            infectious_people_params = np.array([np.ones(number_of_infected),
                                                np.zeros(number_of_infected),
                                                incubation_period_arr,
                                                illness_period_arr]).T
            susceptible.loc[
                infected_people.index,
                ['incubation', 'susceptible', 'incubation_max', 'illness_max']
            ] = infectious_people_params

            # удаление заразившихся из восприимчивых
            houses_class.clean_place(
                zip(infected_people.sp_hh_id, infected_people.sp_id))
            works_class.clean_place(
                zip(infected_work.work_id, infected_work.sp_id))
            schools_class.clean_place(
                zip(infected_school.work_id, infected_school.sp_id))

        # TODO: считать incidence до обновления счетчиков или после?
        newly_incubation = len(
            susceptible[(susceptible.incubation_day == 0) & (susceptible.incubation == 1)])
        curr_incubation = int(susceptible[['incubation']].sum())
        newly_infected = len(
            susceptible[(susceptible.illness_day == 1) & (susceptible.infected == 1)])
        curr_infected = int(susceptible[['infected']].sum())

        infected.append(curr_infected)
        incidence_infected.append(newly_infected)
        incubation.append(curr_incubation)
        incidence_incubation.append(newly_incubation)

        pd.DataFrame(infected, columns=['data']).to_csv(
            out_path + f"prevalence_{number_seed}.csv")
        pd.DataFrame(incidence_infected, columns=['data']).to_csv(
            out_path + f"incidence_{number_seed}.csv")
        pd.DataFrame(incubation, columns=['data']).to_csv(
            out_path + f"prevalence_incubation_{number_seed}.csv")
        pd.DataFrame(incidence_incubation, columns=['data']).to_csv(
            out_path + f"incidence_incubation_{number_seed}.csv")

        # обновление параметров
        susceptible.loc[susceptible.infected == True, 'illness_day'] += 1
        susceptible.loc[susceptible.illness_day >
                        susceptible.illness_max, ['infected', 'illness_day']] = 0

        susceptible.loc[susceptible.incubation == True, 'incubation_day'] += 1
        susceptible.loc[
            susceptible.incubation_day > susceptible.incubation_max,
            ['infected', 'illness_day', 'incubation', 'incubation_day']
        ] = [1, 1, 0, 0]

    return infected, incubation


if __name__ == '__main__':
    np.random.seed(1)
    alpha = 0.4
    lmbd = 0.3
    # init_infected = 10
    days = 150

    data_folder = 'spb/'
    data_path = './data/' + data_folder
    out_path = './results/' + data_folder

    # получаем данные
    people, households, workplaces, schools = load_and_preprocess_data(
        data_path)
    # задаем невосприимчивых и датафрэйм восприимчивых
    people.loc[
        np.random.choice(people.index, round(
            len(people) * alpha), replace=False),
        'susceptible'] = 1
    susceptible = people[people.susceptible == 1]
    susceptible.index = range(len(susceptible))
    susceptible.index = susceptible.index.astype(np.int32)
    hh_id = susceptible.sp_hh_id.to_numpy()
    work_id = susceptible.work_id.to_numpy()
    age = susceptible.age.to_numpy()
    school_id = np.copy(work_id)
    work_id = np.copy(work_id)
    school_id[age > 17] = 0
    school_id[age < 7] = 0
    work_id[age < 18] = 0

    # создаем словари восприимчивых
    dict_hh_id, dict_hh_len, dict_work_id, \
        dict_work_len, dict_school_id, dict_school_len = generate_dict(
            susceptible)

    # проверяем наличие output директории
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

    start_all = time.perf_counter()

    number_of_launches = 10

    def objective_optuna(trial, real_data):
        alpha = trial.suggest_float('alpha', 0.35, 0.45)
        lmbd = trial.suggest_float('lmbd', 0.25, 0.35)
        total_results = [[] for i in range(number_of_launches)]

        simulation(number_of_launches)
        for i in range(number_of_launches):
            total_results[i] = pd.read_csv(
                out_path + r'incidence_{}.csv'.format(i))['data'].to_list()
        sim_data = np.mean(total_results, axis=0)
        plt.plot(real_data, '--o')
        plt.plot(sim_data) 
        plt.show()
        mismatch = r2_score(real_data, sim_data[:len(real_data)])
        return mismatch
    
    epid_data = pd.read_csv(
        './epid_data/SPb.COVID-19.united.csv')['CONFIRMED.sk'][670:770]
    init_infected = epid_data.to_list()[0]
    def simulation(number_of_launches):
        with mp.Pool(number_of_launches) as pool:
            output = pool.map(partial(main, out_path=out_path, susceptible=susceptible, alpha=alpha,
                                      lmbd=lmbd, init_infected=init_infected, days=days,
                                      dict_work_id=dict_work_id, dict_hh_id=dict_hh_id, dict_school_id=dict_school_id,
                                      households=households, people=people, workplaces=workplaces, schools=schools,
                                      dict_hh_len=dict_hh_len, dict_work_len=dict_work_len,
                                      dict_school_len=dict_school_len, hh_id=hh_id, work_id=work_id,
                                      school_id=school_id, age=age),
                              range(number_of_launches))
            
    study = optuna.create_study(direction='maximize')
    epid_data = pd.read_csv(
        './epid_data/SPb.COVID-19.united.csv')['CONFIRMED.sk'][670:770].to_list()
    days = len(epid_data)
    study.optimize(lambda trial: objective_optuna(trial, epid_data),
                   callbacks=[callback])
    print("Best parameters:", study.best_params)
    print(round((time.perf_counter() - start_all), 3), "sec")
