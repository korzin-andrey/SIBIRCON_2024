import numpy as np
import pandas as pd
import datetime
import time

def main_pool(
        Agree, main_to_hh, main_to_school, 
        to_main_hh, to_main_work, to_main_school, 
        barrier, days, susceptible, out_path, number_seed):

    infected, incidence_infected, incubation, incidence_incubation = [], [], [], []
    s_tot = time.perf_counter()
    for day in range(days):
        if len(susceptible[susceptible.illness_day > 2]) != 0:

            curr = susceptible[susceptible.infected == 1]
            index = curr.index.to_numpy() #надо передать через pipe

            ill_day = curr.illness_day.to_numpy() # pipe
            Agree.value = 1
        
            barrier.wait()   # подтверждаем запуск дня в других потоках

            # передача индексов в дома и школы
            main_to_hh.send(index) 
            main_to_school.send(index)

            # передача дней заболевания в дома и школы
            main_to_hh.send(ill_day)
            main_to_school.send(ill_day)


            # Получаем id зараженных
            infected_id_hh = to_main_hh.recv()
            infected_id_work = to_main_work.recv()
            infected_id_school = to_main_school.recv()           

            barrier.wait()   # ждем выполнения процесса заражения остальными

            # удаляем дубликаты и обьединяем
            infected_id = np.concatenate((infected_id_hh, infected_id_work, infected_id_school))
            infected_id = np.unique(infected_id.astype(int))

            infected_people = susceptible[(susceptible.sp_id.isin(infected_id))]

            # отправляем на удаление
            main_to_hh.send(infected_people)

            #infected_work = infected_people[(susceptible.work_id != 0) 
            #                    & (susceptible.age > 17)]
            infected_school = infected_people[(susceptible.work_id != 0) 
                                & (susceptible.age <= 17)]

            # отправляем на удаление
            main_to_school.send(infected_school)


            susceptible.loc[            
                infected_people.index, 
                ['incubation', 'susceptible', 'incubation_max', 'illness_max']
                        ] = [1, 0, 0, 8]

        else:
            Agree.value = 0

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

        print(number_seed, day, curr_incubation, 
            newly_incubation, curr_infected, newly_infected, 
            datetime.datetime.now())
        print()
    print(time.perf_counter() - s_tot)
    return 0
