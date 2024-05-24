import pandas as pd
import numpy as np

# в папке по пути data_path должны находиться файлы: 
#   -- people.txt
#   -- households.txt 
#   -- workplaces.txt 
#   -- schools.txt

def load_and_preprocess_data(data_path):
    # данные о популяции людей
    data = pd.read_csv(data_path + 'people.txt', sep='\t', index_col=0)
    data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]

    # данные о домовладениях
    households = pd.read_csv(data_path + 'households.txt', sep='\t')
    households = households[['sp_id', 'latitude', 'longitude']]

    # данные о рабочих местах
    workplaces = pd.read_csv(data_path + 'workplaces.txt', sep='\t')
    workplaces = workplaces[['sp_id', 'latitude', 'longitude']]

    # данные о школах
    schools = pd.read_csv(data_path + 'schools.txt', sep='\t')
    schools = schools[['sp_id', 'latitude', 'longitude']]

    # подготовка загруженных данных
    data[['sp_id', 'sp_hh_id', 'age']] = data[['sp_id', 'sp_hh_id', 'age']].astype(int)
    data[['work_id']] = data[['work_id']].replace('X', 0).astype(int)
    #data = data.sample(frac=1)

    households[['sp_id']] = households[['sp_id']].astype(int)
    households[['latitude', 'longitude']] = households[['latitude', 'longitude']].astype(float)
    households.index = households.sp_id

    workplaces[['sp_id']] = workplaces[['sp_id']].astype(int)
    workplaces[['latitude', 'longitude']] = workplaces[['latitude', 'longitude']].astype(float)

    schools[['sp_id']] = schools[['sp_id']].astype(int)
    schools[['latitude', 'longitude']] = schools[['latitude', 'longitude']].astype(float)

    data['susceptible'] = 0
    data['infected'] = 0
    data['illness_day'] = 0
    data['illness_max'] = 0
    data['incubation'] = 0
    data['incubation_day'] = 0
    data['incubation_max'] = 0

    return data, households, workplaces, schools



def generate_dict(data):
    # словарь с ключами - id домовладения, а значения - id жителей
    #TODO: так как эта колонка приводится к int, то ключ - int
    dict_hh_id = {i[0]: list(i[1]) 
            for i in data.groupby("sp_hh_id").sp_id}
    dict_hh_len = {i: len(dict_hh_id[i]) for i in dict_hh_id.keys()}

    # словарь с ключами - id работы, а значения - id работающих людей
    #TODO: так как эта колонка str, то ключ - str
    dict_work_id = {i[0]: list(i[1]) 
            for i in data[(data.age>=18)&(data.work_id!=0)].groupby("work_id").sp_id}
    dict_work_len = {i: len(dict_work_id[i]) for i in dict_work_id.keys()}

    # словарь с ключами - id школы, а значения - id учеников
    #TODO: так как эта колонка str, то ключ - str
    dict_school_id = {i[0]: list(i[1]) 
            for i in data[(data.age<18)&(data.work_id!=0)].groupby("work_id").sp_id}
    dict_school_len = {i: len(dict_school_id[i]) for i in dict_school_id.keys()}

    return dict_hh_id, dict_hh_len, dict_work_id, dict_work_len, dict_school_id, dict_school_len


    
