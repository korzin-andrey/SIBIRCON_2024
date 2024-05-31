from infectiousness import vectorized
import numpy as np

class Place:
    '''
    Интерфейс для всех локаций    

    Args:
        lmbd             (int):     вирулентность
        location         (pd.DataFrame):    содержит	sp_id	 latitude	 longitude	  size
        place_id    (dict):    словарь с ключами - id места, а значения - id тех, кто там находится          
        place_len   (dict):    словарь с ключами - id места, а значения - общее количество людей
    '''


    def __init__(self, lmbd, location, place_id, place_len):

        self.lmbd = lmbd
        self.location = location
        self.dict_place_id = place_id
        self.dict_place_len = place_len
        self.vfunc = vectorized() # векторизованная функция вычисления заразности человека
        self.place_inf=None
        
        real_inf_place = None

    def prob(self, temp):
        return np.repeat(temp, 3) * self.lmbd

    def set_place_inf(self, place_inf):
        self.place_inf=place_inf
        return True
        
    def infection(self, x_rand):

        real_inf_place = np.array([])
        for i in self.place_inf:
           
            # текущее количество восприимчивых
            place_len = len(self.dict_place_id[i])

            if place_len != 0:
                # вычисление заразности каждого заболевшего
                temp = self.vfunc(self.place_inf[i])

                # вероятность заражения подверженных
                prob = self.prob(temp)
                contact_length = len(prob)

                # вероятность не заразиться
                place_rand = x_rand[:contact_length]
                x_rand = x_rand[contact_length:]

                # количество реально заразившихся людей
                real_inf = len(place_rand[place_rand < prob])
                real_inf = place_len if place_len < real_inf else real_inf


                real_inf_id = np.random.choice(np.array(self.dict_place_id[i]), real_inf, replace=False)
                real_inf_place = np.concatenate((real_inf_place, real_inf_id))
        return real_inf_place

    def clean_place(self, iterator):
        for place_id, person_id in iterator:
        
            # тест на отсутствие повторов
            #assert len(self.dict_place_id[place_id]) == len(set(self.dict_place_id[place_id]))   
            
            self.dict_place_id[place_id].remove(person_id)
        return True



