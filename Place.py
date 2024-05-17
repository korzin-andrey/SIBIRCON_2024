from infectiousness import vectorized


class Place:
    '''
    Интерфейс для всех локаций    

    Args:
        lmbd             (int):     вирулентность
        location         (pd.DataFrame):    содержит	sp_id	 latitude	 longitude	  size
        dict_place_id    (dict):    словарь с ключами - id места, а значения - id тех, кто там находится          
        dict_place_len   (dict):    словарь с ключами - id места, а значения - общее количество людей
        vfunc            (np.vectorize):    векторизованная функция вычисления заразности человека
    '''


    def __init__(self, lmbd, location, place_id, place_len):

        self.lmbd = lmbd
        self.location = location
        self.dict_place_id = place_id
        self.dict_place_len = place_len
        self.vfunc = vectorized()
        self.place_inf=None
        self.x_len=1000

    def prob(self, temp):
        return np.repeat(temp, 3) * self.lmbd

    def place_inf(self, place_inf):
        self.place_inf=place_inf

    def real_inf(self):
        x_rand = np.random.rand(x_len)
        real_inf_place = np.array([])
        for i in self.place_inf:
            
            # текущее количество восприимчивых
            place_len = len(dict_place_id[i])

            if place_len != 0:
                # вычисление заразности каждого заболевшего
                temp = self.vfunc_b_r(self.place_inf[i])

                # вероятность заражения подверженных
                prob = self.prob(temp)
                contact_length = len(prob)

                # вероятность не заразиться
                place_rand = x_rand[:contact_length]
                x_rand = x_rand[contact_length:]

                # количество реально заразившихся людей
                real_inf = len(place_rand[place_rand < prob])

                if place_len < real_inf:
                    real_inf = place_len

                real_inf_id = np.random.choice(np.array(dict_place_id[i]), real_inf, replace=False)
                real_inf_place = np.concatenate((real_inf_place, real_inf_id))

        return real_inf_place

    def clean_place(self, iterator):
        for place_id, person_id in iterator:
            self.dict_place_id[place_id].remove(person_id)



