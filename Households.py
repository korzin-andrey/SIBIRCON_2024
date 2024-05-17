from Place import Place



class Households(Place):

    def __init__(self, x_size, lmbd, location, place_id, place_len, vfunc):
        super(x_size, lmbd, location, place_id, place_len, vfunc)
        self.x_len=10000

    def real_inf(self, susceptible):

        x_rand = np.random.rand(x_len)
        real_inf_place = np.array([])
        for i in self.place_inf:
            if i not in self.dict_place_id.keys():
                dict_place_id.update(
                    {i: list(susceptible[(susceptible.sp_hh_id == i) & (susceptible.susceptible == 1)].sp_id)}
                        )

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
            if i in self.dict_place_id.keys():
                self.dict_place_id[place_id].remove(person_id)




