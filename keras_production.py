#!/usr/bin/env python
##############################################################
#
#   Commodity price forecast
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   august 29th, 2019
#
##############################################################

import pickle
from keras_train import build_regressor, get_one_prediction
from prepare_input import get_last_friday
from datetime import timedelta
from get_input_data import get_input_vector, get_last_commodity_price

COMMODITY = 'cattle'
with open('keras_model_' + COMMODITY + '.pkl', 'rb') as f:
    keras_model = pickle.load(f)


def get_long_term_forecast(keras_model=keras_model):
    input_vector = get_input_vector(counter=keras_model['counter'])
    forecast_list = keras_model['regressor'].predict(input_vector, steps=None)[:, ].tolist()
    start_date = get_last_friday() + timedelta(days=7)
    return [get_one_prediction('feeder cattle',
                               start_date + timedelta(days=i * 7),
                               get_last_commodity_price(),
                               forecast_list[i],
                               keras_model['sigma'][i]
                               )
            for i in range(keras_model['weeks'])
            ]


if __name__ == "__main__":
    print(get_long_term_forecast())
