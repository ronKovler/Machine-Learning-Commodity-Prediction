#!/usr/bin/env python
##############################################################
#
#   Keras training
#
#   Alexey Goder
#   agoder@yahoo.com
#
#   august 29th, 2019
#
##############################################################

import math
import pickle
from datetime import timedelta
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from prepare_input import get_spot_prices, get_data, prepare_input, advance_to_next_workday, get_monday_after_last_friday
from sklearn.metrics import mean_squared_error

def build_regressor1():
    regressor = Sequential()
    regressor.add(Dense(units=100, input_dim=100))
    regressor.add(Dense(units=100, input_dim=100,kernel_initializer='normal', activation='relu'))
    regressor.add(Dense(units=10,  activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor


def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=100, input_dim=100))
    regressor.add(Dense(units=100, input_dim=100,kernel_initializer='normal', activation='relu'))
    regressor.add(Dense(units=30,  activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor


def get_data_set(commodity, train=True, weeks=30):
    spot_price = get_spot_prices('C:/Users/agode/Documents/AGBlox/cattle_spot_prices.csv')
    data_keys, data_vec, counter = get_data(commodity, train)
    return prepare_input(data_keys, data_vec, spot_price, weeks=weeks, counter=counter)


def get_one_prediction(commodity, forecast_date, base_price, forecast, sigma):
    return {
            'commodity_name': commodity,
            'forecast_date': forecast_date.strftime("%m/%d/%Y"),
            'forecast_price': "%5.2f"%((forecast/100.0 + 1.0)*base_price),
            'sigma': "%5.2f"%((sigma/100.0)*base_price)
           }


if __name__ == "__main__":

    COMMODITY = 'cattle'

    x_keys, x_input, y_output, counter = get_data_set(COMMODITY, weeks=30)
    test_keys, test_input, test_output, counter = get_data_set(COMMODITY, train=False, weeks=30)

    regressor = KerasRegressor(build_fn=build_regressor, batch_size=None, epochs=10, steps_per_epoch=10)
    results=regressor.fit(x_input, y_output)
    y_pred= regressor.predict(test_input, steps=None)

    for i in range(len(test_output[0])):
        sigma = math.sqrt(mean_squared_error(test_input[:, i].tolist(), y_pred[:, i].tolist()))
        print('Week=%d' % (i+1), sigma)

    keras_model={
                 'regressor': regressor,
                 'commodity': COMMODITY,
                 'nlp_parameters': {'idf': True, 'ngram': (1, 2), 'max_f=100': 100},
                 'sigma': [math.sqrt(mean_squared_error(test_input[:, i].tolist(), y_pred[:, i].tolist()))
                           for i in range(len(test_output[0]))
                           ],
                 "counter": counter,
                 'weeks': 30
                 }
    with open('keras_model_' + COMMODITY + '.pkl', 'wb') as f:
        pickle.dump(keras_model, f)

    forecast_list = y_pred[3,].tolist()
    print(len(forecast_list))
    start_date = get_monday_after_last_friday()
    for i in range(30):
        print(get_one_prediction('feeder cattle', start_date+timedelta(days=(1+i)*7), 140.00, forecast_list[i], keras_model['sigma'][i]))
