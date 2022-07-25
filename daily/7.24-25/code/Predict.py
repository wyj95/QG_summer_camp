import random

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
import sqlalchemy
from flask import Flask
import datetime
import time

class PredictPower:
    def __init__(self, mysql_acc):
        self.mysql_acc = mysql_acc
        self.df = pd.DataFrame([])
        self.power_list = []
        self.power_list = []
        self.fit = None

    def get_data(self, date_list, request, new_flag=False):
        df = pd.DataFrame([])
        if new_flag:
            self.df = pd.DataFrame([])
        for date in date_list:
            engine = sqlalchemy.create_engine(self.mysql_acc)
            sql = "select * from device" + date + "where" if request != "" else "" + request
            temp_df = pd.read_sql(sql, engine)
            self.df if new_flag else df = self.df.append(temp_df)
        return self.df if new_flag else df

    def clean_data(self):
        pass

    def create_power_list(self, data_list, request, now_time):
        self.power_list = []
        for date in data_list:
            temp_mat = np.array(PredictPower.get_data(self, [date], request))
            one_day_time_list = [(x * np.shape(temp_mat)[0]) for x in range(5 if date != data_list[-1]
                                                                            else int(now_time[0] / 6 + 1))]
            self.power_list += [np.sum(temp_mat[i:j, -1])
                                for i, j in zip(one_day_time_list[:-1], one_day_time_list[1:])]
        return self.power_list

    def predict(self):
        self.fit = ExponentialSmoothing(self.power_list, seasonal_periods=4, trend='add', seasonal='add', ).fit()
        return self.fit.forecast(2)[-1]

    def draw(self):
        if self.fit is None:
            return
        data_num = len(self.power_list)
        plt.plot(range(data_num), self.power_list)
        plt.plot([data_num+x for x in range(3)], [self.power_list[-1]] + [self.fit.forecast(2)])
        plt.show()


app = Flask(__name__)

@app.route('/')
def return_predict():
    global predict_power_class
    today = datetime.date.today()
    last_three_day_list = [str(today - datetime.timedelta(days=x)) for x in range(0, 4)][::-1]
    request = ""
    predict_power_class.create_power_list(last_three_day_list, request, time.localtime())
    return predict_power_class.predict()


if __name__ == "__main__":
    mysql_acc = ""
    predict_power_class = PredictPower(mysql_acc)
    app.run()
