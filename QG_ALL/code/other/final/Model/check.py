import numpy as np
import pandas as pd
import os
import re

import scipy.io


class CheckData:
    def __init__(self, file, rc):
        self.file = file
        self.rc = rc
        self.x_mat = None
        self.type = None

    def check(self):
        try:
            self.rc = float(self.rc)
        except:
            return 3
        return None

    def get_type(self):
        self.type = re.findall(r"\.(.*)", self.file)

    def load_data(self):
        self.get_type()
        if self.type == "npy":
            try:
                self.x_mat = np.load(self.file)
            except:
                return 2
        elif self.type == 'txt':
            try:
                self.x_mat = np.loadtxt(self.file)
            except:
                return 2
        elif self.type == 'mat':
            try:
                x = scipy.io.loadmat(self.file)
                self.x_mat = np.array([i for i in x if type(i) == np.ndarray][0])
            except:
                return 2
        elif self.type == "csv":
            try:
                self.x_mat = np.array(pd.read_csv(self.file))
            except:
                return 2
        elif self.type == 'xls' or self.type == 'xlsx':
            try:
                self.x_mat = np.array(pd.read_excel(self.file))
            except:
                return 2
        else:
            return 1
        return None

    def check_data(self):
        try:
            self.x_mat = np.array(self.x_mat, dtype=float)
        except:
            return 2
        if np.any(np.isnan(self.x_mat)):
            return 2
        return None

    def main(self):
        statu = self.check()
        if statu is None:
            return statu
        statu = self.load_data()
        if statu is not None:
            return statu
        statu = self.check_data()
        if statu is not None:
            return statu
        return self.x_mat
