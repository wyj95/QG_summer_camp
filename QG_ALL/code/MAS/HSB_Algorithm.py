import copy
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import math
import random
import pandas as pd

class HSB:
    def __init__(self, rc=0.5, DP_flag=False):
        self.DP_flag = DP_flag
        self.rc = rc
        self.engine = create_engine("mysql+pymysql://root:3751ueoxjwgixjw3913@39.98.41.126:3306/qg_final")
        self.location_address = 'file_location'
        self.adjacency_address = 'file_adjacency'
        self.file = ''
        self.x_mat = None
        self.x_mat_list = None
        self.A_mat = None
        self.A_mat_list = None
        self.P_list = None
        self.status_list = None
        self.CWP_mat = None
        self.A_mat1 = None

    def create_A_mat(self):
        """
        create the adjacency matrix using the location matrix(x_mat) of self
        :return: the adjacency matrix --> np.array([[]])
        """
        n, d = np.shape(self.x_mat)
        D_mat = np.array([[self.get_d(self.x_mat[i, :], self.x_mat[j, :]) if j > i else 0  for j in range(n)]
                          for i in range(n)])
        D_mat += D_mat.T
        self.A_mat = np.array([[1 if D_mat[i, j] <= self.rc*1.0000 else 0 for j in range(n)] for i in range(n)])
        for i in range(n):
            self.A_mat[i, i] = 0
        return self.A_mat

    def get_d(self, x, y):
        """
        get the adjacency matrix between x and y
        :param x: vector1 x np.array([])
        :param y: vector1 y np.array([])
        :return: the adjacency matrix between x and y
        """
        return np.linalg.norm(x - y)

    def check_in_hat(self, i, vec):
        """
        check whether the vec in the CS_hat of i_th agent
        :param i: the i_th agent --> int
        :param vec: the vector --> np.array([])
        :return: True or False --> bool
        """
        return self.get_d(vec, self.x_mat[i, :]) < self.rc and all(self.get_d(vec, (self.x_mat[l, :]+self.x_mat[i, :])
                                                                              /2) < self.rc/2
                                                                   for l in (set(np.nonzero(self.A_mat1[i, :])[0]) &
                                                                             set(np.nonzero(self.A_mat[i, :])[0])))

    def check_in(self, i, vec):
        """
        check whether the vec in the CS of i_th agent, which can help to get the area of the CS
        :param i: the i_th agent --> int
        :param vec: the vector --> np.array([])
        :return: True or False --> bool
        """
        return self.get_d(vec, self.x_mat[i, :]) < self.rc and all(self.get_d(vec, self.x_mat[l, :]) < self.rc / 2
                                                                   for l in np.nonzero(self.A_mat[i, :])[0])

    def create_P_list(self, seed=123, run_num=5000):
        """
        create the priorities of every agent
        :param seed: the random seed --> int
        :param run_num: the running number in the Monte Carlo algorithm to get the area of every CS --> int
        :return: the priorities  --> list
        """
        n, d = self.x_mat.shape
        np.random.seed(seed)
        self.P_list = []
        for i in range(n):
            num = 0
            for k in range(run_num):
                x = 20 * self.rc * np.random.random(d) + self.x_mat[i, :] - 20*self.rc
                if self.check_in(i, x):
                    num += 1
            self.P_list.append(self.Hash(num*3.14/4/run_num, i))
        return self.P_list

    def Hash(self, s1, i, keep_num=5):
        """
        use the B1 and B2 to get the priority
        :param s1: B1 in the paper --> float
        :param i: the i_th agent --> int
        :param keep_num: the B1's power --> int
        :return: the priority of i_th agent --> float
        """
        random.seed(i)
        return round(s1, keep_num) + random.random() / pow(10, keep_num)

    def prune(self):
        """
        pruning
        :return: the adjacency matrix after pruning --> np.array([[]])
        """
        n = self.x_mat.shape[0]
        self.status_list = np.ones(n) * 3
        ch = np.zeros(n, dtype="int")

        # CH
        for i in range(n):
            ch[i] = self.P_list.index(max([self.P_list[j] for j in list(np.nonzero(self.A_mat[i, :])[0]) + [i]]))
            self.status_list[ch[i]] = 0
        work_for_list = [[] for _ in range(n)]

        no_ch = set(np.nonzero(self.status_list)[0])

        for i in (set(range(n)) - no_ch):
            work_for_list[i] = list(set(np.nonzero(self.A_mat[i, :])[0]) - no_ch)

        # DW
        for i in range(n):
            if self.status_list[i] == 0:
                continue
            i_nei = set(np.nonzero(self.A_mat[i, :])[0])
            for c1 in i_nei - no_ch:
                for p in (i_nei & no_ch):
                    for c2 in (set(np.nonzero(self.A_mat[p, :])[0]) - no_ch - i_nei -
                               set(np.nonzero(self.A_mat[c1, :])[0])):
                        run_set = i_nei - {c1, p, c2}
                        flag = True
                        for m in run_set:
                            if self.A_mat[c1, m] * self.A_mat[c2, m] > 0:
                                flag = False
                                break

                            if (any(self.status_list[n0] == 0 for n0 in (set(np.nonzero(self.A_mat[m, :])[0]) &
                                                                         i_nei - {c1, i, p, m, c2})) or
                                    (self.status_list[m] == 0) and self.A_mat[c2, m] > 0):
                                flag = False
                                break

                            if self.A_mat[m, c2] > 0 and (self.P_list[m] > self.P_list[i] or
                                                          any(self.P_list[o] > self.P_list[i]
                                                              for o in (set(np.nonzero(self.A_mat[c1, :])[0]) &
                                                                        set(np.nonzero(self.A_mat[m, :])[0]) -
                                                                        {i, p, c2}))):
                                flag = False
                                break

                        if flag:
                            self.status_list[i] = 1
                            work_for_list[i] += [c1, c2]

        # GW
        temp_status_list = np.zeros(n)
        for i in range(n):
            if self.status_list[i] == 0:
                continue
            i_nei = set(np.nonzero(self.A_mat[i, :])[0])
            for c1 in i_nei:
                if self.status_list[c1] == 0 or self.status_list[c1] == 1:
                    for c2 in i_nei - {c1}:
                        if (self.status_list[c2] == 0 or self.status_list[c2] == 1) and \
                                self.status_list[c1] * self.status_list[c2] == 0:
                            flag = True
                            run_set = set(np.nonzero(self.A_mat[c1, :])[0])&set(np.nonzero(self.A_mat[c2, :])[0]) - {i}
                            for m in run_set:
                                if self.status_list[m] == 0 or self.status_list[m] == 1:
                                    flag = False
                                    break
                                if self.P_list[m] > self.P_list[i]:
                                    flag = False
                                    break
                            if flag:
                                work_for_list[i] += [c1, c2]
                                temp_status_list[i] = 2

        self.status_list = np.array([self.status_list[i] if temp_status_list[i] == 0 else 2 for i in range(n)])
        work_for_list = [set(x) for x in work_for_list]

        temp_work_for_list = copy.deepcopy(work_for_list)
        for i in (set(range(n)) - no_ch):
            for j in work_for_list[i]:
                for k in set(work_for_list[i]) & set(work_for_list[j]):
                    temp_p_list = {self.P_list[x] for x in [i, j, k]}
                    min1 = min(temp_p_list)
                    min2 = min(temp_p_list - {min1})
                    min1 = self.P_list.index(min1)
                    min2 = self.P_list.index(min2)
                    temp_work_for_list[min1] -= {min2}
                    temp_work_for_list[min2] -= {min1}
        work_for_list = [list(x) for x in temp_work_for_list]

        temp_work_for_list = [set(x) for x in work_for_list]
        work_for_list = temp_work_for_list

        # DW
        for i in range(n):
            if self.status_list[i] == 1:
                for j in (work_for_list[i] - no_ch):
                    for k in (work_for_list[i] & work_for_list[j] - no_ch):
                        del_i = self.P_list.index(min(self.P_list[j], self.P_list[k]))
                        temp_work_for_list[i] -= {del_i}

        # GW
        for i in range(n):
            if self.status_list[i] == 2:
                clu_list = []
                for j in (temp_work_for_list[i] - no_ch):
                    temp_list = []
                    for k in range(len(clu_list)):
                        if any(j in work_for_list[l] for l in clu_list[k]):
                            temp_list.append(k)
                    if len(temp_list) == 0:
                        clu_list.append([j])
                    elif len(temp_list) == 1:
                        clu_list[temp_list[0]].append(j)
                    else:
                        for l in temp_list[1:]:
                                clu_list[temp_list[0]] += clu_list[l] + [j]
                                clu_list[l] = []
                        clu_list = [x for x in clu_list if len(x) != 0]
                for _list in clu_list:
                    P_clu_list = [self.P_list[j] for j in _list]
                    temp_work_for_list[i] -= set(_list) - {self.P_list.index(max(P_clu_list))}

        work_for_list = [list(x) for x in temp_work_for_list]

        temp_A_mat = np.zeros((n, n), dtype="int")
        for i in range(n):
            if self.status_list[i] == 3:
                temp_A_mat[i, ch[i]] = 1
                temp_A_mat[ch[i], i] = 1
            else:
                temp_A_mat[i, work_for_list[i]] = 1
                temp_A_mat[work_for_list[i], i] = 1

        if (temp_A_mat.T - temp_A_mat > 0).any():
            print("error")
        wrong = np.sum(temp_A_mat, axis=0)
        print(self.status_list[np.where(wrong == 0)])
        if len(self.status_list[np.where(wrong == 0)]) != 0:
            for i in range(n):
                if np.sum(temp_A_mat[i, :]) == 0:
                    temp_A_mat[i, ch[i]] = 1
                    temp_A_mat[ch[i], i] = 1
                    print(str(i) + " " + str(self.status_list[i]) + " csb")
        self.A_mat1 = temp_A_mat
        return self.A_mat1

    def create_CWP_mat(self):
        """
        create the CWP matrix which is where agent expects to go
        :return: the CWP matrix --> np.array([[]])
        """
        n, d = np.shape(self.x_mat)
        self.CWP_mat = np.array([(np.max(self.x_mat[list(set(np.nonzero(self.A_mat1[i, :])[0])) +
                                                         [i], :], axis=0) +
                                  np.min(self.x_mat[list(set(np.nonzero(self.A_mat1[i, :])[0])) +
                                                         [i], :], axis=0)) / 2
                                 for i in range(n)])
        return self.CWP_mat

    def move(self, step_num=100):
        """
        move the agent and keep them in the CS_hat
        :param step_num: the length of the step --> int
        :return: the location matrix after moving --> np.array([[]])
        """
        n, d = self.x_mat.shape
        step = self.rc / step_num
        new_x_mat = np.zeros((n, d))
        lamda = None
        l_sum = 0
        for i in range(n):
            for lamda in np.arange(1, 0-step, -step):
                if self.check_in_hat(i, lamda*self.CWP_mat[i, :] + (1-lamda)*self.x_mat[i, :]):
                    break
            l_sum += lamda
            new_x_mat[i, :] = lamda*self.CWP_mat[i, :] + (1-lamda)*self.x_mat[i, :]
        self.x_mat_list.append(self.x_mat.copy())
        self.x_mat = new_x_mat
        print(l_sum)
        return new_x_mat

    def draw_temp(self, save_time=None, show_flag=False, save_file="", prune_flag=False):
        """
        draw the picture in order to debug while testing
        :param save_time: the run time now --> int
        :param show_flag: plt.show() or not --> bool
        :param save_file: where the file is saved
        :param prune_flag: whether user the A matrix with pruning or not
        :return: None
        """
        plt.cla()
        c_list = ["red", "green", "blue", "black"]
        n = np.shape(self.A_mat)[0]
        for i in range(n):
            plt.scatter(self.x_mat[i, 0], self.x_mat[i, 1], c=c_list[int(self.status_list[i])]
                        if self.status_list is not None else "black")
        for i in range(n):
            for j in range(n):
                if (self.A_mat[i, j] > 0) if not prune_flag else (self.A_mat1[i, j] > 0):
                    plt.plot(self.x_mat[[i, j], 0], self.x_mat[[i, j], 1], c="red")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        if save_time is not None:
            plt.savefig(save_file + "pic" + str(save_time) + ".jpg")
            "1_1000/pic1.jpg"
        if show_flag:
            plt.show()

    def save(self):
        """
        save the x_mat_list and A_mat_list to mysql
        :return: None
        """
        id_mat = np.array([list(range(self.x_mat.shape[0]))]).T
        mat = np.append(self.x_mat_list[0], id_mat, axis=1)
        for i in range(1, len(self.x_mat_list)):
            mat = np.append(mat, np.array([[None, None, id_mat[-1, 0] + 1]]), axis=0)
            id_mat += np.shape(self.x_mat)[0] + 1
            mat = np.append(mat, np.append(self.x_mat_list[i], id_mat, axis=1), axis=0)
        df = pd.DataFrame(mat, columns=["x", 'y', 'id'])
        df.to_sql(self.location_address, con=self.engine, index=False, if_exists="replace")

        mat = np.array([[None, None]])
        id_mat = np.array([list(range(self.x_mat.shape[0]))]).T
        for a in self.A_mat_list:
            temp_mat = np.array([['' for _ in range(self.x_mat.shape[0])]], dtype="<U5000").T
            no_zero = np.nonzero(a)
            for i, j in zip(no_zero[0], no_zero[1]):
                temp_mat[i, 0] += ", " + str(j)
            temp_mat[:, 0] = np.array([x[2:] if x is not None else x for x in temp_mat[:, 0]])
            temp_mat = np.append(temp_mat, id_mat, axis=1)
            mat = np.append(mat, temp_mat, axis=0)
            mat = np.append(mat, np.array([[None, id_mat[-1, 0] + 1]]), axis=0)
            id_mat += self.x_mat.shape[0] + 1
        mat = mat[1:-1, :]
        mat[:, 1] = np.array(mat[:, 1], dtype=int)
        df = pd.DataFrame(mat, columns=["ad_index", "id"])
        df.to_sql(self.adjacency_address, con=self.engine, index=False, if_exists="replace")

    def main(self, x=None):
        """
        the main function to run the data
        :param x: the matrix if not we will read from the file which may not be found
        :return: None
        """
        self.x_mat = np.load(self.file) if x is not None else x
        self.x_mat_list = [self.x_mat]
        self.create_A_mat()
        self.A_mat_list = [self.A_mat]
        loss_list = []
        n, d = self.x_mat.shape
        run_time = 1
        while any(self.get_d(self.x_mat[i, :], self.x_mat[j, :]) > 0.01 for j in range(n) for i in range(n)):
            self.create_A_mat()
            self.create_P_list()
            self.prune()
            self.create_CWP_mat()
            self.move()
            self.x_mat_list.append(self.x_mat.copy())
            self.A_mat_list.append(self.A_mat1.copy())
            loss = round(sum(self.get_d(self.x_mat[i, :], self.x_mat[j, :]) for j in range(n) for i in range(n)), 3)
            loss_list.append(loss)
            run_time += 1
            if run_time > 500:
                break
        self.save()
