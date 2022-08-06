import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

L = np.diag([1, 2, 2, 1, 1, 2, 2, 1])
for i in range(7):
    if i != 3 and 4:
        L[i, i + 1] = -1
        L[i+1, i] = -1
    else:
        L[i, i] = 1

deta_t = 1e-3
sum_t = 10
t_list = np.arange(0, sum_t, deta_t)

c_mat = np.matrix([[0, 1, 0, 1, 1, 1, 1, 0]])
x_mat_list = [np.matrix([[18, 12, 8, 0, 15, 11, 5, 0],
                         [1] * 4 + [3] * 4])]
v_mat_list = [np.matrix([[4.8, 5.2, 4.9, 5.1, 6.0, 6.2, 5.8, 6.1],
                         [0] * 8])]
xL_mat_list = [np.matrix([[23, 20], [1, 3]])]
vL_mat_list = [np.matrix([[5.1, 5.8], [0, 0]])]
R_mat = np.matrix([[x * -5 for x in range(1, 5)] * 2, [1] * 4 + [3] * 4])
RL_mat = np.matrix([[0, 0], [1, 3]])
AL_mat = np.matrix([[1, 0]] * 4 + [[0, 1]] * 4).T
n1 = 0.05014


def get_2d(x):
    y = np.power((np.power(x[0, :], 2) + np.power(x[1, :], 2)), 1 / 2)
    return y


for _ in t_list:
    temp_mat1 = -np.dot((R_mat - x_mat_list[-1] - v_mat_list[-1]), L)
    temp_mat2 = -np.dot((x_mat_list[-1] + v_mat_list[-1]), L)
    h = get_2d(temp_mat1) - n1 * get_2d(temp_mat2)
    deta_v_mat = -np.dot(x_mat_list[-1] - R_mat + v_mat_list[-1], L) - x_mat_list[-1] + R_mat + \
                 np.dot(xL_mat_list[-1] - RL_mat + vL_mat_list[-1], AL_mat) - v_mat_list[-1]
    for i in range(8):
        if h[0, i] < 0:
            deta_v_mat[:, i] *= 0
    deta_vL_mat = np.matrix(np.zeros((2, 2)))
    deta_vL_mat[:, 1] = -(xL_mat_list[-1][:, 1] - xL_mat_list[-1][:, 0] - RL_mat[:, 1] + RL_mat[:, 0] +
                          2 * vL_mat_list[-1][:, 1] - 2 * vL_mat_list[-1][:, 0])
    x_mat_list.append(x_mat_list[-1] + deta_t * v_mat_list[-1])
    xL_mat_list.append(xL_mat_list[-1] + deta_t * vL_mat_list[-1])
    v_mat_list.append(v_mat_list[-1] + deta_t * deta_v_mat)
    vL_mat_list.append(vL_mat_list[-1] + deta_t * deta_vL_mat)
    if np.max(h) < 0.01:
        print('*' * 50)
        print(xL_mat_list[-1])
        break

R_mat = np.matrix([[-5, -15, -25, -30, -10, -20, -30, -35],
                   [1, 1, 1, 1, 3, 3, 3, 3]])
a = 1
b = 5
for _ in t_list:
    temp_mat1 = -np.dot((R_mat - x_mat_list[-1] - v_mat_list[-1]), L)
    temp_mat2 = -np.dot((x_mat_list[-1] + v_mat_list[-1]), L)
    h = get_2d(temp_mat1) - n1 * get_2d(temp_mat2)
    deta_v_mat = -np.dot(a * (x_mat_list[-1] - R_mat) + b * v_mat_list[-1], L) - a * x_mat_list[-1] + a * R_mat + \
                  np.dot(a * (xL_mat_list[-1] - RL_mat) + b * vL_mat_list[-1], AL_mat) - b * v_mat_list[-1]
    for i in range(8):
        if h[0, i] < 0:
            deta_v_mat[:, i] *= 0
    deta_vL_mat = np.matrix(np.zeros((2, 2)))
    deta_vL_mat[:, 1] = -(xL_mat_list[-1][:, 1] - xL_mat_list[-1][:, 0] - RL_mat[:, 1] + RL_mat[:, 0] +
                          10 * vL_mat_list[-1][:, 1] - 10 * vL_mat_list[-1][:, 0])
    x_mat_list.append(x_mat_list[-1] + deta_t * v_mat_list[-1])
    xL_mat_list.append(xL_mat_list[-1] + deta_t * vL_mat_list[-1])
    v_mat_list.append(v_mat_list[-1] + deta_t * deta_v_mat)
    vL_mat_list.append(vL_mat_list[-1] + deta_t * deta_vL_mat)
    if np.max(h) < 0.01:
        print('*' * 50)
        break

R_mat = np.matrix([[-5, -15, -25, -30, -10, -20, -30, -35],
                   [3, 3, 3, 1, 3, 1, 3, 1]])
L = np.zeros((8, 8))
n2 = 0.02733
n3 = 0.05915
AL_mat = np.matrix([[0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0, 1, 0]])
n = np.matrix([[n2 if i == 1 else n3 for i in np.array(AL_mat)[0, :]]])
a = 2
b = 5
for _ in t_list:
    temp_mat1 = -np.dot((R_mat - x_mat_list[-1] - v_mat_list[-1]), L)
    temp_mat2 = -np.dot((x_mat_list[-1] + v_mat_list[-1]), L)
    h = get_2d(temp_mat1) - np.multiply(n, get_2d(temp_mat2))
    deta_v_mat = -np.dot(a * (x_mat_list[-1] - R_mat) + b * v_mat_list[-1], L) - a * x_mat_list[-1] + a * R_mat + \
                 np.dot(a * (xL_mat_list[-1] - RL_mat) + b * vL_mat_list[-1], AL_mat) - b * v_mat_list[-1]
    for i in range(8):
        if h[0, i] < 0:
            deta_v_mat[:, i] *= 0
    deta_vL_mat = np.matrix(np.zeros((2, 2)))
    deta_vL_mat[:, 1] = -(xL_mat_list[-1][:, 1] - xL_mat_list[-1][:, 0] - RL_mat[:, 1] + RL_mat[:, 0] +
                          2 * vL_mat_list[-1][:, 1] - 2 * vL_mat_list[-1][:, 0])
    x_mat_list.append(x_mat_list[-1] + deta_t * v_mat_list[-1])
    xL_mat_list.append(xL_mat_list[-1] + deta_t * vL_mat_list[-1])
    v_mat_list.append(v_mat_list[-1] + deta_t * deta_v_mat)
    vL_mat_list.append(vL_mat_list[-1] + deta_t * deta_vL_mat)
    if abs(x_mat_list[-1][1, 0] - 3) < 0.01 and v_mat_list[-1][1, 0] < 0.01:
        print("*"*50)
        break

L = np.matrix([[1, 0, 0, 0, -1, 0, 0, 0], [0, 2, -1, 0, -1, 0, 0, 0], [0, -1, 2, 0, 0, 0, -1,0],
               [0, 0, 0, 2, 0, -1, 0, -1], [-1, -1, 0, 0, 2, 0, 0, 0], [0, 0, 0, -1, 0, 1, 0, 0],
               [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, -1, 0, 0, 0, 1]])
R_mat = np.matrix([[-5, -15, -20, -10, -10, -5, -25, -15],
                   [3, 3, 3, 1, 3, 1, 3, 1]])
for _ in list(t_list):
    temp_mat1 = -np.dot((R_mat - x_mat_list[-1] - v_mat_list[-1]), L)
    temp_mat2 = -np.dot((x_mat_list[-1] + v_mat_list[-1]), L)
    h = get_2d(temp_mat1) - np.multiply(n, get_2d(temp_mat2))
    deta_v_mat = -np.dot(x_mat_list[-1] - R_mat + v_mat_list[-1], L) - x_mat_list[-1] + R_mat + \
                 np.dot(xL_mat_list[-1] - RL_mat + vL_mat_list[-1], AL_mat) - v_mat_list[-1]
    for i in range(8):
        if h[0, i] < 0 and False:
            deta_v_mat[:, i] *= 0
    deta_vL_mat = np.matrix(np.zeros((2, 2)))
    deta_vL_mat[:, 1] = -(xL_mat_list[-1][:, 1] - xL_mat_list[-1][:, 0] - RL_mat[:, 1] + RL_mat[:, 0] +
                          2 * vL_mat_list[-1][:, 1] - 2 * vL_mat_list[-1][:, 0])
    x_mat_list.append(x_mat_list[-1] + deta_t * v_mat_list[-1])
    xL_mat_list.append(xL_mat_list[-1] + deta_t * vL_mat_list[-1])
    v_mat_list.append(v_mat_list[-1] + deta_t * deta_v_mat)
    vL_mat_list.append(vL_mat_list[-1] + deta_t * deta_vL_mat)
    if np.max(h) < 0.001:
        print('*'*50)
        break

c_list = ["red", "blue", "orange", "green", "olive", "gold", "purple", "darkred"]
for i, c in zip(range(8), c_list):
    plt.plot([x[0, i] for x in x_mat_list], c=c)
plt.show()

for i, c in zip(range(8), c_list):
    plt.plot([x[1, i] for x in x_mat_list], c=c)
for i, c in zip([0, 1], ["black", "brown"]):
    plt.plot([x[1, i] for x in xL_mat_list], c=c)
plt.show()

def draw(i):
    plt.cla()
    for j, c in zip(range(8), c_list):
        plt.plot([x[0, j] for x in x_mat_list[:i*50]], [x[1, j] for x in x_mat_list[:i*50]], c=c)
        plt.annotate('', xy=(x_mat_list[i * 50][:, j] + v_mat_list[i * 50][:, j] * 0.2), xytext=(x_mat_list[i * 50][:, j]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=c))
    for j, c in zip(range(2), ["black", "brown"]):
        plt.plot([x[0, j] for x in xL_mat_list[:i * 50]], [x[1, j] for x in xL_mat_list[:i * 50]], c=c)
        plt.annotate('', xy=(xL_mat_list[i * 50][:, j] + vL_mat_list[i * 50][:, j] * 0.2),
                     xytext=(xL_mat_list[i * 50][:, j]), arrowprops=dict(connectionstyle="arc3", facecolor=c))
    plt.plot([0, 300], [2, 2], c="black")
    plt.xlim(0, 250)


print(x_mat_list[-1])
fig = plt.figure()
an = ani.FuncAnimation(fig, draw, int(len(x_mat_list)/50), interval=1)
an.save("w3.gif")
plt.show()
