import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.animation as ani

deta_t = 1e-3
t_num = int(10 / deta_t)
"""
a_mat = np.matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]).T
x_list = [np.matrix([[40, 40, 30, 30], [1, 2, 1, 2]])]
v_list = [np.matrix([[10, 10, 10, 10], [0, 0, 0, 0]])]
r_mat = np.matrix([[-10, -20, -30, -40], [0, 0, 0, 0]])
xPL_list = [np.matrix([[60], [1]])]
vPL_list = [np.matrix([[10], [0]])]
rPL_mat = np.matrix([[0], [0]])
xRL_list = [np.matrix([[10], [1]])]
vRL_list = [np.matrix([[10], [0]])]
rRL_mat = np.matrix([[-50], [0]])

for _ in range(t_num):
    deta_v_mat = np.zeros((2, 4))
    for i in range(4):
        deta_v_mat[0, i] = xPL_list[-1][0] - 2 * x_list[-1][0, i] + vPL_list[-1][0] -\
                           2 * v_list[-1][0, i] + xRL_list[-1][0] + vRL_list[-1][0] + \
                           2 * r_mat[0, i] - rPL_mat[0] - rRL_mat[0]
        for j in range(4):
            deta_v_mat[0, i] -= a_mat[i, j] * (x_list[-1][0, i] - x_list[-1][0, j] -r_mat[0, i] +
                                               r_mat[0, j] + v_list[-1][0, i] - v_list[-1][0, j])
        deta_v_mat[1, i] = xPL_list[-1][1] + r_mat[1, i] - rPL_mat[1] - x_list[-1][1, i] -\
                           v_list[-1][1, i] + vPL_list[-1][1]
        deta_v_mat[1, i] *= 5
    x_list.append(x_list[-1] + deta_t * v_list[-1])
    v_list.append(v_list[-1] + deta_t * deta_v_mat)
    xPL_list.append(xPL_list[-1] + deta_t * vPL_list[-1])
    vPL_list.append(vPL_list[-1])
    xRL_list.append(xRL_list[-1] + deta_t * vRL_list[-1])
    vRL_list.append(vRL_list[-1])

c_list = ["green", "red", "orange", "blue"]

for i, c in zip(range(4), c_list):
    plt.plot([x[0, i] for x in x_list], [x[1, i] for x in x_list], c=c)
    plt.annotate('', xy=(x_list[-1][:, i] + v_list[-1][:, i] * 0.2), xytext=(x_list[-1][:, i]),
                 arrowprops=dict(connectionstyle="arc3", facecolor=c))
plt.plot([x[0, 0] for x in xPL_list], [ x[1, 0] for x in xPL_list], c="black")
plt.annotate('', xy=(xPL_list[-1] + vPL_list[-1]*0.2), xytext=(xPL_list[-1]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="black"))
plt.plot([x[0, 0] for x in xRL_list], [x[1, 0] for x in xRL_list], c="purple")
plt.annotate('', xy=(xRL_list[-1] + vRL_list[-1] * 0.2), xytext=(xRL_list[-1]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="purple"))
plt.plot([0, 100], [1, 1])
plt.xlabel("X Position(m)")
plt.ylabel("Y Position(m)")
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)
plt.savefig("w2.1.jpg")
plt.show()

plt.figure(figsize=(12, 15))
plt.figure(1)
plt.subplot(3, 2, 1)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, 10 + deta_t, deta_t), [x[0, i] for x in x_list], c=c)
plt.plot(np.arange(0, 10+deta_t, deta_t), [x[0, 0] for x in xPL_list], c="black")
plt.plot(np.arange(0, 10+deta_t, deta_t), [x[0, 0] for x in xRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("X Position(m)")
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)
plt.xlim(0, t)

plt.figure(1)
plt.subplot(3, 2, 2)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, t+deta_t, deta_t), [x[1, i] for x in x_list], c=c)
plt.plot(np.arange(0, t+deta_t, deta_t), [x[1, 0] for x in xPL_list], c="black")
plt.plot(np.arange(0, t+deta_t, deta_t), [x[1, 0] for x in xRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("Y Position(m)")
plt.xlim(0, t)
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)

plt.figure(1)
plt.subplot(3, 2, 3)
for i in range(4):
    plt.plot(np.arange(0, t+deta_t, deta_t), [x[0, i] - y[0, 0] for x, y in zip(x_list, xPL_list)])
for i in range(4):
    plt.plot(np.arange(0, t+deta_t, deta_t), [x[0, i] - y[0, 0] for x, y in zip(x_list, xRL_list)])
str_list = ["L" + str(i) + " - L" + j for i, j in zip(list(range(4)) * 2, ["1"] * 4 + ["2"] * 4)]
plt.legend(str_list, loc=1)
plt.plot([0, t], [0, 0])
plt.xlabel("time(s)")
plt.ylabel("deta_x")
plt.xlim(0, t)

plt.figure(1)
plt.subplot(3, 2, 4)
for i in range(4):
    plt.plot(np.arange(0, t+deta_t, deta_t), [x[1, i] - y[1, 0] for x, y in zip(x_list, xPL_list)])
for i in range(4):
    plt.plot(np.arange(0, t+deta_t, deta_t), [x[1, i] - y[1, 0] for x, y in zip(x_list, xRL_list)])
str_list = ["L" + str(i+1) + " - L" + j for i, j in zip(list(range(4)) * 2, ["1"] * 4 + ["2"] * 4)]
plt.legend(str_list, loc=1)
plt.plot([0, t], [0, 0], c="black")
plt.xlabel("time(s)")
plt.ylabel("deta_y(m)")
plt.xlim(0, t)

plt.figure(1)
plt.subplot(3, 2, 5)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [v[0, i] for v in v_list], c=c)
plt.plot(np.arange(0, 12+deta_t, deta_t), [v[0, 0] for v in vPL_list], c="black")
plt.plot(np.arange(0, 12+deta_t, deta_t),[v[0, 0] for v in vRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel(r"v_x(m/s^2)")
plt.xlim(0, 12)
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)

plt.figure(1)
plt.subplot(3, 2, 6)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [v[1, i] for v in v_list], c=c)
plt.plot(np.arange(0, 12+deta_t, deta_t), [v[1, 0] for v in vPL_list], c="black")
plt.plot(np.arange(0, 12+deta_t, deta_t),[v[1, 0] for v in vRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel(r"v_y(m/s^2)")
plt.xlim(0, 12)
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)

plt.savefig("w2.2.jpg")
plt.show()

def draw(_i):
    plt.cla()
    for _j, _c in zip(range(4), c_list):
        plt.plot([x[0, _j] for x in x_list[:_i*50]], [x[1, _j] for x in x_list[:_i*50]], c=_c)
        plt.annotate('', xy=(x_list[_i*50][:, _j] + v_list[_i*50][:, _j]*0.2), xytext=(x_list[_i*50][:, _j]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=_c))
    plt.plot([x[0, 0] for x in xPL_list[:_i*50]], [x[1, 0] for x in xPL_list[:_i*50]], c="black")
    plt.annotate('', xy=(xPL_list[_i*50] + vPL_list[_i*50]*0.2), xytext=(xPL_list[_i*50]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="black"))
    plt.plot([x[0, 0] for x in xRL_list[:_i * 50]], [x[1, 0] for x in xRL_list[:_i * 50]], c="purple")
    plt.annotate('', xy=(xRL_list[_i * 50] + vRL_list[_i * 50] * 0.2), xytext=(xRL_list[_i * 50]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="purple"))
    plt.xlim(0, xPL_list[_i*50][0] + vPL_list[_i*50][0]*0.2 + 10)
    plt.ylim(0.5, 2.5)
    plt.xlabel("X Position(m)")
    plt.ylabel("Y Position(m)")
    plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"])
    plt.tight_layout()
fig = plt.figure(2)
an = ani.FuncAnimation(fig, draw, 200)
an.save("w2.1.gif")
plt.show()

# *****************************************************************

x_list1 = [np.matrix([[70, 60], [1, 1]])]
v_list1 = [np.matrix([[10, 10], [0, 0]])]
x_list2 = [np.matrix([[20, 10], [2, 2]])]
v_list2 = [np.matrix([[10, 10], [0, 0]])]
xL_list1 = [np.matrix([[80], [1]])]
vL_list1 = [np.matrix([[10], [0]])]
xL_list2 = [np.matrix([[30], [2]])]
vL_list2 = [np.matrix([[10], [0]])]
r_mat1 = np.matrix([[-10, -20], [0, 0]])
r_mat2 = np.matrix([[-40, -50], [0, 0]])
rL_mat2 = np.matrix([[-30], [0]])
rL_mat1 = np.matrix([[0], [0]])
for _ in range(t_num):
    deta_v_mat2 = np.zeros((2, 2))
    for i in range(2):
        deta_v_mat2[0, i] = xL_list2[-1][0, 0] - x_list2[-1][0, i] + vL_list2[-1][0, 0] -\
                           v_list2[-1][0, i] + r_mat2[0, i] - rL_mat2[0, 0]
        deta_v_mat2[1, i] = xL_list2[-1][1, 0] + r_mat2[1, i] - rL_mat2[1, 0] - x_list2[-1][1, i] - \
                           v_list2[-1][1, i] + vL_list2[-1][1, 0]
        deta_v_mat2[1, i] *= 5
    deta_vL_x = rL_mat2[0, 0] - rL_mat1[0, 0] + xL_list1[-1][0, 0] - 2 * xL_list2[-1][0, 0] + vL_list1[-1][0, 0] - vL_list2[-1][0, 0] + \
                x_list1[-1][0, 1] + rL_mat2[0, 0] - r_mat1[0, 1] + v_list1[-1][0, 1] - vL_list2[-1][0, 0]
    deta_vL_y = xL_list1[-1][1, 0] - xL_list2[-1][1, 0] + rL_mat2[1, 0] - rL_mat1[1, 0] + vL_list1[-1][1, 0] - vL_list2[-1][1, 0]
    deta_vL_y *= 5
    deta_vL_mat = np.matrix(np.ones((2,1)))
    deta_vL_mat[0, 0] *= deta_vL_x
    deta_vL_mat[1, 0] *= deta_vL_y
    x_list1.append(x_list1[-1] + deta_t * v_list1[-1])
    v_list1.append(v_list1[-1])
    x_list2.append(x_list2[-1] + deta_t * v_list2[-1])
    v_list2.append(v_list2[-1] + deta_t * deta_v_mat2)
    xL_list1.append(xL_list1[-1] + deta_t * vL_list1[-1])
    vL_list1.append((vL_list1[-1]))
    xL_list2.append(xL_list2[-1] + deta_t * vL_list2[-1])
    vL_list2.append(vL_list2[-1] + deta_t * deta_vL_mat)
print(x_list1[-1])
print(x_list2[-1])
print(xL_list1[-1])
print(xL_list2[-1])
for i in range(2):
    plt.plot([x[0, i] for x in x_list2], [x[1, i] for x in x_list2])
plt.plot([x[0, 0] for x in xL_list2], [x[1, 0] for x in xL_list2])
plt.show()

def draw2(_i):
    for j in range(2):
        plt.plot([x[0, j] for x in x_list2[:_i*50]], [x[1, j] for x in x_list2[:_i*50]])
    plt.plot([x[0, 0] for x in xL_list2[:_i*50]], [x[1, 0] for x in xL_list2[:_i*50]])
    plt.ylim(0.5, 2.2)
    plt.xlim(0, 200)
fig = plt.figure()
an = ani.FuncAnimation(fig, draw2, 200)
plt.show()

# *******************************************************************

x_list1 = [np.matrix([[0, 10, 21, 33, 45, 55], [1]*6])]
x_list2 = [np.matrix([[80, 92], [2, 1]])]
xL_list1 = [np.matrix([[72], [1]])]
xL_list2 = [np.matrix([[104], [1]])]

v_list1 = [np.matrix([[10]*6, [0]*6])]
v_list2 = [np.matrix([[10]*2, [0]*2])]
vL_list1 = [np.matrix([[10], [0]])] * 2
vL_list2 = [np.matrix([[10], [0]])]

r_mat1 = np.matrix([[i*10 for i in range(6)], [1]*6])
r_mat2 = np.matrix([[70, 80], [1, 1]])
rL_mat1 = np.matrix([[60], [1]])
rL_mat2 = np.matrix([[90], [1]])

for _ in range(t_num):
    a_mat1 = np.matrix(np.zeros((2, 6)))
    a_mat1[0, :] = xL_list1[-1][0, 0] - x_list1[-1][0, :] + r_mat1[0, :] - rL_mat1[0, 0] \
                   + vL_list1[-1][0, 0]  - v_list1[-1][0, :]
    a_mat1[1, :] = xL_list1[-1][1, 0] - x_list1[-1][1, :] + r_mat1[1, :] - rL_mat1[1, 0] \
                   + vL_list1[-1][1, 0]  - v_list1[-1][1, :]

    a_mat2 = np.matrix(np.zeros((2, 2)))
    a_mat2[0, :] = xL_list2[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat2[0, 0] \
                   + vL_list1[-1][0, 0] + vL_list2[-1][0, 0] -  2 * v_list2[-1][0, :] \
                   + xL_list1[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat1[0, 0]
    a_mat2[1, :] = xL_list2[-1][1, 0] - x_list2[-1][1, :] + r_mat2[1, :] - rL_mat2[1, 0] \
                   + vL_list2[-1][1, 0] - v_list2[-1][1, :]
    a_mat2[1, :] *= 5

    aL_mat1 = np.matrix(np.zeros((2, 1)))
    xp, vp, rp = (x_list1[-1][0, -1], v_list1[-1][0, -1], r_mat1[0, -1]) if vL_list1[-1][0, 0] < vL_list1[-2][0, 0] \
        else (x_list2[-1][0, 0], v_list2[-1][0, 0], r_mat2[0, 0])
    aL_mat1[0, 0] = xL_list2[-1][0, 0] - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rL_mat2[0, 0] \
                    - vL_list1[-1][0, 0] + vL_list2[-1][0, 0] \
                    + xp - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rp + vp - vL_list1[-1][0, 0]
    aL_mat1[1, 0] = xL_list2[-1][1, 0] - xL_list1[-1][1, 0]  + rL_mat1[1, 0] - rL_mat1[1, 0] \
                    + vL_list2[-1][1, 0] - vL_list1[-1][1, 0]

    aL_mat2 = np.matrix(np.zeros((2, 1)))

    x_list1.append(x_list1[-1] + deta_t*v_list1[-1])
    x_list2.append(x_list2[-1] + deta_t*v_list2[-1])
    xL_list1.append(xL_list1[-1] + deta_t*vL_list1[-1])
    xL_list2.append(xL_list2[-1] + deta_t*vL_list2[-1])
    v_list1.append(v_list1[-1] + deta_t*a_mat1)
    v_list2.append(v_list2[-1] + deta_t*a_mat2)
    vL_list1.append(vL_list1[-1] + deta_t*aL_mat1)
    vL_list2.append(vL_list2[-1] + deta_t*aL_mat2)

vL_list1 = vL_list1[1:]
for i in range(6):
    plt.plot([x[0, i] for x in x_list1], [x[1, i] for x in x_list1])
for i in range(2):
    plt.plot([x[0, i] for x in x_list2], [x[1, i] for x in x_list2])
plt.plot([x[0, 0] for x in xL_list1], [x[1, 0] for x in xL_list1])
plt.plot([x[0, 0] for x in xL_list2], [x[1, 0] for x in xL_list2])
plt.show()

plt.show()

def draw(i):
    plt.cla()
    for j in range(6):
        plt.plot([x[0, j] for x in x_list1[:i*50]], [x[1, j] for x in x_list1[:i*50]])
    for j in range(2):
        plt.plot([x[0, j] for x in x_list2[:i*50]], [x[1, j] for x in x_list2[:i*50]])
    plt.plot([x[0, 0] for x in xL_list1[:i*50]], [x[1, 0] for x in xL_list1[:i*50]])
    plt.plot([x[0, 0] for x in xL_list2[:i*50]], [x[1, 0] for x in xL_list2[:i*50]])

    plt.tight_layout()

fig = plt.figure()
an = ani.FuncAnimation(fig, draw, 200)
plt.show()

# **********************************************************

x_list1 = [np.matrix([[8, 18, 29], [2, 1, 1]])]
x_list2 = [np.matrix([[52, 64, 78, 89], [2, 1, 1, 1]])]
xL_list0 = [np.matrix([[0], [1]])]
xL_list1 = [np.matrix([[44], [1]])]
xL_list2 = [np.matrix([[104], [1]])]

v_list1 = [np.matrix([[10]*3, [0]*3])]
v_list2 = [np.matrix([[10]*4, [0]*4])]
vL_list0 = [np.matrix([[10], [0]])]
vL_list1 = [np.matrix([[10], [0]])] * 2
vL_list2 = [np.matrix([[10], [0]])]

r_mat1 = np.matrix([[i * 10 for i in range(1, 4)], [0]*3])
r_mat2 = np.matrix([[i * 10 for i in range(5, 9)], [0]*4])
rL_mat0 = np.matrix([[0], [0]])
rL_mat1 = np.matrix([[40], [0]])
rL_mat2 = np.matrix([[90], [0]])

for _ in range(t_num):
    a_mat1 = np.matrix(np.zeros((2, 3)))
    a_mat1[0, :] = xL_list1[-1][0, 0] - x_list1[-1][0, :] + r_mat1[0, :] - rL_mat1[0, 0] \
                   + vL_list1[-1][0, 0]  - v_list1[-1][0, :] \
                   + xL_list0[-1][0, 0] - x_list1[-1][0, :] + r_mat1[0, :] - rL_mat0[0, 0] \
                   + vL_list0[-1][0, 0] - v_list1[-1][0, :]
    a_mat1[1, :] = xL_list1[-1][1, 0] - x_list1[-1][1, :] + r_mat1[1, :] - rL_mat1[1, 0] \
                   + vL_list1[-1][1, 0]  - v_list1[-1][1, :]
    a_mat1[1, :] *= 5

    a_mat2 = np.matrix(np.zeros((2, 4)))
    a_mat2[0, :] = xL_list2[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat2[0, 0] \
                   + vL_list1[-1][0, 0] + vL_list2[-1][0, 0] -  2 * v_list2[-1][0, :] \
                   + xL_list1[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat1[0, 0]
    a_mat2[1, :] = xL_list2[-1][1, 0] - x_list2[-1][1, :] + r_mat2[1, :] - rL_mat2[1, 0] \
                   + vL_list2[-1][1, 0] - v_list2[-1][1, :]
    a_mat2[1, :] *= 5

    aL_mat0 = xL_list1[-1][:, 0] - xL_list0[-1][:, 0] + rL_mat0[:, 0] - rL_mat1[:, 0] +\
                    - vL_list0[-1][:, 0] + vL_list1[-1][:, 0]

    aL_mat1 = np.matrix(np.zeros((2, 1)))
    xp, vp, rp = (x_list1[-1][0, -1], v_list1[-1][0, -1], r_mat1[0, -1]) if vL_list1[-1][0, 0] < vL_list1[-2][0, 0] \
        else (x_list2[-1][0, 0], v_list2[-1][0, 0], r_mat2[0, 0])
    aL_mat1[0, 0] = xL_list2[-1][0, 0] - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rL_mat2[0, 0] \
                    - vL_list1[-1][0, 0] + vL_list2[-1][0, 0] \
                    + xp - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rp + vp - vL_list1[-1][0, 0]
    aL_mat1[1, 0] = xL_list2[-1][1, 0] - xL_list1[-1][1, 0]  + rL_mat1[1, 0] - rL_mat1[1, 0] \
                    + vL_list2[-1][1, 0] - vL_list1[-1][1, 0]

    aL_mat2 = np.matrix(np.zeros((2, 1)))

    x_list1.append(x_list1[-1] + deta_t*v_list1[-1])
    x_list2.append(x_list2[-1] + deta_t*v_list2[-1])
    xL_list0.append(xL_list0[-1] + deta_t*vL_list0[-1])
    xL_list1.append(xL_list1[-1] + deta_t*vL_list1[-1])
    xL_list2.append(xL_list2[-1] + deta_t*vL_list2[-1])
    v_list1.append(v_list1[-1] + deta_t*a_mat1)
    v_list2.append(v_list2[-1] + deta_t*a_mat2)
    vL_list0.append(vL_list0[-1] + deta_t*aL_mat0)
    vL_list1.append(vL_list1[-1] + deta_t*aL_mat1)
    vL_list2.append(vL_list2[-1] + deta_t*aL_mat2)
vL_list1 = vL_list1[1:]

for i in range(3):
    plt.plot([x[0, i] for x in x_list1], [x[1, i] for x in x_list1])
for i in range(4):
    plt.plot([x[0, i] for x in x_list2], [x[1, i] for x in x_list2])
for _x_list in [xL_list0, xL_list1, xL_list2]:
    plt.plot([x[0, 0] for x in _x_list], [x[1, 0] for x in _x_list])
plt.show()
"""
# ***************************************************

x_list1 = [np.matrix([[7, 17], [2, 1]])]
x_list2 = [np.matrix([[40, 60], [2, 2]])]
x_list3 = [np.matrix([[82, 93], [2, 2]])]
xL_list0 = [np.matrix([[0], [1]])]
xL_list1 = [np.matrix([[35], [1]])]
xL_list2 = [np.matrix([[75], [1]])]
xL_list3 = [np.matrix([[104], [1]])]

v_list1 = [np.matrix([[10]*2, [0]*2])]
v_list2 = [np.matrix([[10]*2, [0]*2])]
v_list3 = [np.matrix([[10]*2, [0]*2])]
vL_list0 = [np.matrix([[10], [0]])]
vL_list1 = [np.matrix([[10], [0]])] * 2
vL_list2 = [np.matrix([[10], [0]])] * 2
vL_list3 = [np.matrix([[10], [0]])]

rL_mat0 = np.matrix([[0], [0]])
r_mat1 = np.matrix([[10, 20], [0, 0]])
rL_mat1 = np.matrix([[30], [0]])
r_mat2 = np.matrix([[40, 50], [0, 0]])
rL_mat2 = np.matrix([[60], [0]])
r_mat3 = np.matrix([[70, 80], [0, 0]])
rL_mat3 = np.matrix([[90], [0]])

for _ in range(int(t_num / 5)):
    x_list1.append(x_list1[-1] + deta_t * v_list1[-1])
    x_list2.append(x_list2[-1] + deta_t * v_list2[-1])
    x_list3.append(x_list3[-1] + deta_t * v_list3[-1])
    xL_list0.append(xL_list0[-1] + deta_t * vL_list0[-1])
    xL_list1.append(xL_list1[-1] + deta_t * vL_list1[-1])
    xL_list2.append(xL_list2[-1] + deta_t * vL_list2[-1])
    xL_list3.append(xL_list3[-1] + deta_t * vL_list3[-1])
    v_list1.append(v_list1[-1])
    v_list2.append(v_list2[-1])
    v_list3.append(v_list3[-1])
    vL_list0.append(vL_list0[-1])
    vL_list1.append(vL_list1[-1])
    vL_list2.append(vL_list2[-1])
    vL_list3.append(vL_list3[-1])

a = 1
b = 1
c = 0.1
for _ in range(t_num):
    if t_num * 0.3 > _ > t_num * 0.1:
        c += 25 / t_num
    a_mat1 = np.matrix(np.zeros((2, 2)))
    a_mat1[0, :] = a * (xL_list1[-1][0, 0] - x_list1[-1][0, :] + r_mat1[0, :] - rL_mat1[0, 0]) \
                   + b * (vL_list1[-1][0, 0]  - v_list1[-1][0, :]) \
                   + a * (xL_list0[-1][0, 0] - x_list1[-1][0, :] + r_mat1[0, :] - rL_mat0[0, 0]) \
                   + b * (vL_list0[-1][0, 0] - v_list1[-1][0, :])
    a_mat1[1, :] = a * (xL_list1[-1][1, 0] - x_list1[-1][1, :] + r_mat1[1, :] - rL_mat1[1, 0]) \
                   + b * (vL_list1[-1][1, 0]  - v_list1[-1][1, :])
    a_mat1[1, :] *= c

    a_mat2 = np.matrix(np.zeros((2, 2)))
    a_mat2[0, :] = a * (xL_list2[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat2[0, 0]) \
                   + b * (vL_list1[-1][0, 0] + vL_list2[-1][0, 0] -  2 * v_list2[-1][0, :]) \
                   + a * (xL_list1[-1][0, 0] - x_list2[-1][0, :] + r_mat2[0, :] - rL_mat1[0, 0])
    a_mat2[1, :] = a * (xL_list2[-1][1, 0] - x_list2[-1][1, :] + r_mat2[1, :] - rL_mat2[1, 0]) \
                   + b * (vL_list2[-1][1, 0] - v_list2[-1][1, :])
    a_mat2[1, :] *= c

    a_mat3 = np.matrix(np.zeros((2, 2)))
    a_mat3[0, :] = a * (xL_list3[-1][0, 0] - x_list3[-1][0, :] + r_mat3[0, :] - rL_mat3[0, 0]) \
                   + b * (vL_list2[-1][0, 0] + vL_list3[-1][0, 0] - 2 * v_list3[-1][0, :]) \
                   + a * (xL_list2[-1][0, 0] - x_list3[-1][0, :] + r_mat3[0, :] - rL_mat2[0, 0])
    a_mat3[1, :] = a * (xL_list3[-1][1, 0] - x_list3[-1][1, :] + r_mat3[1, :] - rL_mat3[1, 0]) \
                   + b * (vL_list3[-1][1, 0] - v_list3[-1][1, :])
    a_mat3[1, :] *= c

    aL_mat0 = a * (xL_list1[-1][:, 0] - xL_list0[-1][:, 0] + rL_mat0[:, 0] - rL_mat1[:, 0]) +\
                    - b * (vL_list0[-1][:, 0] - vL_list1[-1][:, 0])

    aL_mat1 = np.matrix(np.zeros((2, 1)))
    xp, vp, rp = (x_list1[-1][0, -1], v_list1[-1][0, -1], r_mat1[0, -1]) if vL_list1[-1][0, 0] < vL_list1[-2][0, 0] \
        else (x_list2[-1][0, 0], v_list2[-1][0, 0], r_mat2[0, 0])
    aL_mat1[0, 0] = a * (xL_list2[-1][0, 0] - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rL_mat2[0, 0]) \
                    - b * (vL_list1[-1][0, 0] - vL_list2[-1][0, 0]) \
                    + a * (xp - xL_list1[-1][0, 0] + rL_mat1[0, 0] - rp) + b * (vp - vL_list1[-1][0, 0])
    aL_mat1[1, 0] = a * (xL_list2[-1][1, 0] - xL_list1[-1][1, 0]  + rL_mat1[1, 0] - rL_mat1[1, 0]) \
                    + b * (vL_list2[-1][1, 0] - vL_list1[-1][1, 0])

    aL_mat2 = np.matrix(np.zeros((2, 1)))
    xp, vp, rp = (x_list2[-1][0, -1], v_list2[-1][0, -1], r_mat2[0, -1]) if vL_list1[-1][0, 0] < vL_list1[-2][0, 0] \
        else (x_list3[-1][0, 0], v_list3[-1][0, 0], r_mat3[0, 0])
    aL_mat2[0, 0] = a * (xL_list3[-1][0, 0] - xL_list2[-1][0, 0] + rL_mat2[0, 0] - rL_mat3[0, 0]) \
                    - b * (vL_list2[-1][0, 0] - vL_list3[-1][0, 0]) \
                    + a * (xp - xL_list2[-1][0, 0] + rL_mat2[0, 0] - rp + vp - vL_list2[-1][0, 0])
    aL_mat2[1, 0] = a * (xL_list3[-1][1, 0] - xL_list2[-1][1, 0] + rL_mat2[1, 0] - rL_mat2[1, 0]) \
                    + b * (vL_list3[-1][1, 0] - vL_list2[-1][1, 0])

    aL_mat3 = np.matrix(np.zeros((2, 1)))

    x_list1.append(x_list1[-1] + deta_t*v_list1[-1])
    x_list2.append(x_list2[-1] + deta_t*v_list2[-1])
    x_list3.append(x_list3[-1] + deta_t*v_list3[-1])
    xL_list0.append(xL_list0[-1] + deta_t*vL_list0[-1])
    xL_list1.append(xL_list1[-1] + deta_t*vL_list1[-1])
    xL_list2.append(xL_list2[-1] + deta_t*vL_list2[-1])
    xL_list3.append(xL_list3[-1] + deta_t*vL_list3[-1])
    v_list1.append(v_list1[-1] + deta_t*a_mat1)
    v_list2.append(v_list2[-1] + deta_t*a_mat2)
    v_list3.append(v_list3[-1] + deta_t*a_mat3)
    vL_list0.append(vL_list0[-1] + deta_t*aL_mat0)
    vL_list1.append(vL_list1[-1] + deta_t*aL_mat1)
    vL_list2.append(vL_list2[-1] + deta_t*aL_mat2)
    vL_list3.append(vL_list3[-1] + deta_t*aL_mat3)
vL_list1 = vL_list1[1:]
vL_list2 = vL_list2[1:]

for _list in [x_list1, x_list2, x_list3]:
    for i in range(2):
        plt.plot([x[0, i] for x in _list], [x[1, i] for x in _list])
    print(_list[-1])
for _list in [xL_list1, xL_list2, xL_list3]:
    plt.plot([x[0, 0] for x in _list], [x[1, 0] for x in _list])
    print(_list[-1])
plt.show()

def draw(i):
    plt.cla()
    for _x_list, _v_list, _c_list in zip([x_list1, x_list2, x_list3], [v_list1, v_list2, v_list3],
                                         [["red", "blue"], ["orange", "green"], ["olive", "gold"]]):
        for j, c in zip(range(2), _c_list):
            plt.plot([x[0, j] for x in _x_list[:i*50]], [x[1, j] for x in _x_list[:i*50]], c=c)
            plt.annotate('', xy=(_x_list[i*50][:, j] + _v_list[i*50][:, j]*0.2), xytext=(_x_list[i*50][:, j]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=c))
    for _x_list, _v_list, c in zip([xL_list0, xL_list1, xL_list2, xL_list3],
                                         [vL_list0, vL_list1, vL_list2, vL_list3],
                                         ["brown", "black", "purple", "darkred"]):
        plt.plot([x[0, 0] for x in _x_list], [x[1, 0] for x in _x_list], c=c)
        plt.annotate('', xy=(_x_list[i * 50][:, 0] + _v_list[i * 50][:, 0] * 0.2), xytext=(_x_list[i * 50][:, 0]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=c))
    plt.legend(["Follower" + str(num) for num in range(1, 7)] + ["Leader" + str(num) for num in range(1, 5)])
    plt.xlim(max([xL_list0[i * 50][0, 0] - 20, 0]), xL_list3[i * 50][0, 0] + 30)
    plt.ylim(0, 3)
    plt.tight_layout()

fig = plt.figure(figsize=(18, 10))
an = ani.FuncAnimation(fig, draw, 200)
an.save("f.gif")
plt.show()
