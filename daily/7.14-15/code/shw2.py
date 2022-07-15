import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.animation as ani

deta_t = 1e-3
t_num = int(12 / deta_t)

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
plt.show()

plt.figure(figsize=(12, 15))
plt.figure(1)
plt.subplot(3, 2, 1)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[0, i] for x in x_list], c=c)
plt.plot(np.arange(0, 12+deta_t, deta_t), [x[0, 0] for x in xPL_list], c="black")
plt.plot(np.arange(0, 12+deta_t, deta_t), [x[0, 0] for x in xRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("X Position(m)")
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)
plt.xlim(0, 12)

plt.figure(1)
plt.subplot(3, 2, 2)
for i, c in zip(range(4), c_list):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[1, i] for x in x_list], c=c)
plt.plot(np.arange(0, 12+deta_t, deta_t), [x[1, 0] for x in xPL_list], c="black")
plt.plot(np.arange(0, 12+deta_t, deta_t), [x[1, 0] for x in xRL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("Y Position(m)")
plt.xlim(0, 12)
plt.legend(["Flower 1", "Flower 2", "Flower 3", "Flower 4", "Leader 1", "Leader 2"], loc=1)

plt.figure(1)
plt.subplot(3, 2, 3)
for i in range(4):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[0, i] - y[0, 0] for x, y in zip(x_list, xPL_list)])
for i in range(4):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[0, i] - y[0, 0] for x, y in zip(x_list, xRL_list)])
str_list = ["L" + str(i) + " - L" + j for i, j in zip(list(range(4)) * 2, ["1"] * 4 + ["2"] * 4)]
plt.legend(str_list, loc=1)
plt.plot([0, 12], [0, 0])
plt.xlabel("time(s)")
plt.ylabel("deta_x")
plt.xlim(0, 12)

plt.figure(1)
plt.subplot(3, 2, 4)
for i in range(4):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[1, i] - y[1, 0] for x, y in zip(x_list, xPL_list)])
for i in range(4):
    plt.plot(np.arange(0, 12+deta_t, deta_t), [x[1, i] - y[1, 0] for x, y in zip(x_list, xRL_list)])
str_list = ["L" + str(i+1) + " - L" + j for i, j in zip(list(range(4)) * 2, ["1"] * 4 + ["2"] * 4)]
plt.legend(str_list, loc=1)
plt.plot([0, 12], [0, 0], c="black")
plt.xlabel("time(s)")
plt.ylabel("deta_y(m)")
plt.xlim(0, 12)

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

plt.savefig("w2.1.jpg")
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
