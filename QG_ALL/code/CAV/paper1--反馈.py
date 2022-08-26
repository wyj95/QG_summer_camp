import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

deta_t = 1e-3
t_num = int(10 / deta_t)
beta = 1
gama = 1
case2_flag = False

x_list = []
xL_list = []
x_list.append(np.array([[6, 10, 16], [60, 40, 70]]))
xL_list.append(np.array([20, 50]))

v_list = []
vL_list = []
v_list.append(np.array([[10, 8, 9], [5, 4, 3]]))
vL_list.append(np.array([6, 0]))

r_mat = np.array([np.array([[-15], [0]]), np.array([[-10], [0]]), np.array([[-5], [0]])])
a_mat = np.ones((3, 3)) * 2
if case2_flag:
    a_mat[0, :] = 0
    a_mat[:, 0] = 0
for i in range(3):
    a_mat[i, i] = 0
k_list = np.array([0, 1, 1]) * 5

for t in range(t_num):
    temp_x_mat = x_list[-1] + deta_t * v_list[-1]
    temp_v_mat = np.zeros((2, 3))
    for i in range(3):
        temp_v_mat[:, i] = - k_list[i] * (x_list[-1][:, i] - xL_list[-1] - r_mat[i].T
                           + gama * (v_list[-1][:, i] - vL_list[-1]))
        for j in range(3):
            temp_v_mat[:, i] -= a_mat[i, j] * (x_list[-1][:, i] - x_list[-1][:, j] - (r_mat[i].T - r_mat[j].T)
                                               + beta * (v_list[-1][:, i] - v_list[-1][:, j]))[0]
    x_list.append(temp_x_mat)
    v_list.append(deta_t * temp_v_mat + v_list[-1])
    vL_list.append(vL_list[-1])
    xL_list.append(deta_t * vL_list[-1] + xL_list[-1])

c_list = ["blue", "gold", "red"]
"""
for i, c in zip(range(3), c_list):
    plt.plot([j[0, i] for j in x_list], [j[1, i] for j in x_list], c=c)
    plt.annotate('', xy=(x_list[-1][:, i]), xytext=(x_list[-1][:, i] - v_list[-1][:, i]),
                 arrowprops=dict(connectionstyle="arc3", facecolor=c))
plt.plot([i[0] for i in xL_list], [i[1] for i in xL_list], c="purple")
plt.annotate('', xy=(xL_list[-1]),xytext=(xL_list[-1] - vL_list[-1]),
             arrowprops=dict(connectionstyle="arc3", facecolor="purple"))
plt.xlabel("X Position(m)")
plt.ylabel("Y Position(m)")
plt.legend(["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"])
plt.title("fig.4" if not case2_flag else "fig.7")
plt.savefig("fig.4.jpg" if not case2_flag else "fig.7.jpg")
plt.show()
for i, c in zip(range(3), c_list):
    plt.plot(np.arange(0, 10+deta_t, deta_t), [j[0, i] - k[0] for j, k in zip(x_list, xL_list)], c=c)
plt.plot([0, 10], [0, 0], c="purple")
plt.xlabel("time(s)")
plt.ylabel("Longitudinal Gap(m)")
plt.legend(["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"])
plt.title("fig.5(a)" if not case2_flag else "fig.8(a)")
plt.savefig("fig.5(a).jpg" if not case2_flag else "fig.8(a).jpg")
plt.show()
for i, c in zip(range(3), c_list):
    plt.plot(np.arange(0, 10+deta_t, deta_t), [j[1, i] - k[1] for j, k in zip(x_list, xL_list)], c=c)
plt.plot([0, 10], [0, 0], c="purple")
plt.xlabel("time(s)")
plt.ylabel("Lateral Gap(m)")
plt.legend(["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"])
plt.title("fig.5(b)" if not case2_flag else "fig.8(b)")
plt.savefig("fig.5(b).jpg" if not case2_flag else "fig.8(b).jpg")
plt.show()
for i, c in zip(range(3), c_list):
    plt.plot(np.arange(0, 10+deta_t, deta_t), [j[0, i] for j in v_list], c=c)
plt.plot(np.arange(0, 10+deta_t, deta_t), [j[0] for j in vL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("X-Velocity(m/s)")
plt.legend(["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"])
plt.title("fig.6(a)" if not case2_flag else "fig.9(a)")
plt.savefig("fig.6(a).jpg" if not case2_flag else "fig.9(a).jpg")
plt.show()
for i, c in zip(range(3), c_list):
    plt.plot(np.arange(0, 10+deta_t, deta_t), [j[1, i] for j in v_list], c=c)
plt.plot(np.arange(0, 10+deta_t, deta_t), [j[1] for j in vL_list], c="purple")
plt.xlabel("time(s)")
plt.ylabel("Y-Velocity(m/s)")
plt.legend(["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"])
plt.title("fig.6(b)" if not case2_flag else "fig.9(b)")
plt.savefig("fig.6(b).jpg" if not case2_flag else "fig.9(b).jpg")
plt.show()
"""
def draw(_i):
    plt.cla()
    for _j, c in zip(range(3), c_list):
        plt.plot([x[0, _j] for x in x_list[:_i*50]], [x[1, _j] for x in x_list[:_i*50]], c=c)
        plt.annotate('', xy=(x_list[_i*50][:, _j] + v_list[_i*50][:, _j]*0.2), xytext=(x_list[_i*50][:, _j]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=c))
    plt.plot([x[0] for x in xL_list[:_i*50]], [x[1] for x in xL_list[:_i*50]], c="purple")
    plt.annotate('', xy=(xL_list[_i*50] + vL_list[_i*50]*0.2), xytext=(xL_list[_i*50]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="purple"))
    plt.xlim(0, xL_list[_i*50][0] + vL_list[_i*50][0]*0.2 + 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Follower1", "Follower2", "follower3", "Leader"])
    plt.tight_layout()
fig = plt.figure()
an = ani.FuncAnimation(fig, draw, int(len(x_list)/50))
an.save("w1.gif")
plt.show()