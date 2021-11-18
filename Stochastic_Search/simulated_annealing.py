import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

### Two Dimensional Example ###
def func(x1,x2):
    return x1**2 + 2*x2**2

def sample_prob(prob_acc, x, x_):
    a = np.random.random()
    if a<=prob_acc:
        return x_
    else:
        return x

x1_func = np.arange(-10, 10, 0.01)
x2_func = np.arange(-10, 10, 0.01)
x1_func, x2_func = np.meshgrid(x1_func, x2_func)

fig = plt.figure()
y_func = func(x1_func,x2_func)
contour = plt.contour(x1_func, x2_func, y_func,[1,10,20,40,60,80,100,120,150,200])
plt.clabel(contour,colors='r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Y = X1^2 - 2*X2^2')

ims = []
frames = []

x = [10,10]
initial_T = 10
for i in range(50):
    x_c = np.random.multivariate_normal(x, np.eye(np.shape(x)[0]))
    if func(x_c[0],x_c[1])>=func(x[0],x[1]):
        T = initial_T/(i+1)
        prob = np.exp((func(x[0],x[1])-func(x_c[0],x_c[1]))/T)
        x_ = sample_prob(prob, x, x_c)
    else:
        x_ = x_c

    frame = plt.arrow(x[0], x[1], x_[0] - x[0], x_[1] - x[1], width=0.1, color='r')
    frames.append(frame)
    ims.append(frames.copy())
    x = x_


ani = animation.ArtistAnimation(fig, ims, interval=1)
writer = PillowWriter(fps=3)
ani.save("./gif/SA.gif", writer=writer)

