import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

### One Dimensional Example ###
def func_1(x):
    return x**2
def gradient_1(x):
    return 2*x

def one_dimensional():
    fig1 = plt.figure()
    x_func = np.linspace(-5,5,100)
    y_func = func_1(x_func)
    plt.plot(x_func, y_func)
    plt.title('Y=X^2')
    plt.xlabel('X')
    plt.ylabel('Y')

    x = 4
    alpha = 0.1
    ims=[]
    frames = []
    for i in range(50):
        x_ = x - alpha*gradient_1(x)
        frame = plt.arrow(x, func_1(x), x_-x, func_1(x_)-func_1(x), width=0.1, color='r')
        frames.append(frame)
        x = x_
        ims.append(frames.copy())


    ani = animation.ArtistAnimation(fig1, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/GD_1.gif", writer=writer)

### Two Dimensional Example ###
def two_dimensional():
    def func_2(x1, x2):
        return x1 ** 2 + 2 * x2 ** 2

    def gradient_2(x1, x2):
        return 2 * x1, 4 * x2

    x1_func = np.arange(-10, 10, 0.01)
    x2_func = np.arange(-10, 10, 0.01)
    x1_func, x2_func = np.meshgrid(x1_func, x2_func)

    fig2 = plt.figure()
    y2 = func_2(x1_func,x2_func)
    contour = plt.contour(x1_func, x2_func, y2,[1,10,20,40,60,80,100,120,150,200])
    plt.clabel(contour,colors='r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Y = X1^2 - 2*X2^2')

    alpha = 0.1
    ims = []
    frames = []
    x1 = 10
    x2 = 10
    for i in range(50):
        x1_ = x1 - alpha * gradient_2(x1, x2)[0]
        x2_ = x2 - alpha * gradient_2(x1, x2)[1]
        frame = plt.arrow(x1, x2, x1_-x1, x2_-x2, width=0.1, color='r')

        frames.append(frame)
        x1 = x1_
        x2 = x2_
        ims.append(frames.copy())

    ani = animation.ArtistAnimation(fig2, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/GD_2.gif", writer=writer)

if __name__ == '__main__':
    one_dimensional()
    two_dimensional()