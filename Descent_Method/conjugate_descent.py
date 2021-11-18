import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

### One Dimensional Example ###
def one_dimensional():
    def func_1(x):
        return x**2
    def gradient_1(x):
        return 2*x
    def hessian_1(x):
        return 2

    fig1 = plt.figure()
    x_func = np.linspace(-5,5,100)
    y_func = func_1(x_func)
    plt.plot(x_func, y_func)
    plt.title('Y=X^2')
    plt.xlabel('X')
    plt.ylabel('Y')

    x = 4
    ims=[]
    frames = []
    p = -gradient_1(x)
    H = hessian_1(x)
    for i in range(1):
        alpha_x = -gradient_1(x)*p/(p*H*p)
        x_ = x + alpha_x * p

        frame = plt.arrow(x, func_1(x), x_ - x, func_1(x_) - func_1(x), width=0.1, color='r')
        frames.append(frame)
        x = x_
        ims.append(frames.copy())

        r= gradient_1(x_)
        beta = p*H*r/(p*H*p)
        p = -r + beta * p

    ani = animation.ArtistAnimation(fig1, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/CD_1.gif", writer=writer)

### Two Dimensional Example ###
def two_dimensional():
    def func_2(x1,x2):
        return x1**2 + 2*x2**2

    def gradient_2(x):
        return np.array([2*x[0], 4*x[1]])

    hessian_2 = np.array([[2,0],[0,4]])

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

    ims = []
    frames = []
    x = np.array([10,10])
    p = -gradient_2(x)
    H = hessian_2

    for i in range(10):
        alpha = -gradient_2(x).transpose() @ p / (p.transpose() @ H @ p)
        x_ = x + alpha * p

        frame = plt.arrow(x[0], x[1], x_[0]-x[0], x_[1]-x[1], width=0.1, color='r')
        frames.append(frame)
        x = x_
        ims.append(frames.copy())

        r = gradient_2(x)
        beta = p.transpose() @ H @ r / (p.transpose() @ H @ p)
        p = -r + beta * p

    ani = animation.ArtistAnimation(fig2, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/CD_2.gif", writer=writer)

if __name__ == '__main__':
    one_dimensional()
    two_dimensional()

