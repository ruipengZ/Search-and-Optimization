import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

### One Dimensional Example ###
def func_1(x):
    return x**2
def gradient_1(x):
    return 2*x

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
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
    # frame = ax1.annotate("",
    #             xy=(x_, func_1(x_)),
    #             xytext=(x, func_1(x)),
    #             # xycoords="figure points",
    #             arrowprops=dict(arrowstyle="->", color="r"))
    #             # arrowprops = dict(facecolor='r',  width=0.05))
    frame = plt.arrow(x, func_1(x), x_-x, func_1(x_)-func_1(x), width=0.1, color='r')
    frames.append(frame)
    x = x_
    ims.append(frames.copy())


ani = animation.ArtistAnimation(fig1, ims, interval=1)
writer = PillowWriter(fps=20)
ani.save("./gif/GD_1.gif", writer=writer)

### Two Dimensional Example ###
def func_2(x1,x2):
    return x1**2 + 2*x2**2
def gradient_2(x1,x2):
    return 2*x1, 4*x2
x1_func = np.arange(-10, 10, 0.01)
x2_func = np.arange(-10, 10, 0.01)
x1_func, x2_func = np.meshgrid(x1_func, x2_func)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
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
    # frame = ax.annotate("",
    #             xy=(x_, func_1(x_)),
    #             xytext=(x, func_1(x)),
    #             # xycoords="figure points",
    #             arrowprops=dict(arrowstyle="->", color="r"))
    #             # arrowprops = dict(facecolor='r',  width=0.05))
    frame = plt.arrow(x1, x2, x1_-x1, x2_-x2, width=0.1, color='r')

    frames.append(frame)
    x1 = x1_
    x2 = x2_
    ims.append(frames.copy())

ani = animation.ArtistAnimation(fig2, ims, interval=1)
writer = PillowWriter(fps=20)
ani.save("./gif/GD_2.gif", writer=writer)

#
# # The data to fit
# m = 20
# theta0_true = 2
# theta1_true = 0.5
# x = np.linspace(-1,1,m)
# y = theta0_true + theta1_true * x
#
# # The plot: LHS is the data, RHS will be the cost function.
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
# ax[0].scatter(x, y, marker='x', s=40, color='k')
#
# def cost_func(theta0, theta1):
#     """The cost function, J(theta0, theta1) describing the goodness of fit."""
#     theta0 = np.atleast_3d(np.asarray(theta0))
#     theta1 = np.atleast_3d(np.asarray(theta1))
#     return np.average((y-hypothesis(x, theta0, theta1))**2, axis=2)/2
#
# def hypothesis(x, theta0, theta1):
#     """Our "hypothesis function", a straight line."""
#     return theta0 + theta1*x
#
# # First construct a grid of (theta0, theta1) parameter pairs and their
# # corresponding cost function values.
# theta0_grid = np.linspace(-1,4,101)
# theta1_grid = np.linspace(-5,5,101)
# J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
#                    theta1_grid[:,np.newaxis,np.newaxis])
#
# # A labeled contour plot for the RHS cost function
# X, Y = np.meshgrid(theta0_grid, theta1_grid)
# contours = ax[1].contour(X, Y, J_grid, 30)
# ax[1].clabel(contours)
# # The target parameter values indicated on the cost function contour plot
# ax[1].scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])
#
# # Take N steps with learning rate alpha down the steepest gradient,
# # starting at (theta0, theta1) = (0, 0).
# N = 5
# alpha = 0.7
# theta = [np.array((0,0))]
# J = [cost_func(*theta[0])[0]]
# for j in range(N-1):
#     last_theta = theta[-1]
#     this_theta = np.empty((2,))
#     this_theta[0] = last_theta[0] - alpha / m * np.sum(
#                                     (hypothesis(x, *last_theta) - y))
#     this_theta[1] = last_theta[1] - alpha / m * np.sum(
#                                     (hypothesis(x, *last_theta) - y) * x)
#     theta.append(this_theta)
#     J.append(cost_func(*this_theta))
#
#
# # Annotate the cost function plot with coloured points indicating the
# # parameters chosen and red arrows indicating the steps down the gradient.
# # Also plot the fit function on the LHS data plot in a matching colour.
# colors = ['b', 'g', 'm', 'c', 'orange']
# ax[0].plot(x, hypothesis(x, *theta[0]), color=colors[0], lw=2,
#            label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[0]))
# for j in range(1,N):
#     ax[1].annotate('', xy=theta[j], xytext=theta[j-1],
#                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
#                    va='center', ha='center')
#     ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
#            label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
# ax[1].scatter(*zip(*theta), c=colors, s=40, lw=0)
#
# # Labels, titles and a legend.
# ax[1].set_xlabel(r'$\theta_0$')
# ax[1].set_ylabel(r'$\theta_1$')
# ax[1].set_title('Cost function')
# ax[0].set_xlabel(r'$x$')
# ax[0].set_ylabel(r'$y$')
# ax[0].set_title('Data and fit')
# axbox = ax[0].get_position()
# # Position the legend by hand so that it doesn't cover up any of the lines.
# ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
#              fontsize='small')
#
# plt.show()