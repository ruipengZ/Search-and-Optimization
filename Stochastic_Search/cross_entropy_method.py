import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

### Two Dimensional Example ###
def func(x1,x2):
    return x1**2 + 2*x2**2 / x1 / x2

x1_func = np.arange(-10, 10, 0.01)
x2_func = np.arange(-10, 10, 0.01)
x1_func, x2_func = np.meshgrid(x1_func, x2_func)

fig = plt.figure()
y_func = func(x1_func,x2_func)
contour = plt.contour(x1_func, x2_func, y_func,[1,10,20,40,60,80,100,120,150,200])
plt.clabel(contour,colors='r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Y = X1^2 + 2*X2^2')

ims = []
frames = []
x = [5,5]
n_samples = 50

loss = []
samples = np.random.multivariate_normal(x, np.eye(np.shape(x)[0])*5, n_samples)
for i in range(10):
    frame_i = []
    loss_samples = np.array([func(samples[i][0],samples[i][1]) for i in range(np.shape(samples)[0])])
    loss += [np.mean(loss_samples)]
    elite_samples = samples[np.argsort(loss_samples)[:int(n_samples*0.3)]]

    for sam in samples:
        if sam in elite_samples:
            frame_i.append(plt.scatter(sam[0], sam[1],c='r'))
        else:
            frame_i.append(plt.scatter(sam[0], sam[1],c='g'))
    ims.append(frame_i.copy())

    new_mean = np.mean(elite_samples, axis = 0)
    new_sigma = np.cov(elite_samples.transpose(),ddof=0)
    new_sigma = np.diag(np.diagonal(new_sigma))
    samples = np.random.multivariate_normal(new_mean, new_sigma, n_samples)

ani = animation.ArtistAnimation(fig, ims, interval=1)
writer = PillowWriter(fps=3)
# ani.save("./gif/CEM.gif", writer=writer)
# ani.save("./gif/CEM_cov.gif", writer=writer)
ani.save("./gif/CEM_ind.gif", writer=writer)
