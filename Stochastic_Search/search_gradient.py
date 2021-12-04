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

x1_func = np.arange(-10, 12.5, 0.01)
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

x = [5, 5]
n_samples = 50

mu = x
sigma = np.eye(np.shape(x)[0])*3
eta = 0.1

for _ in range(100):
    if not np.all(np.linalg.eigvals(sigma)>0):
        break

    samples = np.random.multivariate_normal(mu, sigma, n_samples)
    frame_i = []
    for sam in samples:
        frame_i.append(plt.scatter(sam[0], sam[1]))
    ims.append(frame_i.copy())

    loss_samples = np.array([func(samples[i][0],samples[i][1]) for i in range(n_samples)])

    gradient_mu = 0
    gradient_sigma = 0
    inv_sigma = np.linalg.inv(sigma)
    for i in range(n_samples):
        diff = samples[i] - mu
        gradient_mu += (inv_sigma @ diff) * loss_samples[i]
        gradient_sigma += (-0.5*inv_sigma + 0.5 * inv_sigma @ diff[:,np.newaxis] @ diff[np.newaxis,:] @ inv_sigma) * loss_samples[i]

    gradient_mu = gradient_mu / n_samples
    gradient_sigma = gradient_sigma / n_samples
    mu = mu - eta *gradient_mu / np.linalg.norm(gradient_mu)
    sigma = sigma - eta * gradient_sigma / np.linalg.norm(gradient_sigma)



ani = animation.ArtistAnimation(fig, ims, interval=1)
writer = PillowWriter(fps=10)
ani.save("./gif/SG.gif", writer=writer)