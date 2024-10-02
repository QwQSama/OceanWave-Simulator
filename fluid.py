import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define 3d data
domain_min = 0
domain_max = 30
domain_maxy = domain_max
step_siza = 0.1
xx = np.arange(domain_min, domain_max, step_siza)
yy = np.arange(domain_min, domain_maxy, step_siza)
X, Y = np.meshgrid(xx, yy)
print('x shape = ', X.shape)
print('y shape = ', Y.shape)
Z = np.zeros(X.shape)
z0 = np.zeros(X.shape)
init_z = 0.1

for i in range(z0.shape[0]):
    for j in range(z0.shape[1]):
        z0[i, j] = 1

g = 9.8
V = 2
w = np.sqrt(2 / 3) * g / V
r = g / w ** 2

kr = 0.2
k = kr / r
print('k = ', k)

H = 2 * r
L = 2 * np.pi / k
T = 2 * np.pi / w
c = w / k
steepness = H / L
lamb = 2.5

# wind parameters
wind_speed = 5

# Time array
t_start = 0.0  # s, start time
t_stop = 20.0  # s, stop time
num_steps = 200  # number of time steps
dt = (t_stop - t_start) / num_steps  # time step
l_t = np.linspace(t_start, t_stop, num_steps)
t = 0


def depth(x, y):
    if x < domain_max / 2:
        h = 10
    else:
        h = 1
    return h


def depth_effect(x, y):
    h = depth(x, y)

    k_inf = k ** 2 * h
    if h / L < 0.5:
        k_inf = k
    dx = np.abs(xx[0] - xx[1])
    if k_inf == 0:
        res = 0
    else:
        res = k_inf * (-1 + 1 / (np.sqrt(np.tanh(k_inf * h)))) * dx
    return res


l_d_x = []
l_d = []

print('depth', depth(15, 0))
print('depth_E', depth_effect(15, 5))


def sum_depth_effect(x, y):
    if x in l_d_x:
        return l_d[l_d_x.index(x)]
    sum_D = 0
    for i in range(len(xx)):
        if xx[i] < x:
            sum_D += depth_effect(xx[i], y)
    if x not in l_d_x:
        l_d_x.append(x)
        l_d.append(sum_D)
    return sum_D


def wave(x, y, z0, t):
    phi0 = k * x - w * t
    dy = np.abs(yy[0] - yy[1])
    dz = -r * np.cos(phi0)

    D = sum_depth_effect(x, y)

    rotation = 30 * np.pi / 180

    dy = y / np.floor(domain_max / L) * np.cos(rotation)
    x = x + dy

    phi = k * x - w * t - lamb * dz * dt + D

    wave_x = x + r * np.sin(phi)
    wave_z = z0 - r * np.cos(phi)
    wave_y = y

    wave_x += (k / w) * lamb * np.cos(phi) * np.cos(rotation)
    wave_y += (k / w) * lamb * np.cos(phi) * np.sin(rotation)
    wave_z += (k / w) * lamb * np.sin(phi)

    wave_x -= dy

    return [wave_x, wave_y, wave_z]


lx = np.zeros_like(X)
ly = np.zeros_like(Y)
lz = np.zeros_like(Z)
n, m = X.shape
for i in range(n):
    for j in range(m):
        x = X[i, j]
        y = Y[i, j]
        z = z0[i, j]
        pos = wave(x, y, z, t)
        lx[i, j] = pos[0]
        ly[i, j] = pos[1]
        lz[i, j] = pos[2]

waves = [lx, ly, lz]


def update(i):
    ax.clear()
    ax.set_zlim(0, 10)
    t = i * dt
    lx = np.zeros_like(X)
    ly = np.zeros_like(Y)
    lz = np.zeros_like(Z)
    for i in range(len(xx)):
        sum_D = 0
        for j in range(len(yy)):
            pos = wave(xx[i], yy[j], z0[i, j], t)
            lx[i, j] = pos[0]
            ly[i, j] = pos[1]
            lz[i, j] = pos[2]
        sum_D += depth_effect(xx[i], yy[j])

    surf = ax.plot_surface(lx, ly, lz, cmap='coolwarm')


# 3d animation
# Set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(domain_min, domain_max)
ax.set_ylim(domain_min, domain_max)
ax.set_zlim(0, 10)

# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100)

# Save the animation as a GIF
writergif = animation.PillowWriter(fps=10)
ani.save('fluid.gif', writergif)
