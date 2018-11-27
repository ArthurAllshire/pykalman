import math
import matplotlib.pyplot as plt
import numpy as np

from pykalman import Kalman

# simple state space model of a mass moving with 'noisy' unknown acceleration
# noisy samples of position are taken every dt seconds

time_range = 3 # seconds
dt = 1/100 # seconds
samples = time_range/dt + 1
n = 2 * math.pi
# variance in the noisy mesurement of the mass's position
sample_var = 0.1
# variance in the unknown acceleration governing the movement of the mass
acceleration_var = (n**2 / 2) ** 2 # m/s^2

# generate "truth" data
times = np.linspace(0, time_range, samples, endpoint=True)
x = np.cos(n*times)
velocities = -n*np.sin(n*times)
# generate noisy 'observations'. our first observation will be after the first
# timestep
obs = x[1:] + np.random.normal(0, sample_var, x[1:].shape)

# state update rule
F = np.array([[1, dt],
              [0, 1]])
# effect of unknown acceleration (the 'input') on the truck's state
# in this example acceleration is modelling the 'control input', hence this is
# the control input matrix
G = np.array([[dt**2 / 2],
              [dt]])
# covariance matrix representing the new uncertainty 'injected' each timestep
# due to the unknown acceleration
Q = G.dot(G.T)*acceleration_var**2
# the 'observation matrix' transforms the state space to the observation space
H = np.array([[1, 0]])
# covariance matrix representing the noise in the mesaurement of the mass'
# position
R = np.ones(shape=(1))*sample_var

# initial state and state estimate covariance. for the purpose of this
# example, assume we know with certainty the starting state of the system
# x_hat_init must be of shape (-1, 1), ie a column vector
x_hat_init = np.array([[x[0]], [velocities[0]]])
P_init = np.zeros(shape=(2,2))

# Initialise the Kalman filter. Passing R here is optional, it may be passed
# directly in the update function if different measuruments types are required.
fltr = Kalman(x_hat_init, P_init, Q, R=R)

pos_hats = np.zeros(shape=obs.shape)
vel_hats = np.zeros(shape=obs.shape)
for i, observation in enumerate(obs):
    pos_hats[i] = fltr.x_hat[0][0]
    vel_hats[i] = fltr.x_hat[1][0]
    # first predict forward. In our case the expected acceleration has zero
    # mean value, hence u is zero.
    fltr.predict(F, u=np.zeros(shape=(1, 1)), B=G)
    # now update
    z = np.array([[observation]])
    # you may also provide R (the sensor reading covariance) here to support
    # different measurement types or varying sensor reliability over time etc.
    fltr.update(z, H)

# matplotlib code to make a plot of the simulated results
plt_times = times[:-1]
truth_pos = x[:-1]
truth_vel = velocities[:-1]
fig, axs = plt.subplots(2, 1)
axs[0].set_ylabel('Position')
axs[1].set_xlabel('Time (sec)')
axs[1].set_ylabel('Velocity')

truth_line, = axs[0].plot(plt_times, truth_pos, dashes=[2, 2, 10, 2], label='Position')
estimate_line, = axs[0].plot(plt_times, pos_hats, dashes=[6, 2], label='Position Estimate')
observation_scatter = axs[0].scatter(plt_times, obs, c='red', s=1, label='Observations')

truth_v_line, = axs[1].plot(plt_times, truth_vel, dashes=[2, 2, 10, 2], label='Velocity')
estimate_v_line, = axs[1].plot(plt_times, vel_hats, dashes=[6, 2], label='Velocity Estimate')

axs[0].legend()
axs[1].legend()

plt.show()

