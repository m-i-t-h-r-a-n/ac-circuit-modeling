import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class phasor():
    def __init__(
            self,
            A : float,
            phi : float
    ):
        self.amplitude = A
        self.phase = phi
    
    def initialize(self):
        X0 = self.amplitude * np.exp(1j * self.phase) # X = Ae^(i Phi)
        return X0

# Define for test case

t_max = 4 * np.pi                       # two full rotations
n_frames = 200
t = np.linspace(0, t_max, 200)

X = phasor(1.0, np.pi / 2).initialize()     # static part
omega = np.pi                           # half a rotation every second
X_t = X * np.exp(1j * omega * t)        # defining the full phasor

observed_signal = np.real(X_t)

# Plot

# --- Figure: phasor diagram (left) + time trace (right) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Phasor plane
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.set_aspect('equal')
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_title("Phasor plane")
ax1.set_xlabel("Re")
ax1.set_ylabel("Im")

arrow_line, = ax1.plot([], [], 'r-', lw=2)      # the rotating arrow
proj_line, = ax1.plot([], [], 'b--', lw=1)      # projection onto Re axis
proj_point, = ax1.plot([], [], 'bo', ms=6)      # dot at the projection

# Time trace
ax2.set_xlim(0, t_max)
ax2.set_ylim(-1.3, 1.3)
ax2.set_title("Re[X(t)] — the physical signal")
ax2.set_xlabel("t (s)")
trace_line, = ax2.plot([], [], 'b-', lw=2)

def init():
    arrow_line.set_data([], [])
    proj_line.set_data([], [])
    proj_point.set_data([], [])
    trace_line.set_data([], [])
    return arrow_line, proj_line, proj_point, trace_line

def update(frame):
    x, y = X_t[frame].real, X_t[frame].imag

    arrow_line.set_data([0, x], [0, y])          # arrow from origin to current phasor position
    proj_line.set_data([x, x], [0, y])           # vertical dashed line down to Re axis
    proj_point.set_data([x], [0])                # the dot riding along the Re axis

    trace_line.set_data(t[:frame+1], observed_signal[:frame+1])
    return arrow_line, proj_line, proj_point, trace_line

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=30, blit=True
)

plt.tight_layout()
plt.show()
# ani.save("rotating_phasor.gif", writer="pillow", fps=30)  # uncomment to export