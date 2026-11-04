import numpy as np
from phasor_class import Phasor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class VoltageSource:
    def __init__(self, peak_voltage: float, phase: float, omega: float):
        self.eps = Phasor(peak_voltage, phase, omega)


class Capacitor:
    def __init__(self, voltage_source: VoltageSource, capacitance: float):
        self.capacitance = capacitance
        dVdt = voltage_source.eps.dif_t()
        self.I = Phasor(
            dVdt.amplitude * capacitance,
            dVdt.phase,
            dVdt.omega
        )


class Inductor:
    def __init__(self, voltage_source: VoltageSource, inductance: float):
        self.inductance = inductance
        intVdt = voltage_source.eps.int_t()
        self.I = Phasor(
            intVdt.amplitude / self.inductance,
            intVdt.phase,
            intVdt.omega
        )


# --- Create components ---

eps0 = 1.0
phase = 0.0
omega = 2 * np.pi * 0.25   # a quarter rotation per second -> rad/s

C = 1.0
L = 1.0

source = VoltageSource(peak_voltage=eps0, phase=phase, omega=omega)
capacitor = Capacitor(voltage_source=source, capacitance=C)
inductor = Inductor(voltage_source=source, inductance=L)

# --- Time base ---

period = 2 * np.pi / omega
t_max = 10 * period          # 10 full rotations
n_frames = 500
t = np.linspace(0, t_max, n_frames)

# --- Phasors (complex, rotating) ---

voltage_phasor = source.eps.at(t)
capacitor_phasor = capacitor.I.at(t)
inductor_phasor = inductor.I.at(t)

# --- Signals (real, physical) ---

voltage_signal = source.eps.signal(t)
capacitor_signal = capacitor.I.signal(t)
inductor_signal = inductor.I.signal(t)

# --- Figure: phasor plane (left) + 3 stacked signal traces (right) ---

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.3], height_ratios=[1, 1, 1, 0.15])

ax_phasor = fig.add_subplot(gs[:3, 0])
ax_v = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])
ax_l = fig.add_subplot(gs[2, 1])
ax_slider = fig.add_subplot(gs[3, :])

# --- Phasor plane setup ---

max_phasor_amplitude = max(
    source.eps.amplitude,
    capacitor.I.amplitude,
    inductor.I.amplitude
) * 1.3

ax_phasor.set_xlim(-max_phasor_amplitude, max_phasor_amplitude)
ax_phasor.set_ylim(-max_phasor_amplitude, max_phasor_amplitude)
ax_phasor.set_aspect('equal')
ax_phasor.axhline(0, color='gray', lw=0.5)
ax_phasor.axvline(0, color='gray', lw=0.5)
ax_phasor.set_title("Phasor Plane")
ax_phasor.set_xlabel("Re")
ax_phasor.set_ylabel("Im")

voltage_arrow, = ax_phasor.plot([], [], 'r-', lw=2, label='eps')
cap_arrow, = ax_phasor.plot([], [], 'g-', lw=2, label='I_C')
ind_arrow, = ax_phasor.plot([], [], 'b-', lw=2, label='I_L')
ax_phasor.legend(loc='upper right')

# --- Signal axes setup ---

for ax, label, sig in zip(
    [ax_v, ax_c, ax_l],
    ["eps(t)", "I_C(t)", "I_L(t)"],
    [voltage_signal, capacitor_signal, inductor_signal]
):
    ax.set_xlim(0, t_max)
    max_sig = max(abs(sig.min()), abs(sig.max())) * 1.3
    ax.set_ylim(-max_sig, max_sig)
    ax.set_ylabel(label)
ax_l.set_xlabel("t (s)")

# The signal traces are drawn in full up front (no need to build them up
# incrementally like the old frame-by-frame animation) -- the slider instead
# drives a vertical marker showing "where we are" on each fixed trace.
v_trace, = ax_v.plot(t, voltage_signal, 'r-')
c_trace, = ax_c.plot(t, capacitor_signal, 'g-')
l_trace, = ax_l.plot(t, inductor_signal, 'b-')

v_marker = ax_v.axvline(0, color='k', lw=1, ls='--')
c_marker = ax_c.axvline(0, color='k', lw=1, ls='--')
l_marker = ax_l.axvline(0, color='k', lw=1, ls='--')

# --- Slider ---

time_slider = Slider(ax_slider, 't (s)', 0, t_max, valinit=0)


def update(val):
    frame_t = time_slider.val
    idx = np.searchsorted(t, frame_t)
    idx = min(idx, n_frames - 1)

    vx, vy = voltage_phasor[idx].real, voltage_phasor[idx].imag
    cx, cy = capacitor_phasor[idx].real, capacitor_phasor[idx].imag
    lx, ly = inductor_phasor[idx].real, inductor_phasor[idx].imag

    voltage_arrow.set_data([0, vx], [0, vy])
    cap_arrow.set_data([0, cx], [0, cy])
    ind_arrow.set_data([0, lx], [0, ly])

    v_marker.set_xdata([frame_t, frame_t])
    c_marker.set_xdata([frame_t, frame_t])
    l_marker.set_xdata([frame_t, frame_t])

    fig.canvas.draw_idle()


time_slider.on_changed(update)
update(0)

plt.tight_layout()
plt.show()