import numpy as np
from phasor_class import Phasor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class VoltageSource:
    def __init__(self, peak_voltage: float, phase: float, omega: float):
        self.eps = Phasor(peak_voltage, phase, omega)


class LCCircuit:
    """
    Series LC loop driven by a single AC voltage source, via KVL in the
    impedance formulation (§3.3): eps = I * (Z_L + Z_C).
    """

    def __init__(self, voltage_source: VoltageSource, capacitance: float, inductance: float):
        self.capacitance = capacitance
        self.inductance = inductance

        omega = voltage_source.eps.omega
        self.omega = omega

        self.Z_L = 1j * omega * inductance
        self.Z_C = 1 / (1j * omega * capacitance)
        self.Z_total = self.Z_L + self.Z_C

        eps_complex = voltage_source.eps.snapshot()
        I_complex = eps_complex / self.Z_total

        self.I = Phasor(
            amplitude=np.abs(I_complex),
            phi=np.angle(I_complex),
            omega=omega
        )

        V_C_complex = I_complex * self.Z_C
        V_L_complex = I_complex * self.Z_L
        self.V_C = Phasor(np.abs(V_C_complex), np.angle(V_C_complex), omega)
        self.V_L = Phasor(np.abs(V_L_complex), np.angle(V_L_complex), omega)

    @property
    def resonant_omega(self) -> float:
        return 1.0 / np.sqrt(self.inductance * self.capacitance)


# --- Initial parameter values ---

eps0_init = 1.0
omega_init = 2 * np.pi * 0.25
C_init = 1.0
L_init = 1.0
phase = 0.0

n_frames = 500
n_rotations = 10

# --- Figure layout ---
# Phasor plane (top-left), 2 stacked signal traces (top-right),
# 4 parameter sliders + 1 time slider along the bottom.

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(
    7, 2,
    width_ratios=[1, 1.3],
    height_ratios=[1, 1, 0.1, 0.1, 0.1, 0.1, 0.1],
    hspace=0.6
)

ax_phasor = fig.add_subplot(gs[0:2, 0])
ax_v = fig.add_subplot(gs[0, 1])
ax_i = fig.add_subplot(gs[1, 1])

ax_s_eps = fig.add_subplot(gs[2, :])
ax_s_omega = fig.add_subplot(gs[3, :])
ax_s_C = fig.add_subplot(gs[4, :])
ax_s_L = fig.add_subplot(gs[5, :])
ax_s_t = fig.add_subplot(gs[6, :])

ax_phasor.set_aspect('equal')
ax_phasor.axhline(0, color='gray', lw=0.5)
ax_phasor.axvline(0, color='gray', lw=0.5)
ax_phasor.set_title("Phasor Plane -- Series LC")
ax_phasor.set_xlabel("Re")
ax_phasor.set_ylabel("Im")

voltage_arrow, = ax_phasor.plot([], [], 'r-', lw=2, label='eps')
current_arrow, = ax_phasor.plot([], [], 'm-', lw=2, label='I')
ax_phasor.legend(loc='upper right')

ax_v.set_ylabel("eps(t)")
ax_i.set_ylabel("I(t)")
ax_i.set_xlabel("t (s)")

v_trace, = ax_v.plot([], [], 'r-')
i_trace, = ax_i.plot([], [], 'm-')

v_marker = ax_v.axvline(0, color='k', lw=1, ls='--')
i_marker = ax_i.axvline(0, color='k', lw=1, ls='--')

# --- Sliders ---

eps_slider = Slider(ax_s_eps, 'eps0', 0.1, 5.0, valinit=eps0_init)
omega_slider = Slider(ax_s_omega, 'omega', 0.1, 10.0, valinit=omega_init)
C_slider = Slider(ax_s_C, 'C', 0.1, 5.0, valinit=C_init)
L_slider = Slider(ax_s_L, 'L', 0.1, 5.0, valinit=L_init)
t_slider = Slider(ax_s_t, 't (s)', 0, 1.0, valinit=0)  # rescaled to t_max on every rebuild

# --- State that gets recomputed whenever a parameter slider moves ---

state = {}


def rebuild_circuit(_=None):
    eps0 = eps_slider.val
    omega = omega_slider.val
    C = C_slider.val
    L = L_slider.val

    source = VoltageSource(peak_voltage=eps0, phase=phase, omega=omega)
    lc = LCCircuit(voltage_source=source, capacitance=C, inductance=L)

    period = 2 * np.pi / omega
    t_max = n_rotations * period
    t = np.linspace(0, t_max, n_frames)

    voltage_phasor = source.eps.at(t)
    current_phasor = lc.I.at(t)
    voltage_signal = source.eps.signal(t)
    current_signal = lc.I.signal(t)

    state.update(
        source=source, lc=lc, t=t, t_max=t_max,
        voltage_phasor=voltage_phasor, current_phasor=current_phasor,
        voltage_signal=voltage_signal, current_signal=current_signal
    )

    # Rescale axes for the new parameter values
    max_phasor_amplitude = max(source.eps.amplitude, lc.I.amplitude) * 1.3
    ax_phasor.set_xlim(-max_phasor_amplitude, max_phasor_amplitude)
    ax_phasor.set_ylim(-max_phasor_amplitude, max_phasor_amplitude)

    ax_v.set_xlim(0, t_max)
    ax_i.set_xlim(0, t_max)
    max_v = max(abs(voltage_signal.min()), abs(voltage_signal.max())) * 1.3
    max_i = max(abs(current_signal.min()), abs(current_signal.max())) * 1.3
    ax_v.set_ylim(-max_v, max_v)
    ax_i.set_ylim(-max_i, max_i)

    v_trace.set_data(t, voltage_signal)
    i_trace.set_data(t, current_signal)

    # Re-scale the time slider's range to the new t_max, keeping its
    # current fractional position rather than resetting to zero.
    frac = t_slider.val / t_slider.valmax if t_slider.valmax > 0 else 0
    t_slider.valmax = t_max
    t_slider.ax.set_xlim(t_slider.valmin, t_max)
    t_slider.set_val(frac * t_max)

    update_time(t_slider.val)
    fig.canvas.draw_idle()


def update_time(val):
    t = state['t']
    idx = np.searchsorted(t, val)
    idx = min(idx, n_frames - 1)

    voltage_phasor = state['voltage_phasor']
    current_phasor = state['current_phasor']

    vx, vy = voltage_phasor[idx].real, voltage_phasor[idx].imag
    ix, iy = current_phasor[idx].real, current_phasor[idx].imag

    voltage_arrow.set_data([0, vx], [0, vy])
    current_arrow.set_data([0, ix], [0, iy])

    v_marker.set_xdata([val, val])
    i_marker.set_xdata([val, val])

    fig.canvas.draw_idle()


eps_slider.on_changed(rebuild_circuit)
omega_slider.on_changed(rebuild_circuit)
C_slider.on_changed(rebuild_circuit)
L_slider.on_changed(rebuild_circuit)
t_slider.on_changed(update_time)

rebuild_circuit()   # initial draw

plt.show()