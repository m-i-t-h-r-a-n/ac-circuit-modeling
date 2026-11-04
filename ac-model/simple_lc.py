import numpy as np
from phasor_class import Phasor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class VoltageSource:
    def __init__(self, peak_voltage: float, phase: float, omega: float):
        self.eps = Phasor(peak_voltage, phase, omega)


class LCCircuit:
    """
    Series LC loop driven by a single AC voltage source.

    Unlike the standalone Capacitor/Inductor classes, this does NOT treat
    each component as if it alone saw the full source voltage. Instead it
    applies KVL in the impedance formulation directly (per §3.3):

        eps = I * (Z_L + Z_C)

    so there is exactly one loop current, determined by the *combined*
    impedance -- not two independently-computed currents.
    """

    def __init__(self, voltage_source: VoltageSource, capacitance: float, inductance: float):
        self.capacitance = capacitance
        self.inductance = inductance

        omega = voltage_source.eps.omega
        self.omega = omega

        # Impedances are plain complex numbers -- not Phasors. They are
        # frequency-dependent constants, not oscillating physical signals
        # in their own right (see §8.2's impedance() discussion).
        self.Z_L = 1j * omega * inductance
        self.Z_C = 1 / (1j * omega * capacitance)
        self.Z_total = self.Z_L + self.Z_C

        # KVL: eps = I * Z_total  =>  I = eps / Z_total
        eps_complex = voltage_source.eps.snapshot()
        I_complex = eps_complex / self.Z_total

        self.I = Phasor(
            amplitude=np.abs(I_complex),
            phi=np.angle(I_complex),
            omega=omega
        )

        # Voltage drops across each component, recovered from I * Z --
        # useful as a KVL sanity check: V_C + V_L should reconstruct eps.
        V_C_complex = I_complex * self.Z_C
        V_L_complex = I_complex * self.Z_L

        self.V_C = Phasor(np.abs(V_C_complex), np.angle(V_C_complex), omega)
        self.V_L = Phasor(np.abs(V_L_complex), np.angle(V_L_complex), omega)

    @property
    def resonant_omega(self) -> float:
        # omega_0 = 1/sqrt(LC), the omega at which Im[Z_total] = 0 
        return 1.0 / np.sqrt(self.inductance * self.capacitance)


# --- Create the circuit ---

eps0 = 1.0
phase = 0.0
omega = 2 * np.pi * 0.25   # a quarter rotation per second -> rad/s

C = 1.0
L = 1.0

source = VoltageSource(peak_voltage=eps0, phase=phase, omega=omega)
lc = LCCircuit(voltage_source=source, capacitance=C, inductance=L)

print(f"omega           = {omega:.4f} rad/s")
print(f"resonant_omega   = {lc.resonant_omega:.4f} rad/s")
print(f"Z_total          = {lc.Z_total}")
print(f"I amplitude/phase = {lc.I.amplitude:.4f}, {lc.I.phase:.4f} rad")

# --- Time base ---

period = 2 * np.pi / omega
t_max = 10 * period          # 10 full rotations
n_frames = 500
t = np.linspace(0, t_max, n_frames)

# --- Phasors (complex, rotating) ---

voltage_phasor = source.eps.at(t)
current_phasor = lc.I.at(t)

# --- Signals (real, physical) ---

voltage_signal = source.eps.signal(t)
current_signal = lc.I.signal(t)

# --- Figure: phasor plane (left) + 2 stacked signal traces (right) ---

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.3], height_ratios=[1, 1, 0.15])

ax_phasor = fig.add_subplot(gs[:2, 0])
ax_v = fig.add_subplot(gs[0, 1])
ax_i = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_subplot(gs[2, :])

# --- Phasor plane setup ---

max_phasor_amplitude = max(source.eps.amplitude, lc.I.amplitude) * 1.3

ax_phasor.set_xlim(-max_phasor_amplitude, max_phasor_amplitude)
ax_phasor.set_ylim(-max_phasor_amplitude, max_phasor_amplitude)
ax_phasor.set_aspect('equal')
ax_phasor.axhline(0, color='gray', lw=0.5)
ax_phasor.axvline(0, color='gray', lw=0.5)
ax_phasor.set_title("Phasor Plane -- Series LC")
ax_phasor.set_xlabel("Re")
ax_phasor.set_ylabel("Im")

voltage_arrow, = ax_phasor.plot([], [], 'r-', lw=2, label='eps')
current_arrow, = ax_phasor.plot([], [], 'm-', lw=2, label='I')
ax_phasor.legend(loc='upper right')

# --- Signal axes setup ---

for ax, label, sig in zip([ax_v, ax_i], ["eps(t)", "I(t)"], [voltage_signal, current_signal]):
    ax.set_xlim(0, t_max)
    max_sig = max(abs(sig.min()), abs(sig.max())) * 1.3
    ax.set_ylim(-max_sig, max_sig)
    ax.set_ylabel(label)
ax_i.set_xlabel("t (s)")

v_trace, = ax_v.plot(t, voltage_signal, 'r-')
i_trace, = ax_i.plot(t, current_signal, 'm-')

v_marker = ax_v.axvline(0, color='k', lw=1, ls='--')
i_marker = ax_i.axvline(0, color='k', lw=1, ls='--')

# --- Slider ---

time_slider = Slider(ax_slider, 't (s)', 0, t_max, valinit=0)


def update(val):
    frame_t = time_slider.val
    idx = np.searchsorted(t, frame_t)
    idx = min(idx, n_frames - 1)

    vx, vy = voltage_phasor[idx].real, voltage_phasor[idx].imag
    ix, iy = current_phasor[idx].real, current_phasor[idx].imag

    voltage_arrow.set_data([0, vx], [0, vy])
    current_arrow.set_data([0, ix], [0, iy])

    v_marker.set_xdata([frame_t, frame_t])
    i_marker.set_xdata([frame_t, frame_t])

    fig.canvas.draw_idle()


time_slider.on_changed(update)
update(0)   # draw the initial frame

plt.tight_layout()
plt.show()