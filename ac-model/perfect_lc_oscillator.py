import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

@dataclass
class State:
    charge : float
    current : float

class Component():
    def delta_voltage():
        raise NotImplementedError()

class Capacitor(Component):
    def __init__(self, capacitance : float):
        self.capacitance = capacitance
    
    def delta_voltage(self, state : State):
        return state.charge / self.capacitance

class Inductor():
    def __init__(self, inductance):
        self.inductance = inductance

# Designing the Circuit Class

class Circuit():
    def __init__(
            self,
            inductor : Inductor,
            components : list[Component],
            eps : float
    ):
        self.inductor = inductor
        self.components = components
        self.eps = eps
    
    def derivatives(
            self,
            state : State
    ) -> tuple[float, float]:
        v_drop = sum(comp.delta_voltage(state) for comp in self.components)
        dqdt = state.current
        dIdt = (self.eps - v_drop) / self.inductor.inductance
        return dqdt, dIdt
    
    def _rhs(self, t : float, y : np.ndarray) -> list[float]:
        state = State(charge = y[0], current = y[1])
        dqdt, dIdt = self.derivatives(state)
        return [dqdt, dIdt]
    
    def simulate(
            self,
            t_start,
            t_end,
            q0 = 0.0,
            I0 = 0.0,
            n_points = 2000
    ):
        t_span = (t_start, t_end)
        t_eval = np.linspace(t_start, t_end, n_points)
        sol = solve_ivp(
            self._rhs,
            t_span,
            [q0, I0],
            t_eval=t_eval,
            rtol = 1e-9,
            atol = 1e-12
        )
        return sol.t, sol.y[0], sol.y[1] # t, q(t), I(t)
    
if __name__ == "__main__":
    L, C, eps = 1e-3, 1e-6, 1.0

    circuit = Circuit(
        inductor = Inductor(L),
        components = [Capacitor(C)],
        eps = eps
    )

    omega0 = 1 / np.sqrt(L * C)
    t, q, I = circuit.simulate(0, 3 * 2 * np.pi / omega0) # 3 Time periods simulation duration

    q_eq = eps * C
    U_C = q**2 / (2 * C)
    U_L = L * I**2 / 2

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, q, label="q(t)")
    ax[0].axhline(q_eq, color="k", ls="--", label="q_eq = eps*C")
    ax[0].legend(); ax[0].set_ylabel("Charge (C)")
 
    ax[1].plot(t, U_C, label="U_C")
    ax[1].plot(t, U_L, label="U_L")
    ax[1].legend(); ax[1].set_ylabel("Energy (J)"); ax[1].set_xlabel("Time (s)")
 
    plt.tight_layout()
    plt.savefig("driven_lc.png", dpi=150)
    print(f"omega_0 = {omega0:.4e} rad/s, q_eq = {q_eq:.4e} C")