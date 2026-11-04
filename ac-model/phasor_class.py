import numpy as np

class Phasor():
    def __init__(
            self,
            amplitude : float,
            phi : float,
            omega : float
    ):
        self.amplitude = amplitude
        self.phase = phi
        self.omega = omega

    def snapshot(self):
        X0 = self.amplitude * np.exp(1j * self.phase)
        return X0
    
    def at(self, t : np.linspace):
        X0 = self.snapshot()
        Xt = X0 * np.exp(1j * self.omega * t)
        return Xt
    
    def signal(self, t):
        return np.real(self.at(t))
    
    def dif_t(self):
        dXdt = Phasor(
            self.amplitude * self.omega,
            self.phase + np.pi/2,
            self.omega
        )
        return dXdt
    
    def int_t(self):
        if np.abs(self.omega) < 1e-6:
            raise ValueError("Cannot integrate DC phase (angular frequency = 0), result is unbounded, not a phasor")
        else:
            int_Xdt = Phasor(
                self.amplitude / self.omega,
                self.phase - np.pi / 2,
                self.omega
            )
        return int_Xdt