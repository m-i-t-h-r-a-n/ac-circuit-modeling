import numpy as np
from numpy.typing import ArrayLike

class DebyeCapacitor:
    '''
    Capacitor with a single-relaxation (Debye) frequency dependence.

    The measured (real-part) capacitance is

        C(f) = C_inf + dC / (1 + (2*pi*f*tau)**2)

    where the slow polarization responds at low frequency and freezes out
    at high frequency.

    Parameters
    ----------
    c_inf : float
        High-frequency plateau capacitance (the instant response).
    del_c : float
        Step height dC: the extra capacitance the slow process adds when it
        has time to respond. The low-frequency capacitance is c_inf + del_c.
    tau : float
        Relaxation time in seconds. The knee sits at f = 1 / (2*pi*tau).

    Notes
    -----
    c_inf and del_c can be in any capacitance unit (F or nF), as long as
    they are in the same one; the result comes back in that unit.
    '''
    def __init__(
            self,
            c_inf : float,
            del_c : float,
            tau : float
    ):
        self.c_inf = c_inf
        self.del_c = del_c
        self.tau = tau

    def computeCapacitance(self, f : ArrayLike):
        '''Return the capacitance at drive frequency f.

        Parameters
        ----------
        f : float or np.ndarray
            Frequency in Hz.

        Returns
        -------
        float or np.ndarray
            C(f), in the same unit as c_inf and del_c. 
        '''

        denom = 1 + ((2 * np.pi * f) ** 2) * (self.tau ** 2)

        return self.c_inf + self.del_c / denom
    

class ColeColeCapacitor:
    def __init__(
            self,
            c_inf : float,
            del_c : float,
            tau : float,
            alpha : float
    ) :
        self.c_inf = c_inf
        self.del_c = del_c
        self.tau = tau
        self.alpha = alpha
    
    def computeCapacitance(self, f: ArrayLike):
        '''Return the capacitance at drive frequency f, factoring in the broadening exponent alpha.

        Parameters
        ----------
        f : float or np.ndarray
            Frequency in Hz.

        Returns
        -------
        float or np.ndarray
            C(f), in the same unit as c_inf and del_c.
        '''
        w = 2 * np.pi * np.asarray(f, dtype=float)

        z = (1j * w * self.tau) ** (1 - self.alpha)

        return self.c_inf + np.real(self.del_c / (1 + z))