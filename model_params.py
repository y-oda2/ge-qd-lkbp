"""
Device Parameters
"""

import numpy as np
from dataclasses import dataclass, field

# ---------- Constants
HBAR  = 1.054_571_817e-34
E_CH  = 1.602_176_634e-19
M0    = 9.109_383_7015e-31
MU_B  = E_CH * HBAR / (2 * M0)
PI    = np.pi

# ---------- Unit Conversion from eV, Joules and GHz
eV2J  = E_CH
J2neV = 1e9/E_CH
J2ueV = 1e6/E_CH
J2meV = 1e3/E_CH
J2eV  = 1/E_CH
J2keV = 1e-3/E_CH
J2MeV = 1e-6/E_CH
J2GeV = 1e-9/E_CH
J2GHz = 1.509e24
ueV2GHz = 0.2417684
GHz2ueV = 1/ueV2GHz

# ---------- Luttinger-Kohn Parameters
@dataclass
class LKParams:
    gamma1: float = 13.38
    gamma2: float = 4.24
    gamma3: float = 5.69
    kappa:  float = 3.41
    q:      float = 0.067
    gamma_h1: float = 2.5
    eta_h1:   float = 0.41

    # Derived — do not set these directly
    gz_HH:    float = field(init=False)
    gxy_HH:   float = field(init=False)
    mH_z:     float = field(init=False)
    mH_xy:    float = field(init=False)
    mL_z:     float = field(init=False)
    mL_xy:    float = field(init=False)
    m_eff:    float = field(init=False)
    mLH_ratio: float = field(init=False)

    def __post_init__(self):
        self.gz_HH     = 6*self.kappa + 13.5*self.q
        self.gxy_HH    = 3*self.q
        self.mH_z      = M0 / (self.gamma1 - 2*self.gamma2)
        self.mH_xy     = M0 / (self.gamma1 + self.gamma2 - self.gamma_h1)
        self.mL_z      = M0 / (self.gamma1 + 2*self.gamma2)
        self.mL_xy     = M0 / (self.gamma1 - self.gamma2 - self.gamma_h1)
        self.m_eff     = self.mH_xy * (self.gamma1 + self.gamma2) / (self.gamma1 + self.gamma2 - self.gamma_h1)
        self.mLH_ratio = (self.mL_xy / self.mH_xy) ** (1/4)

# ---------- Strain Parameters
@dataclass
class StrainParams:
    a_eV:      float = 2.0       # Hydrostatic deformation potential (eV)
    b_eV:      float = -2.16     # Uniaxial deformation potential (eV)
    d_eV:      float = -6.06     # Shear deformation potential (eV)
    # Biaxial strain components for Ge on Si0.2Ge0.8 (001).
    # Off-diagonal (shear) components assumed zero.
    exx:       float = -6.3e-3   # In-plane strain (compressive)
    eyy:       float = -6.3e-3   # In-plane strain (compressive)
    ezz:       float =  4.4e-3   # Out-of-plane strain (tensile, via Poisson)
    # Derived — do not set these directly
    a_strain:  float = field(init=False)
    b_strain:  float = field(init=False)
    d_strain:  float = field(init=False)
    eps_strain: np.ndarray = field(init=False)

    def __post_init__(self):
        self.a_strain   = self.a_eV * E_CH
        self.b_strain   = self.b_eV * E_CH
        self.d_strain   = self.d_eV * E_CH
        self.eps_strain = np.array([[self.exx, 0.0,      0.0     ],
                                    [0.0,      self.eyy, 0.0     ],
                                    [0.0,      0.0,      self.ezz]])


# ---------- Heterostructure Parameters
@dataclass
class HeterostructureParams:
    # Heterostructure parameters
    di:  float = 60e-9   # distance from oxide layer to well
    dox: float = 5e-9    # oxide layer width
    dw:  float = 18e-9   # well depth

# ---------- Confinement Parameters
@dataclass
class ConfinementParams:
    axH: float = 50e-9
    ayH: float = 50e-9
    axL: float = None
    ayL: float = None
    UH: float = 150 / J2meV
    UL: float = 100 / J2meV

# -------- Collective Parameters
class GeQDParams():
    def __init__(self, lk: LKParams, st: StrainParams, ht: HeterostructureParams, cn: ConfinementParams):
        self.lk = lk
        self.st = st
        self.ht = ht
        self.cn = cn
        # Derive LH lengths from HH if not explicitly set
        if self.cn.axL is None:
            self.cn.axL = self.cn.axH / self.lk.mLH_ratio
        if self.cn.ayL is None:
            self.cn.ayL = self.cn.ayH / self.lk.mLH_ratio

# -------- External Fields (Electric, Magnetic)
class FieldParams():
    def __init__(self, Fz, B):
        self.Fz = Fz
        self.B  = B