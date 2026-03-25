# ge-qd-lkbp

Numerical simulation of hole spin qubits in planar Ge/SiGe quantum dots using the four-band Luttinger-Kohn-Bir-Pikus (LKBP) model.

Solves for single-hole eigenstates and eigenenergies in a realistic SiGe/Ge/SiGe heterostructure, including heavy-hole/light-hole mixing, spin-orbit coupling, anisotropic in-plane confinement, vertical electric field, biaxial strain, and an arbitrary external magnetic field.

## Physics

The Hamiltonian is assembled in the four-band basis [HH↑, HH↓, LH↑, LH↓] with basis states

|n, m, l⟩ = ψ_n(x) ψ_m(y) φ_l(z)

where ψ_n/m are anisotropic harmonic oscillator (Fock-Darwin) eigenstates and φ_l are piecewise Airy functions matched across the heterostructure interfaces. The five contributions to the Hamiltonian are:

- **Luttinger-Kohn** kinetic energy with Peierls substitution k → k + (e/ℏ) A
- **Bir-Pikus** strain Hamiltonian for biaxial epitaxial strain
- **Zeeman** interaction for J = 3/2 holes (κ, q parameters)
- **Confinement** — in-plane harmonic potential + vertical electric field F_z
- **Heterostructure well** — finite rectangular well potential

## Module structure

```
model_params.py     — Physical parameters (LKParams, StrainParams,
                      HeterostructureParams, ConfinementParams, GeQDParams, FieldParams)
z_subbands.py       — 1D Schrödinger equation solver; piecewise Airy z-subbands
orbital_basis.py    — HarmonicOscillator and OrbitalBasis (3D matrix elements)
hamiltonian.py      — LKBPHamiltonian assembly; mixed HH↔LH orbital operators
visualizations.py   — Wavefunction density plots
```

Dependencies flow as:

```
model_params  →  z_subbands  →  orbital_basis  →  hamiltonian
```

## Installation

```bash
git clone https://github.com/<your-username>/ge-qd-lkbp.git
cd ge-qd-lkbp
pip install -r requirements.txt
```

### Data files

Two precomputed data files are required at runtime:

| File | Used in | Description |
|---|---|---|
| `z_wavefncs_coeffs.pk` | `z_subbands.py` | Airy z-subband coefficients for a grid of F_z values |
| `ho_mixed_matelems.p` | `hamiltonian.py` | Analytic HO mixed matrix elements (sympy expressions) |

Set the environment variable `GE_QD_DATA_DIR` to the directory containing these files:

```bash
export GE_QD_DATA_DIR=/path/to/data
```

To regenerate `z_wavefncs_coeffs.pk` from scratch, instantiate `ZSubbands` with `compute_basis=True`.

## Usage

```python
import numpy as np
from model_params import LKParams, StrainParams, HeterostructureParams, ConfinementParams, GeQDParams, FieldParams
from orbital_basis import OrbitalBasis
from hamiltonian import LKBPHamiltonian

# --- Define parameters
gp = GeQDParams(LKParams(), StrainParams(), HeterostructureParams(), ConfinementParams())
fp = FieldParams(Fz=1.5e6, B=[1.0, 0.0, 1e-3])  # Fz in V/m, B in Tesla

# --- Build orbital bases (s_max shells, l_max_H HH z-subbands, l_max_L LH z-subbands)
obH = OrbitalBasis(s_max=4, l_max_H=4, l_max_L=10, gp=gp, fp=fp, holetype='H')
obL = OrbitalBasis(s_max=4, l_max_H=4, l_max_L=10, gp=gp, fp=fp, holetype='L')

# --- Assemble and diagonalize Hamiltonian
ham = LKBPHamiltonian(gp, fp, obH, obL)
H = ham.assemble_H()
energies, states = ham.get_eigh(H, units='meV')

print("Lowest 6 energies (meV):", energies[:6])
print("Qubit gap (meV):", energies[1] - energies[0])
```

## Known limitations and future work

- [ ] `sys.path.append('Code')` in all modules should be replaced with a proper package structure (`__init__.py` + `pip install -e .`)
- [ ] `ho_mixed_matelems.p` regeneration script not yet included — currently requires running a separate sympy notebook, that needs to be uploaded
- [ ] No unit tests; a `tests/` directory with at least orthonormality and Hermiticity checks would be valuable
- [ ] Example notebook (`examples/single_qubit.ipynb`) not yet included
- [ ] `z_wavefncs_coeffs.pk` for a standard set of F_z values and its generating code should be deposited on Zenodo with a download script

## References

- Wang et al., *npj Quantum Information* **10**, 102 (2024) — heterostructure model and sweet spots
- Burkard et al., *PRR* **2**, 043180 (2020) — Airy z-subband basis
- Luttinger & Kohn, *Phys. Rev.* **97**, 869 (1955) — k·p Hamiltonian
- Bir & Pikus, *Symmetry and Strain-Induced Effects in Semiconductors* (1974) — strain Hamiltonian
