"""
3D orbital operator basis for HH and LH holes in a Ge/SiGe quantum dot.

Provides:
  - HarmonicOscillator: analytic 1D HO matrix elements <n|x|m>, <n|k|m>, etc.
  - OrbitalBasis: full 3D basis built from anisotropic in-plane HO states (shells
    indexed by energy shell s = n+m) tensored with Airy z-subbands (ZSubbands).
    Computes position, momentum, and canonical momentum operators including the
    Peierls substitution k -> k + (e/hbar) A for an arbitrary magnetic field B.
"""
import sys
sys.path.append('Code')
from model_params import *
from z_subbands import *
import math

# ----------- Some useful functions
def hermitize(A): 
    return 0.5*(A + A.conj().T)

def anticomm(A,B): 
    return A@B + B@A

def commutator(A,B): 
    return A@B - B@A

def symprod(A,B): 
    return 0.5*anticomm(A,B)

# ======================================================================
#                      Harmonic Oscillator Basis
# ======================================================================

class HarmonicOscillator():
	"""
	1D quantum harmonic oscillator with optional Fock-Darwin correction for Bz != 0.

	Provides analytic matrix elements <n|x|m>, <n|k|m>, <n|x²|m>, <n|k²|m>
	used to build the in-plane orbital operators in OrbitalBasis.
	"""
	def __init__(self, m0, a0, Bz=0.):
		self.m0 = m0
		self.a0 = a0
		self.Bz = Bz
		self.omega = HBAR / ( self.m0 * self.a0**2) # rad/s
	    # Rescale HO length if Bz!=0
		if abs(self.Bz)>0:
			self.a0 = self.ho_eff_len()

	def ho_len(self):
	    """Oscillator length x0 = sqrt(ħ/(m ω)). Utility method, not used internally."""
	    return np.sqrt(HBAR/(self.m0 * self.omega))

	def ho_psi(self, n, x, x0):
	    # ψ_n(x) = [1/(√(2^n n!) π^1/4 √x0)] H_n(x/x0) e^{-(x/x0)^2/2}
	    xi = x/x0
	    # Physicists' Hermite via hermval; H_n(x) = hermval(x, [0,...,1]) with length n+1
	    coeffs = np.zeros(n+1); coeffs[-1] = 1.0
	    Hn = np.polynomial.hermite.hermval(xi, coeffs)
	    norm = 1.0 / ( (2.0**n * math.factorial(n))**0.5 * (PI**0.25) * (x0**0.5) )
	    return norm * Hn * np.exp(-0.5*xi**2)

	def ho_omega_eff(self):
	    """Fock-Darwin effective frequency. Utility method, not used internally."""
	    omega_c = abs(E_CH * self.Bz) / self.m0
	    omega_eff = np.sqrt(self.omega**2 + (omega_c**2)/4)
	    return omega_eff

	def ho_eff_len(self):
		wc = E_CH*np.abs(self.Bz)/self.m0
		a0_eff = np.sqrt(HBAR/E_CH/np.abs(self.Bz)) / (1/4+self.omega**2/wc**2)**0.25
		return a0_eff

	### 1D analytic matrix elements
	def get_op_x(self, n, m):  # <n|x|m>
	    if n == m+1: return self.a0 * np.sqrt((m+1)/2)
	    if n == m-1: return self.a0 * np.sqrt(m/2)
	    return 0.0

	def get_op_k(self, n, m):  # <n|k|m>
	    if n == m+1: return 1j*np.sqrt((m+1)/2) / self.a0
	    if n == m-1: return -1j*np.sqrt(m/2) / self.a0
	    return 0.0

	def get_op_x2(self, n, m):  # <n|x@x|m>
	    if n == m:   return self.a0**2 * (2*m+1)/2
	    if n == m+2: return self.a0**2 * np.sqrt((m+1)*(m+2))/2
	    if n == m-2: return self.a0**2 * np.sqrt(m*(m-1))/2
	    return 0.0

	def get_op_k2(self, n, m):  # <n|kx@kx|m>
	    if n == m:   return (2*m+1)/2 / self.a0**2
	    if n == m+2: return -np.sqrt((m+1)*(m+2))/2 / self.a0**2
	    if n == m-2: return -np.sqrt(m*(m-1))/2 / self.a0**2
	    return 0.0

# ======================================================================
#                      3D Orbital Operator Basis
# ======================================================================

class OrbitalBasis():
	"""
	3D orbital basis for a single hole band (HH or LH) in a Ge/SiGe quantum dot.

	Basis states are |n, m, l> = ψ_n(x) ψ_m(y) φ_l(z), where ψ_n/m are
	anisotropic harmonic oscillator eigenstates and φ_l are Airy z-subbands.
	States are grouped into energy shells s = n + m (0, 1, 2, ..., s_max-1).
	All orbital operators (position, momentum, canonical momentum) are built
	analytically and stored in self.orb_ops.
	"""
	def __init__(self, s_max, l_max_H, l_max_L, gp:GeQDParams, fp:FieldParams, holetype='H',
				compute_z_basis=False, verbose=False, symmetrize_kz=False, nz=2000):
		self.m0 = gp.lk.mH_xy if holetype=='H' else gp.lk.mL_xy
		self.Lx = gp.cn.axH if holetype=='H' else gp.cn.axL
		self.Ly = gp.cn.ayH if holetype=='H' else gp.cn.ayL

		self.s_max = s_max
		self.l_max_H = l_max_H
		self.l_max_L = l_max_L
		self.l_max = l_max_H if holetype=='H' else l_max_L

		self.gp = gp
		self.fp = fp
		self.B  = fp.B

		self.holetype = holetype

		self.verbose = verbose
		self.symmetrize_kz = symmetrize_kz
		self.compute_z_basis = compute_z_basis
		self.nz = nz
		if self.verbose: print("Computing orbital operators.")
		self.states = self.build_shell_states()
		self.orb_ops = self.build_ops_shell()

	### Indexes orbital states based on energy shells
	def ho_shell_pairs(self):
	    out = []
	    for s in range(self.s_max):
	        for n in range(s+1):
	            out.append((n, s-n))
	    return out

	def build_shell_states(self):
	    nm = self.ho_shell_pairs()
	    self.states = [(n,m,l) for (n,m) in nm for l in range(self.l_max)]
	    return self.states

	### Vector potential A in several gauges
	def get_vector_potential(self, ops):
		Bx,By,Bz = self.B
		X,Y,Z,X2,Y2,Z2 = ops

		Ax = 0.5*(2*By*Z - Bz*Y)
		Ay = 0.5*(Bz*X - 2*Bx*Z)
		Az = np.zeros_like(X)

		Ax2 = 0.25*(4 * By**2 * Z2 + Bz**2 * Y2 - 2*By*Bz*(Y@Z + Z@Y))
		Ay2 = 0.25*(Bz**2 * X2 + 4 * Bx**2 * Z2 - 2*Bz*Bx*(X@Z + Z@X))
		Az2 = np.zeros_like(X)

		return Ax, Ay, Az, Ax2, Ay2, Az2

	### Build operators
	def build_ops_shell(self, inplane_shift=(0,0)):
		N = len(self.states)
		Iorb = np.eye(N)
		X  = np.zeros((N,N), float);  Y  = np.zeros((N,N), float);  Z  = np.zeros((N,N), float)
		X2 = np.zeros((N,N), float);  Y2 = np.zeros((N,N), float);  Z2 = np.zeros((N,N), float)
		Kx = np.zeros((N,N), complex);  Ky = np.zeros((N,N), complex);  Kz = np.zeros((N,N), complex)
		Kx2= np.zeros((N,N), float);  Ky2= np.zeros((N,N), float);  Kz2= np.zeros((N,N), float)
		ZKz = np.zeros((N,N), complex);  KzZ= np.zeros((N,N), complex);

		Bx,By,Bz = self.B
		# Get Harmonic Oscillator objects for x and y
		self.hox = HarmonicOscillator(self.m0, self.Lx, Bz)
		self.hoy = HarmonicOscillator(self.m0, self.Ly, Bz)
		self.zsb = ZSubbands(self.gp, self.fp, self.l_max_H, self.l_max_L, self.holetype, compute_basis=self.compute_z_basis, \
					verbose=self.verbose, symmetrize_kz=self.symmetrize_kz, nz=self.nz)

		for i,(ni,mi,li) in enumerate(self.states):
			for j,(nj,mj,lj) in enumerate(self.states):
				if (mi==mj) and (li==lj):
					X[i,j]  = self.hox.get_op_x(ni,nj)
					Kx[i,j] = self.hox.get_op_k(ni,nj)
					X2[i,j]  = self.hox.get_op_x2(ni,nj)
					Kx2[i,j] = self.hox.get_op_k2(ni,nj)
				if (ni==nj) and (li==lj):
					Y[i,j]  = self.hoy.get_op_x(mi,mj)
					Ky[i,j] = self.hoy.get_op_k(mi,mj)
					Y2[i,j]  = self.hoy.get_op_x2(mi,mj)
					Ky2[i,j] = self.hoy.get_op_k2(mi,mj)
				if (ni==nj) and (mi==mj):
					Z[i,j]  = self.zsb.get_op_z(li,lj) 
					Kz[i,j] = self.zsb.get_op_kz(li,lj) 
					Kz2[i,j] = self.zsb.get_op_kz2(li,lj) 
					Z2[i,j]  = self.zsb.get_op_z2(li,lj) 
					ZKz[i,j] = self.zsb.get_op_zkz(li,lj) 
					KzZ[i,j] = self.zsb.get_op_kzz(li,lj)

		# Implement in-plane shift if necessary
		x0,y0 = inplane_shift
		if (x0,y0) != (0,0):
		    X = X + x0*Iorb
		    Y = Y + y0*Iorb
		    X2 = X2 + 2*x0*X + (x0**2)*Iorb
		    Y2 = Y2 + 2*y0*Y + (y0**2)*Iorb

		# Peierls substitution: canonical momenta C = k + (e/hbar) A, symmetric gauge
		Ax,Ay,Az,Ax2,Ay2,Az2 = self.get_vector_potential(ops=(X,Y,Z,X2,Y2,Z2))
		alpha = - E_CH/HBAR

		# Canonical Momenta
		Cx = Kx + alpha*Ax
		Cy = Ky + alpha*Ay
		Cz = Kz + alpha*Az

		# minimal coupling: (k + αA)^2 = k^2 + 2 α{ k, A } + α^2 A^2
		Cx2 = Kx2 + alpha*anticomm(Kx,Ax) + (alpha**2)*Ax2
		Cy2 = Ky2 + alpha*anticomm(Ky,Ay) + (alpha**2)*Ay2
		Cz2 = Kz2 + alpha*anticomm(Kz,Az) + (alpha**2)*Az2

		# Anticommutators 
		KxKy = symprod(Cx,Cy)
		KzAy = Bz/2*symprod(Kz,X) - Bx*(KzZ+ZKz)/2
		KyKz = symprod(Ky,Kz) + alpha * KzAy
		KzAx = - Bz/2*symprod(Kz,Y) + By*(KzZ+ZKz)/2
		KxKz = symprod(Kx,Kz) + alpha * KzAx

		# Enforce Hermiticity
		Cx, Cy, Cz = hermitize(Cx), hermitize(Cy), hermitize(Cz)
		Cx2, Cy2, Cz2 = hermitize(Cx2), hermitize(Cy2), hermitize(Cz2)
		KxKy, KyKz, KxKz = hermitize(KxKy), hermitize(KyKz), hermitize(KxKz)

		self.orb_ops = dict(Iorb=Iorb,X=X,Y=Y,Z=Z,X2=X2,Y2=Y2,Z2=Z2,Kx=Kx,Ky=Ky,Kz=Kz,Cx=Cx,Cy=Cy,Cz=Cz,
		    	        Kx2=Kx2,Ky2=Ky2,Kz2=Kz2,Cx2=Cx2,Cy2=Cy2,Cz2=Cz2,KxKy=KxKy,KyKz=KyKz,KxKz=KxKz)

		return self.orb_ops