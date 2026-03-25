"""
Luttinger-Kohn-Bir-Pikus (LKBP) Hamiltonian for a four-band (HH/LH) hole
in a Ge/SiGe quantum dot.

Assembles the full Hamiltonian in the basis [HH+, HH-, LH+, LH-] with
block sizes [Nh, Nh, Nl, Nl], where Nh and Nl are the number of HH and LH
orbital basis states respectively (from OrbitalBasis). Includes:
  - Luttinger-Kohn kinetic energy with Peierls substitution
  - Bir-Pikus strain Hamiltonian
  - Zeeman interaction (J=3/2)
  - In-plane harmonic + vertical electric field confinement
  - Finite heterostructure well potential

Mixed HH<->LH orbital operators (rectangular, Nh x Nl) are built by
build_mixed_orbital_ops() and its helpers _ho_mixed_tables(), _z_mixed_tables().
"""
import sys
sys.path.append('Code')
from model_params import *
from z_subbands import *
from orbital_basis import *
import pickle as pk
from sympy import symbols 

# ---------- Spin (J=3/2) and Identity
def J_mats():
    s3 = np.sqrt(3.0)
    Jx = 0.5*np.array([[0, 0, s3,0],
                       [0, 0, 0,s3],
                       [s3,0, 0, 2],
                       [0,s3, 2, 0]], complex)
    Jy = 0.5j*np.array([[0, 0,-s3,0],
                        [0, 0, 0, s3],
                        [s3,0, 0,-2],
                        [0,-s3,2, 0]], complex)
    Jz = 0.5*np.array([[3, 0, 0, 0],
                       [0,-3, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0,-1]], complex)
    return Jx,Jy,Jz

I4 = np.eye(4)

def expect_J(vec, Nh, Nl, J4, I_HL):
    """
    Compute <J> for a 4-band eigenvector with unequal orbital spaces.
    J4 is 4x4 (e.g. Jx, Jy, Jz).
    I_HL is Nh x Nl overlap; I_LH = I_HL^†
    """
    I_H  = np.eye(Nh, dtype=complex)
    I_L  = np.eye(Nl, dtype=complex)
    I_LH = I_HL.conj().T

    def orb_id(a, b):
        aH = (a < 2)
        bH = (b < 2)
        if aH and bH:   return I_H
        if (not aH) and (not bH): return I_L
        if aH and (not bH): return I_HL
        return I_LH

    blocks = [[None]*4 for _ in range(4)]
    for a in range(4):
        for b in range(4):
            blocks[a][b] = J4[a,b] * orb_id(a,b)

    J_big = np.block(blocks)
    return np.vdot(vec, J_big @ vec).real

# Directory for saved z wavefunctions
import platform
dir2file_ho = r"C:\\Users\\14432\\UMBC\\Research\\Code\\Ge_pO\\Code\\" if platform.system() == "Windows" else "/home/yoda1/Ge/"


### Build Hamiltonian
class LKBPHamiltonian():
	"""
	Four-band LKBP Hamiltonian for a single hole in a Ge/SiGe quantum dot.

	Takes pre-built HH and LH OrbitalBasis objects and assembles the full
	Hamiltonian matrix. Call assemble_H() to get the complete matrix, or
	call individual get_H_*() methods to inspect specific contributions.
	"""
	def __init__(self, gp:GeQDParams, fp:FieldParams, obH:OrbitalBasis, obL:OrbitalBasis):
		self.gp = gp
		self.fp = fp
		self.obH = obH
		self.obL = obL

		self.B  = fp.B
		self.states_H = obH.states
		self.states_L = obL.states
		self.Norb_H = len(self.states_H)
		self.Norb_L = len(self.states_L)
		self.opsH = obH.orb_ops
		self.opsL = obL.orb_ops

		self.ax_H = obH.hox.a0
		self.ay_H = obH.hoy.a0
		self.omega_x_H = obH.hox.omega
		self.omega_y_H = obH.hoy.omega

		self.ax_L = obL.hox.a0
		self.ay_L = obL.hoy.a0
		self.omega_x_L = obL.hox.omega
		self.omega_y_L = obL.hoy.omega

		self.symmetrize_kz = obH.symmetrize_kz

		if self.obH.verbose: print("Computing mixed orbital operators.")
		self.opsHL = self.build_mixed_orbital_ops(obH, obL)

	def get_H_zeeman(self):
		"""
		Zeeman Hamiltonian for J=3/2 holes:
		  H_Z = -2 μ_B [ κ (B·J) + q (Bx Jx^3 + By Jy^3 + Bz Jz^3) ]
		Embedded into full basis with block sizes [Nh, Nh, Nl, Nl].
		"""
		Bx, By, Bz = self.B
		Jx, Jy, Jz = J_mats()

		gp = self.gp
		HZ4 = - 2 * MU_B * ( gp.lk.kappa * (Bx*Jx + By*Jy + Bz*Jz)
		    + gp.lk.q * (Bx*(Jx@Jx@Jx) + By*(Jy@Jy@Jy) + Bz*(Jz@Jz@Jz)))

		Nh, Nl = self.Norb_H, self.Norb_L
		I_H  = np.eye(Nh, dtype=complex)
		I_L  = np.eye(Nl, dtype=complex)

		# This is the correct "identity operator" between different orbital bases
		I_HL = self.opsHL["Iorb"]           # Nh x Nl
		I_LH = self.opsHL["Iorb"].conj().T  # Nl x Nh

		Z_HH = np.zeros((Nh, Nh), dtype=complex)
		Z_LL = np.zeros((Nl, Nl), dtype=complex)

		# Embed HZ4_{ab} ⊗ I_{orbital}(a,b)
		# with basis order [HH+, HH-, LH+, LH-] and sizes [Nh, Nh, Nl, Nl]
		def embed(a, b):
		    """Return the orbital factor for band indices a,b in {0,1,2,3}."""
		    a_is_H = (a < 2)
		    b_is_H = (b < 2)
		    if a_is_H and b_is_H:
		        return I_H
		    if (not a_is_H) and (not b_is_H):
		        return I_L
		    if a_is_H and (not b_is_H):
		        return I_HL
		    return I_LH  # (LH -> HH) i.e. rows in L, cols in H

		blocks = [[None]*4 for _ in range(4)]
		for a in range(4):
		    for b in range(4):
		        orb = embed(a, b)
		        blocks[a][b] = HZ4[a, b] * orb

		H = np.block(blocks)
		return H


	# ---------- Scalar confinement + vertical field + strain (Bir–Pikus)
	def get_V_well(self):
		"""
		Vertical confinement from heterostructure (finite well):
		  V_well(z) = -U * 1_{-di <= z <= 0}
		implemented as a projector/operator in each sector's z basis, then
		lifted to the full (n,m,l) orbital basis.

		Returns full matrix of size (2*Nh + 2*Nl) x (2*Nh + 2*Nl),
		with block sizes [Nh, Nh, Nl, Nl] in (HH+, HH-, LH+, LH-) order.
		"""

		def build_W_orb(ob):
		    """
		    Build W_orb (Norb x Norb) for a given OrbitalBasis ob using its z basis.
		    """
		    zz = ob.zsb.zz
		    dz = ob.zsb.dz
		    Phi = np.array(ob.zsb.z_basis)  # shape (lmax, nz)
		    lmax = Phi.shape[0]

		    # indicator function of the well region
		    dw = self.gp.ht.dw
		    Uz = ((zz >= -dw) & (zz <= 0.0)).astype(float)

		    # W_lm = ∫ φ_l(z) φ_m(z) Uz(z) dz
		    # (Phi are real in your construction; keep complex-safe anyway)
		    W_lm = np.zeros((lmax, lmax), dtype=complex)
		    for l in range(lmax):
		        for m in range(lmax):
		            W_lm[l, m] = np.sum(np.conjugate(Phi[l]) * Phi[m] * Uz) * dz

		    # Lift to full orbital basis: couples only same (n,m)
		    states = ob.states
		    Norb = len(states)
		    W_orb = np.zeros((Norb, Norb), dtype=complex)
		    for i, (ni, mi, li) in enumerate(states):
		        for j, (nj, mj, lj) in enumerate(states):
		            if (ni == nj) and (mi == mj):
		                W_orb[i, j] = W_lm[li, lj]
		    return W_orb

		# Build HH and LH well operators in their own orbital spaces
		W_H = build_W_orb(self.obH)  # Nh x Nh
		W_L = build_W_orb(self.obL)  # Nl x Nl

		Nh, Nl = self.Norb_H, self.Norb_L
		Z_HH = np.zeros((Nh, Nh), dtype=complex)
		Z_LL = np.zeros((Nl, Nl), dtype=complex)
		Z_HL = np.zeros((Nh, Nl), dtype=complex)
		Z_LH = np.zeros((Nl, Nh), dtype=complex)

		UH = self.gp.cn.UH
		UL = self.gp.cn.UL

		# Potential is -U inside well (your convention)
		V_H = -UH * W_H
		V_L = -UL * W_L

		# Embed into full basis: band-diagonal (same for + and - within each sector)
		V = np.block([
		    [V_H,  Z_HH, Z_HL, Z_HL],
		    [Z_HH, V_H,  Z_HL, Z_HL],
		    [Z_LH, Z_LH, V_L,  Z_LL],
		    [Z_LH, Z_LH, Z_LL, V_L ],
		])
		return V


	def get_H_conf_Fz(self):
		"""
		Confinement potential:
		  Vxy = (1/2) m_eff (ωx^2 x^2 + ωy^2 y^2)
		  Vz  = -e Fz z  
		Embedded into full 4-band basis with unequal HH/LH orbital dimensions.
		"""
		Nh, Nl = self.Norb_H, self.Norb_L
		Z_HH = np.zeros((Nh, Nh), dtype=complex)
		Z_LL = np.zeros((Nl, Nl), dtype=complex)
		Z_HL = np.zeros((Nh, Nl), dtype=complex)
		Z_LH = np.zeros((Nl, Nh), dtype=complex)

		# --- HH orbital operators ---
		Vxy_H = 0.5 * self.gp.lk.mH_xy * (
		    (self.omega_x_H**2) * self.opsH["X2"] + (self.omega_y_H**2) * self.opsH["Y2"]
		)
		Vz_H  = - E_CH * self.fp.Fz * self.opsH["Z"]
		V_H   = Vxy_H + Vz_H  # Nh x Nh

		# --- LH orbital operators ---
		Vxy_L = 0.5 * self.gp.lk.mL_xy * (
		    (self.omega_x_L**2) * self.opsL["X2"] + (self.omega_y_L**2) * self.opsL["Y2"]
		)
		Vz_L  = - E_CH * self.fp.Fz * self.opsL["Z"]
		V_L   = Vxy_L + Vz_L  # Nl x Nl

		# Embed as band-diagonal (same potential for HH± and LH±)
		H = np.block([
		    [V_H,  Z_HH, Z_HL, Z_HL],
		    [Z_HH, V_H,  Z_HL, Z_HL],
		    [Z_LH, Z_LH, V_L,  Z_LL],
		    [Z_LH, Z_LH, Z_LL, V_L ],
		])
		return H

	# Strain
	def get_H_strain(self):
		if self.gp.st.eps_strain is None: return 0.
		eps = self.gp.st.eps_strain

		a, b, d = self.gp.st.a_strain, self.gp.st.b_strain, self.gp.st.d_strain

		P = -a * np.trace(eps)
		Q = -0.5 * b * (eps[0,0] + eps[1,1] - 2*eps[2,2])
		L = d * (eps[0,2] - 1j*eps[1,2])
		M = (np.sqrt(3)/2) * b * (eps[0,0] - eps[1,1]) - d * eps[0,1]

		Nh, Nl = self.Norb_H, self.Norb_L
		I_H = np.eye(Nh, dtype=complex)
		I_L = np.eye(Nl, dtype=complex)

		Z_HH = np.zeros((Nh, Nh), dtype=complex)
		Z_LL = np.zeros((Nl, Nl), dtype=complex)
		Z_HL = np.zeros((Nh, Nl), dtype=complex)
		Z_LH = np.zeros((Nl, Nh), dtype=complex)

		# Scalars times "identity map" between spaces:
		# For a *spatially uniform* strain Hamiltonian, the orbital operator is identity.
		# Mixed blocks should therefore use the HH<->LH overlap map Iorb (Nh x Nl),
		# not a rectangular "identity". You already compute opsHL["Iorb"].
		I_HL = self.opsHL["Iorb"]          # Nh x Nl
		I_LH = self.opsHL["Iorb"].conj().T # Nl x Nh

		H = np.block([
		    [(P+Q)*I_H,  Z_HH,        L*I_HL,        M*I_HL],
		    [Z_HH,       (P+Q)*I_H,   np.conj(M)*I_HL, -np.conj(L)*I_HL],
		    [np.conj(L)*I_LH,  M*I_LH, (P-Q)*I_L,     Z_LL],
		    [np.conj(M)*I_LH, -L*I_LH, Z_LL,          (P-Q)*I_L],
		])
		return H


	# ---------- LK blocks
	def get_LK_blocks(self):
		gp = self.gp

		# Diagonal HH blocks (Nh x Nh)
		P_H = (HBAR**2) / (2*M0) * gp.lk.gamma1 * (self.opsH["Cx2"] + self.opsH["Cy2"] + self.opsH["Cz2"])
		Q_H = (HBAR**2) / (2*M0) * gp.lk.gamma2 * (self.opsH["Cx2"] + self.opsH["Cy2"] - 2*self.opsH["Cz2"])

		# Diagonal LH blocks (Nl x Nl)
		P_L = (HBAR**2) / (2*M0) * gp.lk.gamma1 * (self.opsL["Cx2"] + self.opsL["Cy2"] + self.opsL["Cz2"])
		Q_L = (HBAR**2) / (2*M0) * gp.lk.gamma2 * (self.opsL["Cx2"] + self.opsL["Cy2"] - 2*self.opsL["Cz2"])

		# Off-diagonal HL mixing blocks (Nh x Nl)
		# --- R^{HL} (Nh x Nl)
		R_HL     = np.sqrt(3) * (HBAR**2 / (2*M0)) * ( - gp.lk.gamma2 * (self.opsHL["Cx2"] - self.opsHL["Cy2"])
		    						    		  + 2j * gp.lk.gamma3 *  self.opsHL["KxKy"] )
		R_HL_dag = np.sqrt(3) * (HBAR**2 / (2*M0)) * ( - gp.lk.gamma2 * (self.opsHL["Cx2"] - self.opsHL["Cy2"])
		    						    		  - 2j * gp.lk.gamma3 *  self.opsHL["KxKy"] )
		# --- S^{HL} (Nh x Nl)
		S_HL     = - np.sqrt(3) * (HBAR**2 / M0) * gp.lk.gamma3 * (self.opsHL["KxKz"] - 1j*self.opsHL["KyKz"])
		S_HL_dag = - np.sqrt(3) * (HBAR**2 / M0) * gp.lk.gamma3 * (self.opsHL["KxKz"] + 1j*self.opsHL["KyKz"])

		return P_H, Q_H, P_L, Q_L, R_HL, R_HL_dag, S_HL, S_HL_dag

	# Assemble the LK Hamiltonian by blocks
	def get_H_LK(self):
		# Get matrix blocks
		P_H, Q_H, P_L, Q_L, R_HL, R_HL_dag, S_HL, S_HL_dag = self.get_LK_blocks()

		Z_HH = np.zeros_like(P_H)
		Z_LL = np.zeros_like(P_L)

		# rows/cols: [HH+, HH-, LH+, LH-] with sizes [Nh, Nh, Nl, Nl]
		H = np.block([
		    [P_H + Q_H,        Z_HH,                  S_HL,          R_HL],
		    [Z_HH,             P_H + Q_H,             R_HL_dag,     -S_HL_dag],
		    [S_HL.conj().T,    R_HL_dag.conj().T,     P_L - Q_L,     Z_LL],
		    [R_HL.conj().T,   -S_HL_dag.conj().T,     Z_LL,          P_L - Q_L],
		])

		return H


	# ---------- Total Hamiltonian -----------------
	def assemble_H(self):
		if self.obH.verbose: print("Generating mixed basis Hamiltonians.")
		# Get Hamiltonians
		H_LK   = self.get_H_LK()
		H_BP   = self.get_H_strain()
		H_Z    = self.get_H_zeeman()
		H_conf = self.get_H_conf_Fz()
		V_well = self.get_V_well()

		# Put everything together
		if self.obH.verbose: print("Assembling LKBP Hamiltonian.")
		self.H = H_LK + H_BP + H_conf + H_Z + V_well

		return self.H

	# ---------- Utilities --------------------
	def get_eigh(self, H, units='meV'):
	    e, v = np.linalg.eigh(H)
	    if units=='eV':
	        e_units = e*J2eV
	    elif units=='meV':
	        e_units = e*J2meV
	    elif units=='ueV':
	        e_units = e*J2ueV
	    elif units=='neV':
	        e_units = e*J2neV
	    idx_sort = np.argsort(e_units)
	    return e_units[idx_sort], v[:,idx_sort]

	def get_all_states(self, latex=False):
		self.all_states = []
		NH = len(self.states_H)
		NL = len(self.states_L)
		for i in range((NH+NL)*2):
		    if i<NH:
		        self.all_states += [list(self.states_H[i])+['$+$H' if latex else '+H']]
		    if NH<=i<2*NH:
		        self.all_states += [list(self.states_H[i-NH])+['$-$H' if latex else '-H']]
		    if 2*NH<=i<2*NH+NL:
		        self.all_states += [list(self.states_L[i-2*NH])+['$+$L' if latex else '+L']]
		    if 2*NH+NL<=i:
		        self.all_states += [list(self.states_L[i-2*NH-NL])+['$-$L' if latex else '-L']]
		assert len(self.all_states) == (NH+NL)*2
		return self.all_states

	# ----------------------------
	# Build a full mixed 3D operator matrix between OrbitalBasis obH -> obL
	# ----------------------------
	def build_mixed_orbital_ops(self, obH, obL, npts_xy=4001, pad_xy=5.0):
		"""Thin wrapper delegating to the module-level build_mixed_orbital_ops."""
		return build_mixed_orbital_ops(obH, obL, npts_xy=npts_xy, pad_xy=pad_xy,
		                               symmetrize_kz=self.symmetrize_kz,
		                               dir2file_ho=dir2file_ho, gp=self.gp)

	# Build full 4-band block-diagonal orbital operators
	def build_full_orb_ops_z(self):

		ZERO_HH = np.zeros_like(self.opsH["Z"])
		ZERO_HL = np.zeros_like(self.opsHL["Z"])
		ZERO_LL = np.zeros_like(self.opsL["Z"])
		ZERO_LH = np.zeros_like(self.opsHL["Z"]).T

		# rows/cols: [HH+, HH-, LH+, LH-] with sizes [Nh, Nh, Nl, Nl]
		X = np.block([
		    [self.opsH["X"],  ZERO_HH,         ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,         self.opsH["X"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	  ZERO_LH,  	   self.opsL["X"],  ZERO_LL],
		    [ZERO_LH,    	  ZERO_LH,    	   ZERO_LL,         self.opsL["X"]]
		])

		Y = np.block([
		    [self.opsH["Y"],  ZERO_HH,         ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,         self.opsH["Y"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	  ZERO_LH,  	   self.opsL["Y"],  ZERO_LL],
		    [ZERO_LH,    	  ZERO_LH,    	   ZERO_LL,         self.opsL["Y"]]
		])

		Z = np.block([
		    [self.opsH["Z"],  ZERO_HH,         ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,         self.opsH["Z"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	  ZERO_LH,  	   self.opsL["Z"],  ZERO_LL],
		    [ZERO_LH,    	  ZERO_LH,    	   ZERO_LL,         self.opsL["Z"]]
		])

		Kx = np.block([
		    [self.opsH["Kx"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["Kx"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,          self.opsL["Kx"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["Kx"]]
		])

		Ky = np.block([
		    [self.opsH["Ky"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["Ky"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,          self.opsL["Ky"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["Ky"]]
		])

		Kz = np.block([
		    [self.opsH["Kz"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["Kz"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,          self.opsL["Kz"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["Kz"]]
		])

		X2 = np.block([
		    [self.opsH["X2"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["X2"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,  	     self.opsL["X2"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["X2"]]
		])

		Y2 = np.block([
		    [self.opsH["Y2"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["Y2"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,  	     self.opsL["Y2"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["Y2"]]
		])

		Z2 = np.block([
		    [self.opsH["Z2"],  ZERO_HH,          ZERO_HL,    		ZERO_HL],
		    [ZERO_HH,          self.opsH["Z2"],  ZERO_HL,    		ZERO_HL],
		    [ZERO_LH,    	   ZERO_LH,  	     self.opsL["Z2"],   ZERO_LL],
		    [ZERO_LH,    	   ZERO_LH,    	     ZERO_LL,           self.opsL["Z2"]]
		])

		Kx2 = np.block([
		    [self.opsH["Kx2"],  ZERO_HH,           ZERO_HL,    	    	ZERO_HL],
		    [ZERO_HH,           self.opsH["Kx2"],  ZERO_HL,    		    ZERO_HL],
		    [ZERO_LH,    	    ZERO_LH,           self.opsL["Kx2"],    ZERO_LL],
		    [ZERO_LH,    	    ZERO_LH,    	   ZERO_LL,             self.opsL["Kx2"]]
		])

		Ky2 = np.block([
		    [self.opsH["Ky2"],  ZERO_HH,           ZERO_HL,    	    	ZERO_HL],
		    [ZERO_HH,           self.opsH["Ky2"],  ZERO_HL,    		    ZERO_HL],
		    [ZERO_LH,    	    ZERO_LH,           self.opsL["Ky2"],    ZERO_LL],
		    [ZERO_LH,    	    ZERO_LH,    	   ZERO_LL,             self.opsL["Ky2"]]
		])

		Kz2 = np.block([
		    [self.opsH["Kz2"],  ZERO_HH,           ZERO_HL,    	    	ZERO_HL],
		    [ZERO_HH,           self.opsH["Kz2"],  ZERO_HL,    		    ZERO_HL],
		    [ZERO_LH,    	    ZERO_LH,           self.opsL["Kz2"],    ZERO_LL],
		    [ZERO_LH,    	    ZERO_LH,    	   ZERO_LL,             self.opsL["Kz2"]]
		])

		return X,Y,Z,Kx,Ky,Kz,X2,Y2,Z2,Kx2,Ky2,Kz2

		
# ===========================================================================
# Module-level mixed-basis helper functions (HH <-> LH orbital operators)
# ===========================================================================

def _ho_mixed_tables(hox_H, hox_L, nmax_H, nmax_L, gp, npts=4001, pad=5.0, dir2file_ho=None):
	# x, dx = self._x_grid_for_ho(hox_H.a0, hox_L.a0, npts=npts, pad=pad)

	# psiH = np.stack([hox_H.ho_psi(n, x, hox_H.a0) for n in range(nmax_H + 1)], axis=0)
	# psiL = np.stack([hox_L.ho_psi(n, x, hox_L.a0) for n in range(nmax_L + 1)], axis=0)

	# dpsiL  = np.gradient(psiL, dx, axis=1)
	# d2psiL = np.gradient(dpsiL, dx, axis=1)

	exprs_O,exprs_X,exprs_K,exprs_X2,exprs_K2,exprs_KX,exprs_XK = pk.load(open(dir2file_ho+"ho_mixed_matelems.p",'rb'))

	O  = np.zeros((nmax_H+1, nmax_L+1), dtype=complex)
	X  = np.zeros_like(O); X2 = np.zeros_like(O)
	Kx  = np.zeros_like(O); Kx2 = np.zeros_like(O)
	XK = np.zeros_like(O); KX = np.zeros_like(O)
	Y  = np.zeros_like(O); Y2 = np.zeros_like(O)
	Ky  = np.zeros_like(O); Ky2 = np.zeros_like(O)
	YK = np.zeros_like(O); KY = np.zeros_like(O)

	aH = symbols('a_H', positive=True)
	aL = symbols('a_L', positive=True)
	reps_x = [(aH,gp.cn.axH),(aL,gp.cn.axL)]
	reps_y = [(aH,gp.cn.ayH),(aL,gp.cn.ayL)]
	for nH in range(nmax_H+1):
	    for nL in range(nmax_L+1):
	        # X ops
	        O[nH,nL]  = exprs_O[(nH,nL)].subs(reps_x).evalf()
	        X[nH,nL]  = exprs_X[(nH,nL)].subs(reps_x).evalf()
	        X2[nH,nL] = exprs_X2[(nH,nL)].subs(reps_x).evalf()
	        Kx[nH,nL]  = exprs_K[(nH,nL)].subs(reps_x).evalf()           # <bra|k|ket>
	        Kx2[nH,nL] = exprs_K2[(nH,nL)].subs(reps_x).evalf()         # <bra|k^2|ket>

	        XK[nH,nL] = exprs_XK[(nH,nL)].subs(reps_x).evalf()
	        # kx = -i d(x*ket)/dx = -i(ket + x ket')
	        KX[nH,nL] = exprs_KX[(nH,nL)].subs(reps_x).evalf()
	        # Y ops
	        O[nH,nL]  = exprs_O[(nH,nL)].subs(reps_y).evalf()
	        Y[nH,nL]  = exprs_X[(nH,nL)].subs(reps_y).evalf()
	        Y2[nH,nL] = exprs_X2[(nH,nL)].subs(reps_y).evalf()
	        Ky[nH,nL]  = exprs_K[(nH,nL)].subs(reps_y).evalf()           # <bra|k|ket>
	        Ky2[nH,nL] = exprs_K2[(nH,nL)].subs(reps_y).evalf()         # <bra|k^2|ket>

	        YK[nH,nL] = exprs_XK[(nH,nL)].subs(reps_y).evalf()
	        # kx = -i d(x*ket)/dx = -i(ket + x ket')
	        KY[nH,nL] = exprs_KX[(nH,nL)].subs(reps_y).evalf()
	return dict(O=O, X=X, Y=Y, X2=X2, Y2=Y2, Kx=Kx, Ky=Ky, Kx2=Kx2, Ky2=Ky2, XK=XK, KX=KX, YK=YK, KY=KY)



# ----------------------------
# z mixed tables using your existing Airy basis arrays
# ----------------------------

def _z_mixed_tables(zsb_H, zsb_L, symmetrize_kz=False):
    """
    Mixed z tables:
      Oz[lH,lL]  = <lH|lL>
      Z[lH,lL]   = <lH|z|lL>
      Z2[lH,lL]  = <lH|z^2|lL>
      Kz[lH,lL]  = <lH|kz|lL>     kz = -i d/dz on ket
      Kz2[lH,lL] = <lH|kz^2|lL>   kz^2 = - d^2/dz^2 on ket
      ZKz[lH,lL] = <lH| z kz |lL>
      KzZ[lH,lL] = <lH| kz z |lL>  (use product rule form like your get_op_kzz)
    """
    zz = zsb_H.zz
    dz = zsb_H.dz
    # (important) assume both use same zz grid; in your code zz is built from di,dw,nz
    # If nz differs, you'd need interpolation.

    PhiH = np.array(zsb_H.z_basis)        # (lH, z)
    dPhiH  = np.array(zsb_H.z_basis_der)  # (lL, z)
    PhiL = np.array(zsb_L.z_basis)        # (lL, z)
    dPhiL  = np.array(zsb_L.z_basis_der)  # (lL, z)
    d2PhiL = np.array(zsb_L.z_basis_der2)

    lH_max = PhiH.shape[0]
    lL_max = PhiL.shape[0]

    Oz  = np.zeros((lH_max, lL_max), dtype=complex)
    Z   = np.zeros_like(Oz)
    Z2  = np.zeros_like(Oz)
    Kz  = np.zeros_like(Oz)
    Kz2 = np.zeros_like(Oz)
    ZKz = np.zeros_like(Oz)
    KzZ = np.zeros_like(Oz)

    for lH in range(lH_max):
        bra = np.conjugate(PhiH[lH])
        for lL in range(lL_max):
            ket = PhiL[lL]
            Oz[lH, lL]  = np.sum(bra * ket) * dz
            Z[lH, lL]   = np.sum(bra * (zz * ket)) * dz
            Z2[lH, lL]  = np.sum(bra * (zz**2 * ket)) * dz
            Kz[lH, lL]  = (-1j) * np.sum(bra * dPhiL[lL]) * dz if not symmetrize_kz else (-0.5j) * np.sum( (bra * dPhiL[lL] - np.conjugate(dPhiH[lH]) * ket) ) * dz
            Kz2[lH, lL] = (-1.0) * np.sum(bra * d2PhiL[lL]) * dz if not symmetrize_kz else np.sum( np.conjugate(dPhiH[lH]) * dPhiL[lL] ) * dz
            ZKz[lH, lL] = (-1j) * np.sum(bra * (zz * dPhiL[lL])) * dz
            #  -i ∫ φH*( φL + z φL' ) dz
            KzZ[lH, lL] = (-1j) * np.sum(bra * (ket + zz * dPhiL[lL])) * dz

    return dict(O=Oz, Z=Z, Z2=Z2, K=Kz, K2=Kz2, ZK=ZKz, KZ=KzZ)


# ----------------------------
# Build a full mixed 3D operator matrix between OrbitalBasis obH -> obL
# ----------------------------

def build_mixed_orbital_ops(obH: OrbitalBasis, obL: OrbitalBasis,
                            npts_xy=4001, pad_xy=5.0,
                            symmetrize_kz=False, dir2file_ho=None, gp=None):
	"""
	Mixed (HH rows, LH cols) orbital operators including Peierls substitution,
	matching the keys produced by OrbitalBasis.build_ops_shell(), but rectangular.

	Returns a dict with keys:
	  Iorb, X,Y,Z, X2,Y2,Z2, Kx,Ky,Kz, Cx,Cy,Cz,
	  Kx2,Ky2,Kz2, Cx2,Cy2,Cz2,
	  KxKy, KyKz, KxKz
	All matrices are (Nh x Nl), where Nh=len(obH.states), Nl=len(obL.states).
	"""
	statesH = obH.states
	statesL = obL.states

	# max quantum numbers
	nH_max = max(n for (n, _, _) in statesH)
	mH_max = max(m for (_, m, _) in statesH)
	lH_max = max(l for (_, _, l) in statesH)

	nL_max = max(n for (n, _, _) in statesL)
	mL_max = max(m for (_, m, _) in statesL)
	lL_max = max(l for (_, _, l) in statesL)

	# 1D mixed tables
	Tx = _ho_mixed_tables(obH.hox, obL.hox, nH_max, nL_max, gp, npts=npts_xy, pad=pad_xy, dir2file_ho=dir2file_ho)
	Ty = _ho_mixed_tables(obH.hoy, obL.hoy, mH_max, mL_max, gp, npts=npts_xy, pad=pad_xy, dir2file_ho=dir2file_ho)
	Tz = _z_mixed_tables(obH.zsb, obL.zsb, symmetrize_kz=symmetrize_kz)

	Nh = len(statesH)
	Nl = len(statesL)

	def _zeros(dtype=complex):
	    return np.zeros((Nh, Nl), dtype=dtype)

	# Identity-like map: not a true identity when Nh!=Nl, but often useful as <H|L>
	Iorb = _zeros(dtype=complex)
	# primitive mixed ops
	X  = _zeros(dtype=complex);   Y  = _zeros(dtype=complex);   Z  = _zeros(dtype=complex)
	X2 = _zeros(dtype=complex);   Y2 = _zeros(dtype=complex);   Z2 = _zeros(dtype=complex)
	Kx  = _zeros(dtype=complex); Ky  = _zeros(dtype=complex); Kz  = _zeros(dtype=complex)
	Kx2 = _zeros(dtype=complex);   Ky2 = _zeros(dtype=complex);   Kz2 = _zeros(dtype=complex)
	# extra mixed coordinate products needed for A^2 and A_i A_j terms
	XY = _zeros(dtype=complex); XZ = _zeros(dtype=complex); YZ = _zeros(dtype=complex)
	# extra mixed momentum-coordinate products needed for canonical products
	# x-k ordering matters in x and y:
	XKx = _zeros(dtype=complex)  # < x kx >
	KxX = _zeros(dtype=complex)  # < kx x >
	YKy = _zeros(dtype=complex)  # < y ky >
	KyY = _zeros(dtype=complex)  # < ky y >
	# separable mixed momentum products (pure k products)
	KxKy_pure = _zeros(dtype=complex)
	KyKz_pure = _zeros(dtype=complex)
	KxKz_pure = _zeros(dtype=complex)
	# z ordering-sensitive (already provided by z tables)
	ZKz = _zeros(dtype=complex)
	KzZ = _zeros(dtype=complex)

	# First fill primitives + the extra products
	for i, (nH, mH, lH) in enumerate(statesH):
	    for j, (nL, mL, lL) in enumerate(statesL):
	        ox = Tx["O"][nH, nL]
	        oy = Ty["O"][mH, mL]
	        oz = Tz["O"][lH, lL]

	        Iorb[i, j] = ox * oy * oz

	        # coords
	        X[i, j]  = Tx["X"][nH, nL]  * oy * oz
	        Y[i, j]  = ox * Ty["Y"][mH, mL]  * oz
	        Z[i, j]  = ox * oy * Tz["Z"][lH, lL]

	        X2[i, j] = Tx["X2"][nH, nL] * oy * oz
	        Y2[i, j] = ox * Ty["Y2"][mH, mL] * oz
	        Z2[i, j] = ox * oy * Tz["Z2"][lH, lL]

	        # coord products (commuting, so XY=YX etc.)
	        XY[i, j] = Tx["X"][nH, nL] * Ty["Y"][mH, mL] * oz
	        XZ[i, j] = Tx["X"][nH, nL] * oy * Tz["Z"][lH, lL]
	        YZ[i, j] = ox * Ty["Y"][mH, mL] * Tz["Z"][lH, lL]

	        # momenta
	        Kx[i, j]  = Tx["Kx"][nH, nL] * oy * oz
	        Ky[i, j]  = ox * Ty["Ky"][mH, mL] * oz
	        Kz[i, j]  = ox * oy * Tz["K"][lH, lL]

	        Kx2[i, j] = Tx["Kx2"][nH, nL] * oy * oz
	        Ky2[i, j] = ox * Ty["Ky2"][mH, mL] * oz
	        Kz2[i, j] = ox * oy * Tz["K2"][lH, lL]

	        # ordering-sensitive x and y (needed when A contains X or Y)
	        XKx[i, j] = Tx["XK"][nH, nL] * oy * oz
	        KxX[i, j] = Tx["KX"][nH, nL] * oy * oz
	        YKy[i, j] = ox * Ty["YK"][mH, mL] * oz
	        KyY[i, j] = ox * Ty["KY"][mH, mL] * oz

	        # pure k products
	        KxKy_pure[i, j] = Tx["Kx"][nH, nL] * Ty["Ky"][mH, mL] * oz
	        KxKz_pure[i, j] = Tx["Kx"][nH, nL] * oy * Tz["K"][lH, lL]
	        KyKz_pure[i, j] = ox * Ty["Ky"][mH, mL] * Tz["K"][lH, lL]

	        # z ordering sensitive
	        ZKz[i, j] = ox * oy * Tz["ZK"][lH, lL]
	        KzZ[i, j] = ox * oy * Tz["KZ"][lH, lL]

	# Vector potential A and A^2, symmetric gauge
	Bx, By, Bz = obH.B

	Ax  = 0.5 * (2 * By * Z - Bz * Y)
	Ay  = 0.5 * (Bz * X - 2 * Bx * Z)
	Az  = 0.0 * X

	Ax2 = 0.25 * (4 * By**2 * Z2 + Bz**2 * Y2 - 4 * By * Bz * YZ)
	Ay2 = 0.25 * (Bz**2 * X2 + 4 * Bx**2 * Z2 - 4 * Bz * Bx * XZ)
	Az2 = 0.0 * X

	AyAz = 0.0 * X
	AxAz = 0.0 * X

	# Canonical momenta C = K + alpha A  (mixed)
	alpha = - E_CH / HBAR
	Cx = Kx + alpha * Ax
	Cy = Ky + alpha * Ay
	Cz = Kz + alpha * Az

	# ------------------------------------------------------------
	# Build <Kx Ax>, <Ky Ay>, <Kz Az> using separability expansions.
	KxY = (KxKy_pure * 0.0)
	KxZ = (KxKy_pure * 0.0)
	KyX = (KxKy_pure * 0.0)
	KyZ = (KxKy_pure * 0.0)
	KzX = (KxKy_pure * 0.0)
	KzY = (KxKy_pure * 0.0)

	# Fill using already-built separable pieces:
	for i, (nH, mH, lH) in enumerate(statesH):
	    for j, (nL, mL, lL) in enumerate(statesL):
	        ox = Tx["O"][nH, nL]
	        oy = Ty["O"][mH, mL]
	        oz = Tz["O"][lH, lL]

	        # <kx y>
	        KxY[i, j] = Tx["Kx"][nH, nL] * Ty["Y"][mH, mL] * oz
	        # <kx z>
	        KxZ[i, j] = Tx["Kx"][nH, nL] * oy * Tz["Z"][lH, lL]
	        # <ky x>
	        KyX[i, j] = Tx["X"][nH, nL] * Ty["Ky"][mH, mL] * oz
	        # <ky z>
	        KyZ[i, j] = ox * Ty["Ky"][mH, mL] * Tz["Z"][lH, lL]
	        # <kz x>
	        KzX[i, j] = Tx["X"][nH, nL] * oy * Tz["K"][lH, lL]
	        # <kz y>
	        KzY[i, j] = ox * Ty["Y"][mH, mL] * Tz["K"][lH, lL]

	# Compute {Kx,Ax}, {Ky,Ay}, {Kz,Az} without @
	# Ax = 0.5(2 By Z - Bz Y), Ay = 0.5(Bz X - 2 Bx Z), Az=0
	KxAx = 0.5 * (2 * By * KxZ - Bz * KxY)
	KyAy = 0.5 * (Bz * KyX - 2 * Bx * KyZ)
	KzAz = 0.0 * Kz

	# squares
	Cx2 = Kx2 + 2 * alpha * KxAx + (alpha**2) * Ax2
	Cy2 = Ky2 + 2 * alpha * KyAy + (alpha**2) * Ay2
	Cz2 = Kz2 + 2 * alpha * KzAz + (alpha**2) * Az2

	# ------------------------------------------------------------
	# Mixed symmetrized products needed by LK:
	sym_KxKy = KxKy_pure

	# Ay = 0.5(Bz X - 2Bx Z)
	sym_KxX = 0.5 * (KxX + XKx)
	sym_KxZ = KxZ
	sym_KxAy = 0.5 * (Bz * sym_KxX - 2 * Bx * sym_KxZ)

	# Ax = 0.5(2By Z - Bz Y)
	sym_KyY = 0.5 * (KyY + YKy)
	sym_KyZ = KyZ
	sym_AxKy = 0.5 * (2 * By * sym_KyZ - Bz * sym_KyY)

	# AxAy = 0.25(2By Z - Bz Y)(Bz X - 2Bx Z)
	# = 0.25( 2By*Bz ZX - 4By*Bx Z2 - Bz^2 YX + 2Bz*Bx YZ )
	AxAy = 0.25 * (2*By*Bz * XZ - 4*By*Bx * Z2 - (Bz**2) * XY + 2*Bz*Bx * YZ)
	sym_AxAy = AxAy

	KxKy = sym_KxKy + alpha * sym_KxAy + alpha * sym_AxKy + (alpha**2) * sym_AxAy

	# --- sym(Cy,Cz) ---
	# Cz = Kz + αAz.  In sym gauge Az depends on X,Y (no z), so [Kz,Az]=0.
	sym_KyKz = KyKz_pure
	XKz = KzX
	sym_ZKz = 0.5 * (ZKz + KzZ)
	sym_AyKz = 0.5 * (Bz * XKz - 2 * Bx * sym_ZKz)
	sym_KyAz = 0.0 * Ky
	sym_AyAz = 0.0 * Ky

	KyKz = sym_KyKz + alpha * sym_KyAz + alpha * sym_AyKz + (alpha**2) * sym_AyAz

	# --- sym(Cx,Cz) ---
	sym_KxKz = KxKz_pure
	sym_ZKz = 0.5 * (ZKz + KzZ)
	YKz = KzY
	sym_AxKz = 0.5 * (2 * By * sym_ZKz - Bz * YKz)
	sym_KxAz = 0.0 * Kx
	sym_AxAz = 0.0 * Kx

	KxKz = sym_KxKz + alpha * sym_KxAz + alpha * sym_AxKz + (alpha**2) * sym_AxAz

	# Return everything
	return dict(Iorb=Iorb,
	            X=X, Y=Y, Z=Z, X2=X2, Y2=Y2, Z2=Z2,
	            Kx=Kx, Ky=Ky, Kz=Kz, Cx=Cx, Cy=Cy, Cz=Cz,
	            Kx2=Kx2, Ky2=Ky2, Kz2=Kz2, Cx2=Cx2, Cy2=Cy2, Cz2=Cz2,
	            KxKy=KxKy, KyKz=KyKz, KxKz=KxKz)