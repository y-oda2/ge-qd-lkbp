"""
Z subbands and Airy-function basis for a three-region SiGe/Ge/SiGe heterostructure.

Solves the 1D Schrödinger equation in a triangular-well potential (vertical electric
field Fz) with a finite rectangular well (Ge quantum well) and SiGe barriers above
and below. Eigenfunctions are piecewise Airy functions matched at the two interfaces
(z=0 and z=-dw). Provides matrix elements <l|z|l'>, <l|kz|l'>, <l|kz²|l'> etc.
for use in the 3D orbital basis (orbital_basis.py).

Reference: Burkard et al., PRR 2, 043180 (2020).
"""
import sys
sys.path.append('Code')
from model_params import *
import scipy
import scipy.special
import scipy.optimize
import pickle as pk

# Airy functions
def Ai(x):  return scipy.special.airy(x)[0]
def Aip(x): return scipy.special.airy(x)[1]
def Bi(x):  return scipy.special.airy(x)[2]
def Bip(x): return scipy.special.airy(x)[3]

# Directory for saved z wavefunctions
import platform
dir2file_z = r"C:\\Users\\14432\\UMBC\\Research\\Code\\Ge_pO\\Code\\" if platform.system() == "Windows" else "/home/yoda1/Ge/"
# import os
# dir2file_z = os.environ.get("GE_QD_DATA_DIR", ".")

# Z Subbands definition and calculation
class ZSubbands():
    def __init__(self, gp:GeQDParams, fp:FieldParams, l_max_H=3, l_max_L=3, holetype='H', verbose=False, \
                    compute_basis=False, nz=10000, symmetrize_kz=False):
        # Get device parameters
        self.di = gp.ht.di    # Distance from top of well to bottom of oxide layer
        self.dw = gp.ht.dw    # Well width
        self.Fz = fp.Fz        # Vertical electric field
        self.U0 = gp.cn.UH if holetype=='H' else gp.cn.UL        # Potential offset between Ge and SiGe layers, different for HH vs LH
        self.mz = gp.lk.mH_z if holetype=='H' else gp.lk.mL_z    # Effective out-of-plane mass, different for HH vs LH 

        # Compute length and energy scales for the triangular well
        self.z0   = ( HBAR**2 / ( 2 * self.mz * E_CH * np.abs(self.Fz) ) )**(1/3)
        self.eps0 =   HBAR**2 / ( 2 * self.mz * self.z0**2 )
        # Print
        self.verbose = verbose
        if verbose:
            print("z0 = %.4f nm" % (self.z0*1e9))
            print("ε0 = %.4f meV" % (self.eps0*J2meV))

        # Compute normalized quantities
        self.di_tilde = self.di / self.z0
        self.dw_tilde = self.dw / self.z0
        self.U0_tilde = self.U0 / self.eps0

        # Initialize relevant quantities
        self.M = None
        self.E_zns = []
        self.c_basis = []
        self.z_basis = []

        # Number of basis states
        self.l_max = l_max_H if holetype=='H' else l_max_L
        self.symmetrize_kz = symmetrize_kz
        self.holetype = holetype

        # Get basis
        if compute_basis:
            self.get_z_wavefunction_params(nz=nz)
            self.get_z_basis()
        else:
            c_basis_all = pk.load(open(dir2file_z+"z_wavefncs_coeffs.pk","rb"))
            if self.verbose: print("Retrieved z wavefunction basis parameters.")
            try:
                self.c_basis = c_basis_all[holetype][self.Fz]['c'][:self.l_max]
                self.E_zns   = c_basis_all[holetype][self.Fz]['Ezn'][:self.l_max]
                self.zmin    = c_basis_all[holetype][self.Fz]['zmins'][l_max_L]
                nz = 100 * int( (self.di-self.zmin)/self.dw )
                if self.verbose: print("nz = %d"%nz)
                self.zz = np.linspace(self.zmin,self.di,nz)
                self.dz = self.zz[1]-self.zz[0]
                if self.verbose: print("Computing z wavefunction from saved basis parameters.")
                self.get_z_basis()
            except:
                print("Error: Fz likely not found in c_basis. Run with 'compute_basis=True'.")


    # Defining algebraic equation to find eigenvalues (from determinant)
    # eps_zn: normalized eigenenergies (to be determined)
    def f1(self, eps_zn):
        return Bip(self.dw_tilde-eps_zn) * Ai(self.U0_tilde+self.dw_tilde-eps_zn) - \
               Bi(self.dw_tilde-eps_zn) * Aip(self.U0_tilde+self.dw_tilde-eps_zn)

    def f2(self, eps_zn):
        return Ai(self.dw_tilde-eps_zn) * Aip(self.U0_tilde+self.dw_tilde-eps_zn) - \
               Aip(self.dw_tilde-eps_zn) * Ai(self.U0_tilde+self.dw_tilde-eps_zn)

    def g1(self, eps_zn):
        chi = self.U0_tilde - self.di_tilde - eps_zn
        return Ai(-eps_zn) * ( Bip(self.U0_tilde-eps_zn) - Aip(self.U0_tilde-eps_zn) * Bi(chi)/Ai(chi) ) - \
               Aip(-eps_zn) * ( Bi(self.U0_tilde-eps_zn) - Ai(self.U0_tilde-eps_zn)  * Bi(chi)/Ai(chi) )

    def g2(self, eps_zn):
        chi = self.U0_tilde - self.di_tilde - eps_zn
        return Bip(-eps_zn) * ( Bi(self.U0_tilde-eps_zn) - Ai(self.U0_tilde-eps_zn)  * Bi(chi)/Ai(chi) ) - \
               Bi(-eps_zn) * ( Bip(self.U0_tilde-eps_zn) - Aip(self.U0_tilde-eps_zn) * Bi(chi)/Ai(chi) )

    def algebraic_eq(self, eps_zn):
        return self.f1(eps_zn) * self.g1(eps_zn) - self.f2(eps_zn) * self.g2(eps_zn)

    ### Get boundary condition matrix M
    def get_M(self, eps_zn):
        # Compute effective lengths
        zeta_upper_top    = self.U0_tilde - self.di_tilde - eps_zn
        zeta_upper_bottom = self.U0_tilde - eps_zn
        zeta_well_top     = - eps_zn
        zeta_well_bottom  = self.dw_tilde - eps_zn
        zeta_lower_top    = self.U0_tilde + self.dw_tilde - eps_zn
        # Compute coefficient matrix
        self.M = np.array([[Ai(zeta_upper_top), Bi(zeta_upper_top), 0, 0, 0],
                           [Ai(zeta_upper_bottom), Bi(zeta_upper_bottom), - Ai(zeta_well_top), - Bi(zeta_well_top), 0],
                           [Aip(zeta_upper_bottom), Bip(zeta_upper_bottom), - Aip(zeta_well_top), - Bip(zeta_well_top), 0],
                           [0, 0, Ai(zeta_well_bottom), Bi(zeta_well_bottom), - Ai(zeta_lower_top)],
                           [0, 0, Aip(zeta_well_bottom), Bip(zeta_well_bottom), - Aip(zeta_lower_top)] ])
        return self.M

    ### Get reduced boundary condition matrix M after removing c5
    def get_reduced_M(self, eps_zn):
        # Airy arguments
        zeta_upper_top    = self.U0_tilde - self.di_tilde - eps_zn
        zeta_upper_bottom = self.U0_tilde - eps_zn
        zeta_well_top     = -eps_zn
        zeta_well_bottom  = self.dw_tilde - eps_zn
        zeta_lower_top    = self.U0_tilde + self.dw_tilde - eps_zn

        # c5-eliminated compatibility at z = -dw:
        # (Ai(zwb) * Aip(zlt) - Aip(zwb) * Ai(zlt)) * c3 + (Bi(zwb) * Aip(zlt) - Bip(zwb) * Ai(zlt)) * c4 = 0
        comp_c3 = Ai(zeta_well_bottom) * Aip(zeta_lower_top) - Aip(zeta_well_bottom) * Ai(zeta_lower_top)
        comp_c4 = Bi(zeta_well_bottom) * Aip(zeta_lower_top) - Bip(zeta_well_bottom) * Ai(zeta_lower_top)

        G = np.array([
            [Ai(zeta_upper_top),     Bi(zeta_upper_top),      0.0,                0.0],
            [Ai(zeta_upper_bottom),  Bi(zeta_upper_bottom),  -Ai(zeta_well_top), -Bi(zeta_well_top)],
            [Aip(zeta_upper_bottom), Bip(zeta_upper_bottom), -Aip(zeta_well_top),-Bip(zeta_well_top)],
            [0.0,                    0.0,                     comp_c3,            comp_c4] ], dtype=complex)

        return G, Ai(zeta_well_bottom) / Ai(zeta_lower_top), Bi(zeta_well_bottom) / Ai(zeta_lower_top)

    ### Calculate determinant
    def get_detM(self, eps_zn):
        # Update M
        self.get_M(eps_zn)
        # Get determinant
        return np.linalg.det(self.M)
    
    # Get c's analytically
    def get_c_vals(self, eps_zn, tol=1e-100):
        # --- Airy arguments ---
        zut = self.U0_tilde - self.di_tilde - eps_zn
        zub = self.U0_tilde - eps_zn
        zwt = -eps_zn
        zwb = self.dw_tilde - eps_zn
        zlt = self.U0_tilde + self.dw_tilde - eps_zn

        # --- From Mathematica 
        W_ut_ub = Ai(zut) * Bi(zub) - Ai(zub) * Bi(zut)
        A = Ai(zwb) * Aip(zlt) - Ai(zlt) * Aip(zwb)
        B = Aip(zlt) * Bi(zwb) - Ai(zlt) * Bip(zwb)
        D = Bi(zut) * (-A * Bi(zwt) + Ai(zwt) * B)

        # --- Fix gauge ---
        c1 = 1.0

        # Find c2
        if abs(Bi(zut)) < tol:
            if self.verbose: print("Bi(zut) too small — cannot compute c2. εzn=%.2f"%(eps_zn))
            return []
        c2 = -c1 * Ai(zut) / Bi(zut)
        # Find c3,c4
        if abs(D) < tol:
            if self.verbose: print("Denominator D nearly zero — eps_zn likely not an eigenvalue. εzn=%.2f"%(eps_zn))
            return []
        c3 = - c1 * (W_ut_ub * B) / D
        c4 =   c1 * (W_ut_ub * A) / D
        # Check derivative at z=0
        psip_ub = c1 * Aip(zub) + c2 * Bip(zub)
        psip_wt = c3 * Aip(zwt) + c4 * Bip(zwt)
        if abs( psip_ub - psip_wt) / np.sqrt( psip_ub**2 + psip_wt**2 )>1e-1 and abs( psip_ub - psip_wt)>1e-1:
            if self.verbose: 
                print("Derivability at z=0 not satisfied. |ψ'(ζub)-ψ'(ζwt)| = %.2e; |(ψ'(ζub),ψ'(ζwt))| = %.2e. eps_zn = %.2f" % (abs( (c1*Aip(zub)+c2*Bip(zub)) - (c3*Aip(zwt)+c4*Bip(zwt)) ),np.sqrt( (c1*Aip(zub)+c2*Bip(zub))**2 + (c3*Aip(zwt)+c4*Bip(zwt))**2 ),eps_zn))
            return []
        # Find c5
        if abs(Ai(zlt)) < tol:
            if self.verbose: print("Ai(zlt) too small — cannot compute c5. εzn=%.2f"%(eps_zn))
            return []
        c5 = (c3 * Ai(zwb) + c4 * Bi(zwb)) / Ai(zlt)

        return np.array([c1, c2, c3, c4, c5], dtype=complex)


    ### Find eigenenergies
    def get_z_wavefunction_params(self, 
        E_1=-10/J2meV, # Starting point for the search. From Burkard's Fig. 2 we can see that there aren't any values lower than this.
        dE=0.01/J2meV, # Incremental energy step of the search, in J. In practice this seems small enough, but if there are two zeros closer than dE they could be missed.
        nz=10000       # Number of z steps, for normalization.
        ):

        # z range and steps
        self.zz = np.linspace(-9*self.di,self.di,nz)
        self.dz = self.zz[1]-self.zz[0]

        # Find eigenvalues using algebraic equation
        self.E_zns = []
        self.c_basis = []
        # Iterate over energies until the length has been reached
        while len(self.E_zns)<self.l_max:
            # Iterate over steps of dE
            E_2 = E_1 + dE
            # Find det(M(ε)) for the window 
            eq1, eq2 = self.get_detM(E_1/self.eps0), self.get_detM(E_2/self.eps0)
            # Check if the function has changed signs
            if np.sign(eq1)!=np.sign(eq2):
                # Find eigenvalue candidate εzn
                eps_zn = scipy.optimize.brentq( self.get_detM, E_1/self.eps0, E_2/self.eps0, xtol=1e-30, maxiter=1000)
        
                # Get wavefunctions parameters. Check for consistency
                c_vec = self.get_c_vals(eps_zn)
                if len(c_vec)==0: E_1 = E_2; continue
                # Get wavefunction and normalize
                psi_vec = np.array([self.get_psi_zn(z,eps_zn,c_vec) for z in self.zz])
                norm    = self.get_norm(psi_vec)
                psi_norm = psi_vec/np.sqrt(norm)
                # Check orthogonality
                if not self.check_orthogonality(psi_norm): E_1 = E_2; continue

                # If checks were passed, add to basis list
                self.E_zns += [eps_zn*self.eps0]
                self.c_basis += [c_vec/np.sqrt(norm)]
                self.z_basis += [psi_norm]

            # Continue to next interval
            E_1 = E_2

        self.E_zns = np.array(self.E_zns)
        return self.E_zns, self.c_basis

    # Check orthogonality of given function w.r.t existing basis
    def check_orthogonality(self, psi, tol=1e-1):
        for i,psi_b in enumerate(self.z_basis):
            overlap = np.sum(psi_b.conj() * psi)*self.dz
            # Compare overlap to given tolerance, or a reasonable ~O(dz/(ztop-zbottom)) error
            if abs( overlap ) > max(tol, abs(self.dz/(self.zz[-1]-self.zz[0]))):
                if self.verbose: print("Overlap failed with basis state number %d: <ψi|ψ>=%.2e"%(i,abs(overlap)))
                return False
        return True

    # Check weight in well
    def check_well_weight(self, psi, tol=0.5):
        psi_well = np.abs(psi[(-self.dw<=self.zz) & (self.zz <= 0)])**2 
        weight = np.sum(psi_well)*self.dz
        if weight>1-tol:
            return True
        else:
            if self.verbose: print("HH wavefunction lies outside the well: %.2e"%weight)
            return False

    ### Get z wavefunction. See Burkard's Eqs.(7,8). Unnormalized, e.g. ψ(z) = c1 Ai(ζ(z)) + c2 Bi(ζ(z))
    def get_psi_zn(self, z, eps_zn, c_vec):
        c1, c2, c3, c4, c5 = c_vec

        if (0<z) and (z<=self.di):
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return (c1*Ai(zeta_n) + c2*Bi(zeta_n))/np.sqrt(self.z0)
        elif (-self.dw<=z) and (z<=0):
            zeta_n = - z/self.z0 - eps_zn
            return (c3*Ai(zeta_n) + c4*Bi(zeta_n))/np.sqrt(self.z0)
        elif z<-self.dw:
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return c5*Ai(zeta_n)/np.sqrt(self.z0)
        else:
            raise ValueError(f"z={z:.3e} is outside all defined regions for this heterostructure.")

    ### Get z wavefunction first derivative. Unnormalized, e.g. ψ'(z) = ( c1 Ai'(ζ(z)) + c2 Bi'(ζ(z)) ) * (-1/z0), due to dζ/dz = -1/z0
    def get_psi_zn_der(self, z, eps_zn, c_vec):
        c1, c2, c3, c4, c5 = c_vec

        jac = -1/self.z0
        if (0<z) and (z<=self.di):
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return (c1*Aip(zeta_n) + c2*Bip(zeta_n))/np.sqrt(self.z0) * jac
        elif (-self.dw<=z) and (z<=0):
            zeta_n = - z/self.z0 - eps_zn
            return (c3*Aip(zeta_n) + c4*Bip(zeta_n))/np.sqrt(self.z0) * jac
        elif z<-self.dw:
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return c5*Aip(zeta_n)/np.sqrt(self.z0) * jac
        else:
            raise ValueError(f"z={z:.3e} is outside all defined regions for this heterostructure.")

    ### Get z wavefunction first derivative. Unnormalized, e.g. ψ''(z) = ψ(z) * ( ζ(z)/z0^2 ) due to Ai''(x) = x Ai(x)
    def get_psi_zn_der2(self, z, eps_zn, c_vec):
        c1, c2, c3, c4, c5 = c_vec

        jac = 1/self.z0**2
        if (0<z) and (z<=self.di):
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return (c1*Ai(zeta_n) + c2*Bi(zeta_n))/np.sqrt(self.z0) * jac*zeta_n
        elif (-self.dw<=z) and (z<=0):
            zeta_n = - z/self.z0 - eps_zn
            return (c3*Ai(zeta_n) + c4*Bi(zeta_n))/np.sqrt(self.z0) * jac*zeta_n
        elif z<-self.dw:
            zeta_n = self.U0_tilde - z/self.z0 - eps_zn
            return c5*Ai(zeta_n)/np.sqrt(self.z0) * jac*zeta_n
        else:
            raise ValueError(f"z={z:.3e} is outside all defined regions for this heterostructure.")

    ### Normalize the wavefunction in vector form (numpy array)
    def get_norm(self, psi_vec):
        return np.sum(np.abs(psi_vec)**2) * self.dz

    def normalize(self, psi_vec):
        norm = self.get_norm(psi_vec)
        return psi_vec/np.sqrt(norm)

    ### Compute ψn(z),ψn'(z),ψn''(z). 
    ### It needs self.c_basis and self.zz; ensure this is run after get_z_wavefunction_params()
    def get_z_basis(self):
        self.z_basis = []
        self.z_basis_der = []
        self.z_basis_der2 = []

        for i,E_zn in enumerate(self.E_zns):
            psi_vec = np.array([self.get_psi_zn(z, E_zn/self.eps0, self.c_basis[i]) for z in self.zz])
            psip_vec = np.array([self.get_psi_zn_der(z, E_zn/self.eps0, self.c_basis[i]) for z in self.zz])
            psipp_vec = np.array([self.get_psi_zn_der2(z, E_zn/self.eps0, self.c_basis[i]) for z in self.zz])

            self.z_basis += [psi_vec]
            self.z_basis_der += [psip_vec]
            self.z_basis_der2 += [psipp_vec]

        return self.z_basis, self.z_basis_der, self.z_basis_der2

    ### Get matrix elements
    def get_op_z(self, l, lp):
        return np.sum(self.z_basis[l] * self.zz * self.z_basis[lp]).real * self.dz

    def get_op_z2(self, l, lp):
        return np.sum(self.z_basis[l] * self.zz**2 * self.z_basis[lp]).real * self.dz

    def get_op_kz(self, l, lp):
        if self.symmetrize_kz:
            a = np.conj(self.z_basis[l]) * self.z_basis_der[lp]
            b = np.conj(self.z_basis_der[l]) * self.z_basis[lp]
            return (-1j/2) * np.sum(a - b) * self.dz
        else:
            return -1j * np.sum(self.z_basis[l] * self.z_basis_der[lp]).real * self.dz

    def get_op_kz2(self, l, lp):
        if self.symmetrize_kz:
            return np.sum(np.conj(self.z_basis_der[l]) * self.z_basis_der[lp]).real * self.dz
        else:
            return - np.sum(self.z_basis[l] * self.z_basis_der2[lp]).real * self.dz

    def get_op_zkz(self, l, lp):
        return -1j * np.sum(self.z_basis[l] * self.zz * self.z_basis_der[lp]).real * self.dz

    def get_op_kzz(self, l, lp):
        return -1j * np.sum(self.z_basis[l] * ( self.z_basis[lp] + self.zz * self.z_basis_der[lp] ) ).real * self.dz