from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'cm'
pycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2
import numpy as np

def xy_density_from_eigenvector_HL(vec, statesH, statesL, obH, obL,
                                   nx=201, ny=201, pad_x=4.5, pad_y=4.5):
    """
    Returns (X, Y, rho) where rho(x,y) is the band- (HH/LH), spin-, and z-marginalized density.
    Basis ordering assumed: [HH+, HH-, LH+, LH-] with sizes [Nh, Nh, Nl, Nl].

    - HH components are expanded in statesH with HO lengths from obH.hox/hoy
    - LH components are expanded in statesL with HO lengths from obL.hox/hoy

    Marginalization over z is implemented by summing |A_l,s(x,y)|^2 per l, per component,
    i.e. assuming orthonormal z-basis within each sector (HH z-basis and LH z-basis separately).
    """
    Nh = len(statesH)
    Nl = len(statesL)

    # --- slice coefficients
    c_hhp = vec[0*Nh : 1*Nh]
    c_hhm = vec[1*Nh : 2*Nh]
    c_lhp = vec[2*Nh : 2*Nh + Nl]
    c_lhm = vec[2*Nh + Nl : 2*Nh + 2*Nl]

    # Choose a plotting range. Use the larger HO length so both sectors fit comfortably.
    Lx = max(obH.hox.a0, obL.hox.a0)
    Ly = max(obH.hoy.a0, obL.hoy.a0)

    x = np.linspace(-pad_x*Lx, pad_x*Lx, nx)
    y = np.linspace(-pad_y*Ly, pad_y*Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    rho = np.zeros_like(X, dtype=float)

    def add_sector(states, ob, c1, c2):
        nonlocal rho

        nml = np.array(states, dtype=int)
        n_max = int(nml[:, 0].max()) if len(states) else 0
        m_max = int(nml[:, 1].max()) if len(states) else 0

        # precompute 1D HO functions for this sector's HO lengths
        psi_x = [ob.hox.ho_psi(n, x, ob.hox.a0) for n in range(n_max + 1)]
        psi_y = [ob.hoy.ho_psi(m, y, ob.hoy.a0) for m in range(m_max + 1)]

        l_vals = np.unique(nml[:, 2])

        # component 0: "+" (HH+ or LH+), component 1: "-" (HH- or LH-)
        for coeffs in (c1, c2):
            # group by l to marginalize z
            for l in l_vals:
                idx = np.where(nml[:, 2] == l)[0]
                if idx.size == 0:
                    continue

                A = np.zeros_like(X, dtype=complex)
                for k in idx:
                    n, m, _ = nml[k]
                    c = coeffs[k]
                    if c == 0:
                        continue
                    A += c * (psi_x[n][None, :] * psi_y[m][:, None])

                rho += (A * A.conjugate()).real

    # Add HH sector (HH+, HH-)
    add_sector(statesH, obH, c_hhp, c_hhm)
    # Add LH sector (LH+, LH-)
    add_sector(statesL, obL, c_lhp, c_lhm)

    return X, Y, rho

def z_density_from_eigenvector_HL(vec, statesH, statesL, obH, obL):
    """
    Returns (zz, rho) where rho(z) is band- (HH/LH), spin-, and xy-marginalized density.
    Basis ordering assumed: [HH+, HH-, LH+, LH-] with sizes [Nh, Nh, Nl, Nl].

    If HH and LH z-grids differ, LH density is interpolated onto HH grid before summing.
    """
    Nh = len(statesH)
    Nl = len(statesL)

    c_hhp = vec[0*Nh : 1*Nh]
    c_hhm = vec[1*Nh : 2*Nh]
    c_lhp = vec[2*Nh : 2*Nh + Nl]
    c_lhm = vec[2*Nh + Nl : 2*Nh + 2*Nl]

    def sector_rho_z(states, ob, c1, c2):
        zz = ob.zsb.zz
        Phi = np.array(ob.zsb.z_basis)  # (lmax, nz)

        nml = np.array(states, dtype=int)
        ls = nml[:, 2]
        nm_to_idx = {}
        for k, (n, m, l) in enumerate(nml):
            nm_to_idx.setdefault((n, m), []).append(k)

        rho = np.zeros_like(zz, dtype=float)

        for coeffs in (c1, c2):
            for idx_list in nm_to_idx.values():
                A = np.zeros_like(zz, dtype=complex)
                for k in idx_list:
                    l = int(ls[k])
                    c = coeffs[k]
                    if c != 0.0:
                        A += c * Phi[l]
                rho += (A * A.conjugate()).real

        return zz, rho

    zzH, rhoH = sector_rho_z(statesH, obH, c_hhp, c_hhm)
    zzL, rhoL = sector_rho_z(statesL, obL, c_lhp, c_lhm)

    # If grids match, sum directly; else interpolate LH onto HH grid.
    if (len(zzH) == len(zzL)) and np.allclose(zzH, zzL, rtol=0, atol=1e-12):
        return zzH, rhoH + rhoL
    else:
        rhoL_on_H = np.interp(zzH, zzL, rhoL, left=0.0, right=0.0)
        return zzH, rhoH + rhoL_on_H


# Matrix plots
def plot_mat(op,label='',vs=None):
    plt.figure(figsize=(4,3),dpi=150)
    plt.title(label,size=16)
    if np.all(op==0.):
        plt.imshow(op,cmap='seismic')
    elif vs is not None:
        vm,vc,vM = vs
        norm=colors.TwoSlopeNorm(vmin=vm, vcenter=vc, vmax=vM)
        plt.imshow(op,cmap='seismic',norm=norm)
    else:
        norm=colors.TwoSlopeNorm(vmin=min(np.min(op),-np.max(op)), vcenter=0., vmax=max(np.max(op),-np.min(op)))
        plt.imshow(op,cmap='seismic',norm=norm)
    plt.colorbar()
    plt.ylabel("$m$",size=14)
    plt.xlabel("$n$",size=14)
    plt.show()
