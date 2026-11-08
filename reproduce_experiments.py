# =============================================================================
# Reproduction script for the revised manuscript
# "Real-Time Grasping Force Optimization via a Parameter-Free Adaptive
#  Golden-Ratio Algorithm with Automatic Active-Set Identification"
#
# This single script verifies EVERY theoretical statement and regenerates
# EVERY numerical value, table and figure of the revised manuscript:
#   * correct circular-cone projection (Lemma 2.3) incl. optimality check
#     and a counterexample to the previously used (incorrect) formula
#   * exact Lipschitz constant L_F = ||A||^2 (Proposition 2.6)
#   * stepsize inequality (Lemma 4.1) and the unconditional lower bound
#     (Lemma 4.2)
#   * energy inequality (Lemma 4.3) with the sharp dual term, incl. the
#     observed violation of its (invalid) k = 1 instance
#   * ergodic O(1/k) bound with explicit constant C_0 = V_2/2 (Theorem 4.4)
#   * Bregman identity (Lemma 4.6)
#   * projection Jacobian spectrum {1, sigma, 0} (Lemma 4.8) vs finite diff.
#   * mu_T = lambda_min(A J* A^T), kappa_T, strict-complementarity margin gamma
#   * finite active-set identification (observed K_id, Theorem 4.15)
#   * a-posteriori certified local contraction factor (Theorem 4.17) with the
#     strong-monotonicity modulus verified along the trajectory
#   * sensitivity identity ||dP/d mu|| = ||z||/(1+mu^2) (Lemma 6.1),
#     checked symbolically (sympy) and by finite differences, and the
#     robustness constant C_mu = 1/(1+mu_min^2)
#   * all performance tables (cold start, warm trajectory, robustness,
#     phi-sensitivity, per-iteration cost) and all figures
#
# Requires: numpy, scipy, matplotlib.   Run:  python3 reproduce_experiments.py
# Tested with Python 3.12, numpy 2.x, scipy 1.x.
# =============================================================================
import numpy as np
from scipy.optimize import minimize, root

# ---------------------------------------------------------------- projection
def proj_circ(z, mu):
    """Correct metric projection onto L_theta = {(x,s): ||x|| <= mu s}."""
    x = np.asarray(z[:2], dtype=float); s = float(z[2]); nx = np.linalg.norm(x)
    if nx <= mu * s:    return np.array([x[0], x[1], s])     # z in L_theta
    if nx <= -s / mu:   return np.zeros(3)                   # z in polar cone
    w = (mu * nx + s) / (1.0 + mu**2)                        # boundary sheet
    return np.array([w*mu*x[0]/nx, w*mu*x[1]/nx, w])

def proj_soc(z):
    """Metric projection onto the standard second-order cone (mu = 1)."""
    x = np.asarray(z[:2], dtype=float); s = float(z[2]); nx = np.linalg.norm(x)
    if nx <= s:   return np.array([x[0], x[1], s])
    if nx <= -s:  return np.zeros(3)
    a = s / nx
    return 0.5*(1+a)*np.array([x[0], x[1], nx])

def wrong_proj(z, mu):
    """The INCORRECT formula P_L = T^{-1} P_K (T .) of the rejected version."""
    T  = np.diag([1.0, 1.0, mu]); Ti = np.diag([1.0, 1.0, 1.0/mu])
    return Ti @ proj_soc(T @ z)

def check_projection():
    rng = np.random.default_rng(0)
    worst_kkt = 0.0; worst_lip = 0.0
    for _ in range(200000):
        z = rng.normal(size=3); mu = rng.uniform(0.2, 1.0)
        p = proj_circ(z, mu)
        t = rng.uniform(0, 5); ang = rng.uniform(0, 2*np.pi)
        q = np.array([mu*t*np.cos(ang)*rng.uniform(0,1),
                      mu*t*np.sin(ang)*rng.uniform(0,1), t])
        worst_kkt = max(worst_kkt, np.linalg.norm(p-z) - np.linalg.norm(q-z))
        z2 = rng.normal(size=3)
        d = np.linalg.norm(proj_circ(z,mu)-proj_circ(z2,mu)) \
            / max(np.linalg.norm(z-z2), 1e-300)
        worst_lip = max(worst_lip, d)
    z = np.array([2.0, 0.0, 1.0]); mu = 0.5
    pc, pw = proj_circ(z, mu), wrong_proj(z, mu)
    print('[Lemma 2.2] optimality max violation: %.2e (should be <= 0)' % worst_kkt)
    print('            max observed Lipschitz constant: %.6f (<= 1)' % worst_lip)
    print('            counterexample z=(2,0,1), mu=0.5: correct P=%s (dist %.4f),'
          ' old formula gives %s (dist %.4f -- NOT the nearest point)'
          % (np.round(pc,4), np.linalg.norm(pc-z), np.round(pw,4),
             np.linalg.norm(pw-z)))
    return worst_lip <= 1 + 1e-9 and worst_kkt <= 1e-8

def check_jacobian_fd():
    """Lemma 4.6: spectrum of P'_{L_theta} at a boundary point vs finite diff."""
    rng = np.random.default_rng(3); worst = 0.0
    for _ in range(2000):
        mu = rng.uniform(0.2, 0.95)
        # generate a point strictly in the exterior (boundary) region
        xh = rng.normal(size=2); xh /= np.linalg.norm(xh)
        w = rng.uniform(0.5, 2.0); nu = rng.uniform(0.05, 1.0)
        n = np.array([xh[0], xh[1], -mu])/np.sqrt(1+mu**2)
        r = np.array([mu*xh[0], mu*xh[1], 1.0])/np.sqrt(1+mu**2)
        z = w*np.sqrt(1+mu**2)*r + nu*n   # f = w(mu xh,1), z = f + nu n
        f = proj_circ(z, mu); nfx = np.linalg.norm(f[:2])
        rho = np.linalg.norm(z-f); cth = 1.0/np.sqrt(1+mu**2)  # cos(theta)
        if not (np.linalg.norm(z[:2]) > mu*z[2] and
                np.linalg.norm(z[:2]) > -z[2]/mu):
            continue                                  # stay on boundary sheet
        sigma = nfx/(nfx + rho*cth)
        c = np.array([-xh[1], xh[0], 0.0])
        J = np.outer(r,r) + sigma*np.outer(c,c)
        # finite-difference Jacobian
        Jfd = np.zeros((3,3)); h = 1e-7
        for j in range(3):
            dz = np.zeros(3); dz[j] = h
            Jfd[:,j] = (proj_circ(z+dz,mu)-proj_circ(z-dz,mu))/(2*h)
        worst = max(worst, np.linalg.norm(J-Jfd))
    print('[Lemma 4.6] max ||J_exact - J_fd|| over 2000 boundary points: %.2e'
          % worst)
    return worst < 1e-4

# ------------------------------------------------------------------ geometry
def skew(p):
    return np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])

def grasp_cylinder(nf, radius, zoff):
    """Case A: cylindrical part; contact normals horizontal (inward);
    fingers alternate between two height levels +- zoff. Units: cm, N."""
    A = np.zeros((6, 3*nf)); pos = []
    for i in range(nf):
        th = 2*np.pi*i/nf
        e3 = -np.array([np.cos(th), np.sin(th), 0.0])
        e1 = np.array([-np.sin(th), np.cos(th), 0.0]); e2 = np.array([0,0,1.0])
        Q = np.column_stack([e1, e2, e3])
        z  = zoff if (i % 2 == 0) else -zoff
        p  = np.array([radius*np.cos(th), radius*np.sin(th), z])
        pos.append(p)
        A[:, 3*i:3*i+3] = np.vstack([Q, skew(p) @ Q])
    return A, np.array(pos)

def grasp_sphere(nf, R, zlevels):
    """Case B: spherical fruit; contact normals point to the fruit centre."""
    A = np.zeros((6, 3*nf)); pos = []
    for i in range(nf):
        th = 2*np.pi*i/nf
        z = zlevels[i]; r = np.sqrt(max(R**2 - z**2, 1e-12))
        p = np.array([r*np.cos(th), r*np.sin(th), z])
        e3 = -p/np.linalg.norm(p)
        e1 = np.array([-np.sin(th), np.cos(th), 0.0])
        e1 = e1 - (e1 @ e3)*e3; e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        Q = np.column_stack([e1, e2, e3])
        pos.append(p)
        A[:, 3*i:3*i+3] = np.vstack([Q, skew(p) @ Q])
    return A, np.array(pos)

# Case A: industrial assembly, 5 fingers (cylinder of radius 1.6 cm)
MUS_A  = [0.458, 0.481, 0.441, 0.438, 0.392]
WEXT_A = np.array([1.837, -0.605, -4.70, -1.003, -0.546, 0.946])
# Case B: apple harvesting, 3 fingers (sphere of radius 3.75 cm)
MUS_B  = [0.619, 0.592, 0.631]
ZLEV_B = [0.068, -1.326, -2.213]
WEXT_B = np.array([0.75, 0.063, -5.29, -1.806, -0.872, -0.113])

def make_instance(A, mus, wext):
    nf = len(mus)
    def P_L(v):
        out = np.zeros(3*nf)
        for i in range(nf): out[3*i:3*i+3] = proj_circ(v[3*i:3*i+3], mus[i])
        return out
    def F(y):     return -y @ wext + 0.5*np.sum(P_L(-A.T @ y)**2)
    def gradF(y): return -wext - A @ P_L(-A.T @ y)
    return F, gradF, P_L

def solve_dual(A, mus, wext):
    F, gradF, _ = make_instance(A, mus, wext)
    best = None
    for x0 in [np.zeros(6), 0.1*np.ones(6)]:
        r = minimize(F, x0, jac=gradF, method='BFGS',
                     options={'maxiter': 9000, 'gtol': 1e-14})
        if best is None or F(r.x) < F(best.x): best = r
    r2 = root(gradF, best.x, method='krylov', tol=1e-15)   # Newton polish
    if np.linalg.norm(gradF(r2.x)) < np.linalg.norm(gradF(best.x)):
        best = r2
    return best.x, F(best.x)

def jacobian_star(A, mus, ystar):
    """Exact Jacobian J* of P_L at -A^T y*, Hessian stats, margin gamma."""
    nf = len(mus); Jstar = np.zeros((3*nf, 3*nf)); stat = []; gamma = np.inf
    zstar = -A.T @ ystar
    for i in range(nf):
        z = zstar[3*i:3*i+3]; mu = mus[i]; x = z[:2]; s = z[2]
        nx = np.linalg.norm(x)
        if nx <= mu*s - 1e-9:
            stat.append('I'); J = np.eye(3)
            gamma = min(gamma, (mu*s-nx)/np.sqrt(1+mu**2))
        elif nx <= -s/mu + 1e-9:
            stat.append('V'); J = np.zeros((3,3))
            # distance to the polar-cone surface = |mu*nx + s|/sqrt(1+mu^2)
            gamma = min(gamma, -(mu*nx+s)/np.sqrt(1+mu**2))
        else:
            stat.append('B'); f = proj_circ(z, mu); nfx = np.linalg.norm(f[:2])
            rho = np.linalg.norm(z-f); cth = 1.0/np.sqrt(1+mu**2)  # cos(theta)
            sigma = nfx/(nfx + rho*cth); xh = x/nx
            r_ = np.array([mu*xh[0], mu*xh[1], 1.0])/np.sqrt(1+mu**2)
            c_ = np.array([-xh[1], xh[0], 0.0])
            J = np.outer(r_, r_) + sigma*np.outer(c_, c_)
            # margin to leave the boundary region: min(dist to L, dist to polar)
            gamma = min(gamma, rho, np.linalg.norm(f))
        Jstar[3*i:3*i+3, 3*i:3*i+3] = J
    ev = np.linalg.eigvalsh(A @ Jstar @ A.T); evp = ev[ev > 1e-10]
    return Jstar, stat, gamma, (evp.min() if len(evp) else 0.0)

# ------------------------------------------------------------------ aGRAAL
def run_agraal(gradF, y0, y1, lam0, phi, lam_bar, tol=1e-4, Kmax=60000):
    """aGRAAL exactly as in Malitsky (2020), Alg. 1:  lam_1 computed by the
    stepsize rule with theta_0 = 1; update anchored at the extension bar y."""
    rho_ = 1/phi + 1/phi**2
    d0 = gradF(y0); d1 = gradF(y1)
    dy = np.sum((y1-y0)**2); dd = np.sum((d1-d0)**2)
    lam1 = lam_bar if dd == 0 else min(rho_*lam0, phi/(4*lam0)*dy/dd, lam_bar)
    ys    = [y0.copy(), y1.copy()]
    ybars = [y1.copy(), y1.copy()]          # bar y^0 = bar y^1 = y^1
    lams  = [lam0, lam1]; ths = [1.0, phi*lam1/lam0]
    ds    = [d0, d1]
    gevals = 2
    lam_prev, th_prev, ybar_prev, y_prev, d_prev = lam1, ths[1], y1.copy(), y1.copy(), d1
    for k in range(1, Kmax):
        y = ybar_prev - lam_prev*d_prev                    # anchored update
        dy = y - y_prev; d = gradF(y); gevals += 1
        dd = d - d_prev
        lam = min(rho_*lam_prev,
                  phi*th_prev/(4*lam_prev)*np.sum(dy**2)/max(np.sum(dd**2),1e-300),
                  lam_bar)
        th = phi*lam/lam_prev
        ybar = ((phi-1)*y + ybar_prev)/phi
        ys.append(y); ybars.append(ybar); lams.append(lam); ths.append(th); ds.append(d)
        lam_prev, th_prev, ybar_prev, y_prev, d_prev = lam, th, ybar, y, d
        if np.linalg.norm(d) < tol: break
    return ys, ybars, lams, ths, ds, gevals

def run_dgd(gradF, y0, L, tol=1e-4, Kmax=200000):
    y = y0.copy(); ys = [y]; g = gradF(y); ge = 1
    for k in range(Kmax):
        y = y - g/L; ys.append(y); g = gradF(y); ge += 1
        if np.linalg.norm(g) < tol: break
    return ys, ge

def run_fista_bt(F, gradF, y0, L0=1.0, eta=2.0, tol=1e-4, Kmax=60000):
    y = y0.copy(); x = y.copy(); t = 1.0; L = L0; fe = 0; ge = 0; ys = [y]
    for k in range(Kmax):
        g = gradF(x); ge += 1
        while True:
            yn = x - g/L; fe += 1
            if F(yn) <= F(x) + g@(yn-x) + L/2*np.sum((yn-x)**2) + 1e-15: break
            L *= eta
        tnew = (1+np.sqrt(1+4*t*t))/2
        x = yn + (t-1)/tnew*(yn - y)
        y = yn; t = tnew; ys.append(y)
        if np.linalg.norm(gradF(y)) < tol: ge += 1; break
    return ys, ge, fe

# ------------------------------------------- He-style PC baselines (Mu-Zhang)
def run_dmpc(gradF, y0, LF, nu=0.9, gam=1.8, beta0=None, tol=1e-4,
             Kmax=300000, hist=False):
    """DMPC: projection-contraction for the monotone dual equation
    e(y) = grad F(y) = 0 (unconstrained; projection trivial).
    Prediction yb = y - beta e(y); self-adaptive beta with acceptance
    beta ||e(y)-e(yb)|| <= nu ||y-yb||; direction d = (y-yb) - beta(e(y)-e(yb));
    correction y+ = y - gam*alpha*d with alpha = <y-yb,d>/||d||^2.
    beta0 defaults to nu/LF (oracle-tuned, generous to the baseline).
    Returns (iters, grad-evals, y); with hist=True also returns the list
    of iterates (one per outer iteration)."""
    y = y0.copy(); e = gradF(y); ge = 1
    beta = (nu/LF) if beta0 is None else beta0
    ys_rec = [y.copy()]
    for k in range(1, Kmax):
        while True:
            yb = y - beta*e; eb = gradF(yb); ge += 1
            r = y - yb; de = e - eb
            if beta*np.linalg.norm(de) <= nu*np.linalg.norm(r) + 1e-300:
                break
            beta *= 0.5
        d = r - beta*de
        alpha = np.dot(r, d)/np.dot(d, d)
        y = y - gam*alpha*d
        e = gradF(y); ge += 1
        ys_rec.append(y.copy())
        if np.linalg.norm(e) < tol:
            if hist: return k, ge, y, ys_rec
            return k, ge, y
    if hist: return Kmax, ge, y, ys_rec
    return Kmax, ge, y

def run_fopc(A, mus, wext, f0, lam0, LT, nu=0.9, gam=1.8, beta0=None,
             tol=1e-4, Kmax=300000, hist=False):
    """FOPC: projection-contraction for the KKT system
    T(u) = (f + A^T lam, -A f - wext),  u = (f, lam) in L_Theta x R^6.
    Prediction ub = P_Omega(u - beta T(u)); same acceptance test;
    direction d = (u-ub) - beta(T(u)-T(ub)); u+ = u - gam*alpha*d.
    Stops on the natural residual ||u - P_Omega(u - T(u))|| < tol.
    beta0 defaults to nu/||M|| (oracle-tuned, generous to the baseline).
    Returns (iters, T-evals, u); with hist=True also returns the list of
    dual parts lambda^k (one per outer iteration)."""
    nf = len(mus)
    def PL(v):
        out = np.zeros(3*nf)
        for i in range(nf):
            out[3*i:3*i+3] = proj_circ(v[3*i:3*i+3], mus[i])
        return out
    def T(u):
        f, lam = u[:3*nf], u[3*nf:]
        return np.concatenate([f + A.T @ lam, -A @ f - wext])
    def PO(u):
        return np.concatenate([PL(u[:3*nf]), u[3*nf:]])
    u = np.concatenate([f0, lam0]); Tu = T(u); Te = 1
    beta = (nu/LT) if beta0 is None else beta0
    lam_rec = [u[3*nf:].copy()]
    for k in range(1, Kmax):
        while True:
            ub = PO(u - beta*Tu); Tub = T(ub); Te += 1
            r = u - ub; dT = Tu - Tub
            if beta*np.linalg.norm(dT) <= nu*np.linalg.norm(r) + 1e-300:
                break
            beta *= 0.5
        d = r - beta*dT
        alpha = np.dot(r, d)/np.dot(d, d)
        u = u - gam*alpha*d
        Tu = T(u); Te += 1
        lam_rec.append(u[3*nf:].copy())
        if np.linalg.norm(u - PO(u - Tu)) < tol:
            if hist: return k, Te, u, lam_rec
            return k, Te, u
    if hist: return Kmax, Te, u, lam_rec
    return Kmax, Te, u

# --------------------------------------------- explicit constants (Sec. 4)
def exact_hessian(A, mus, y):
    """Exact Hessian nabla^2 F(y) = A J(y) A^T at points where all contact
    statuses are region-interior (piecewise-smooth sheets)."""
    nf = len(mus); J = np.zeros((3*nf, 3*nf)); z = -A.T @ y
    for i in range(nf):
        zi = z[3*i:3*i+3]; mu = mus[i]; x = zi[:2]; s = zi[2]
        nx = np.linalg.norm(x)
        if nx <= mu*s: Ji = np.eye(3)
        elif nx <= -s/mu: Ji = np.zeros((3, 3))
        else:
            xh = x/nx; w = (mu*nx+s)/(1+mu**2); sig = mu*w/nx
            r_ = np.array([mu*xh[0], mu*xh[1], 1.0])/np.sqrt(1+mu**2)
            c_ = np.array([-xh[1], xh[0], 0.0])
            Ji = np.outer(r_, r_) + sig*np.outer(c_, c_)
        J[3*i:3*i+3, 3*i:3*i+3] = Ji
    return A @ J @ A.T

def explicit_constants(A, mus, wext, ystar, V2, phi, lam0, lam_bar, name,
                       nmc=3000):
    """Explicit local-regularity and quadratic-growth constants of
    Lemma 4.13/4.14 (revised):  L_H, delta_nu, mu_QG; all verified
    numerically on grids/trajectories."""
    nf = len(mus); LF = np.linalg.norm(A, 2)**2; nA = np.linalg.norm(A, 2)
    Jstar, stat, gamma, muT = jacobian_star(A, mus, ystar)
    zstar = -A.T @ ystar
    nAi = [np.linalg.norm(A[:, 3*i:3*i+3], 2) for i in range(nf)]
    LH = 0.0; per = []
    for i in range(nf):
        if stat[i] != 'B':
            continue
        z = zstar[3*i:3*i+3]; mu = mus[i]; f = proj_circ(z, mu)
        w = f[2]; rho = np.linalg.norm(z-f); cth = 1.0/np.sqrt(1+mu**2)
        gi = min(rho, np.linalg.norm(f))          # per-contact margin
        nxz = mu*w + rho*cth                       # ||x_{z_i^*}||
        chi = nxz - gi/2.0                         # >= gi/2 > 0
        # rigorously derived Lipschitz bound of the boundary-sheet Jacobian
        # on B(z_i^*, gamma_i/2):  J = r r^T + g C,  g = mu^2/(1+mu^2)
        #   + (mu/(1+mu^2))(s/||x||),  ||g|| <= 1 on the sheet
        ell = ((2*mu*np.sqrt(1+mu**2) + max(1.0, mu)*(1+mu))
               / ((1+mu**2)*chi)) + 2.0/chi
        # Hessian Lipschitz constant: nabla^2 F = sum_i A_i J_i A_i^T,
        # so ||d nabla^2 F|| <= sum_i ||A_i||^2 * ell_i * ||A_i|| ||dy||
        LH += nAi[i]**3 * ell
        per.append((i, w, rho, gi, chi, ell))
    rho_ = 1/phi + 1/phi**2
    lam_under = min(rho_*lam0, phi/(4*lam0*LF**2), lam_bar,
                    phi**2/(4*lam_bar*LF**2))
    th_minus = phi*lam_under/max(lam0, lam_bar)
    R = (np.sqrt((phi-1)/phi) + 1/np.sqrt(th_minus))*np.sqrt(V2)
    d_half = min(gamma/(2*nA), 0.5*muT/LH)
    muQG = (muT/2)*min(1.0, d_half/R)
    # asymptotic stepsize constants (sharpened rate, Corollary 4.18)
    m_ = min(lam_bar, phi**2/(4*lam0*LF**2), phi**2/(4*lam_bar*LF**2))
    lam1_lb = min(rho_*lam0, phi/(4*lam0*LF**2), lam_bar)
    Klam = 1 if lam1_lb >= m_ else 1+int(np.ceil(np.log(m_/lam1_lb)
                                                 /np.log(rho_)))
    th_asy = phi*m_/max(lam0, lam_bar)
    # numerical verification
    rng = np.random.default_rng(2024)
    ok_J = True; ok_H = True; ok_QG = True
    d_fr = gamma/(2*nA)
    for (i, w, rho, gi, chi, ell) in per:
        mu = mus[i]; zi = zstar[3*i:3*i+3]; worst = 0.0
        for _ in range(800):
            v1 = rng.normal(size=3)
            z1 = zi + v1/np.linalg.norm(v1)*rng.uniform()*d_fr
            v2 = rng.normal(size=3)
            z2 = zi + v2/np.linalg.norm(v2)*rng.uniform()*d_fr
            J1 = J_block(z1, mu); J2 = J_block(z2, mu)
            dz = np.linalg.norm(z1-z2)
            if dz > 1e-14:
                worst = max(worst, np.linalg.norm(J1-J2)/dz)
        if worst > ell: ok_J = False
    d_nu = min(gamma/(2*nA), 0.1*muT/LH)
    mineig = np.inf
    for _ in range(300):
        v = rng.normal(size=6); y = ystar + d_nu*v/np.linalg.norm(v)*rng.uniform()**(1/6)
        mineig = min(mineig, np.linalg.eigvalsh(exact_hessian(A, mus, y))[0])
    if mineig < 0.9*muT - 1e-12: ok_H = False
    F, gradF, P_L = make_instance(A, mus, wext)
    Fstar = F(ystar)
    nchecked = 0
    for _ in range(nmc):
        v = rng.normal(size=6); y = ystar + R*v/np.linalg.norm(v)*rng.uniform()**(1/6)
        t = np.linalg.norm(y-ystar)
        if t < 1e-4: continue     # avoid round-off in F(y)-F* near y*
        nchecked += 1
        if (F(y)-Fstar) < muQG/2*t**2 - 1e-12: ok_QG = False
    print('[%s] explicit constants: L_H = %.2f, R = %.3f, gamma/(2||A||) = %.3e'
          % (name, LH, R, gamma/(2*nA)))
    print('   per-boundary-contact (i, w*, rho, gamma_i, chi_i, ell_i):')
    for p in per:
        print('     i=%d  w*=%.4f  rho=%.4f  g_i=%.4f  chi=%.4f  ell=%.3f' % p)
    print('   delta_0 = %.4e,  delta_(1/2) = %.4e,  delta_(0.1) = %.4e'
          % (d_fr, d_half, d_nu))
    print('   mu_QG (explicit) = %.4e   (mu_T/2 = %.4e)' % (muQG, muT/2))
    print('   asymptotic stepsize: m = %.6f (= %.4f/L_F),  K_lam = %d, '
          'th_asy = %.4f' % (m_, m_*LF, Klam, th_asy))
    print('   checks: Lip(J_i) <= ell_i: %s;  mineig nabla^2F >= 0.9 mu_T on '
          'B(y*,d_0.1): %s (%.5f vs %.5f);  explicit QG on %d samples: %s'
          % (ok_J, ok_H, mineig, 0.9*muT, nchecked, ok_QG))
    return dict(LH=LH, R=R, d_fr=d_fr, d_half=d_half, d_nu=d_nu, muQG=muQG,
                m=m_, Klam=Klam, th_asy=th_asy, ok=ok_J and ok_H and ok_QG)

def J_block(z, mu):
    x = z[:2]; s = z[2]; nx = np.linalg.norm(x)
    if nx <= mu*s: return np.eye(3)
    if nx <= -s/mu: return np.zeros((3, 3))
    xh = x/nx; w = (mu*nx+s)/(1+mu**2); sig = mu*w/nx
    r_ = np.array([mu*xh[0], mu*xh[1], 1.0])/np.sqrt(1+mu**2)
    c_ = np.array([-xh[1], xh[0], 0.0])
    return np.outer(r_, r_) + sig*np.outer(c_, c_)

# ------------------------------------------------------- theory verification
def verify_theory(A, mus, wext, ystar, Fstar, muT, LF, phi, lam0, lam_bar, name):
    F, gradF, P_L = make_instance(A, mus, wext)
    nf = len(mus)
    ys, ybars, lams, ths, ds, _ = run_agraal(gradF, np.zeros(6),
                                             0.05*np.ones(6), lam0, phi,
                                             lam_bar, tol=1e-12)
    N = len(ys)
    # (i) stepsize inequality (4.1) from k = 1 on
    v24 = 0
    for k in range(1, N-1):
        dy = ys[k+1]-ys[k]; dd = ds[k+1]-ds[k]
        if lams[k+1]**2*np.sum(dd**2) > ths[k+1]*ths[k]/4*np.sum(dy**2)*(1+1e-9):
            v24 += 1
    th_ok = all(0 < t <= 1+1/phi+1e-12 for t in ths[1:])
    # (ii) energy inequality (4.6), sharp form, valid from k = 2 on
    # (at k = 1 the initialisation bar y^0 = y^1 breaks the prox-inequality;
    #  cf. Malitsky (2020), Theorem 2, eq. (35), where the sum starts at i = 2)
    def V(k): return phi/(phi-1)*np.sum((ybars[k]-ystar)**2) \
                    + ths[k-1]/2*np.sum((ys[k]-ys[k-1])**2)
    Vs = np.array([V(k) for k in range(1, N-1)])
    v34 = 0
    for k in range(2, N-2):
        lhs = Vs[k] + 2*lams[k]*np.dot(ds[k], ys[k]-ystar) \
              + ths[k]*np.sum((ys[k]-ybars[k])**2)
        if lhs > Vs[k-1]*(1+1e-12) + 1e-12: v34 += 1
    # the same inequality at k = 1 is NOT guaranteed (and fails in practice):
    # quantify the observed violation V_2 + 2*lam_1<d^1,e^1> + th_1|u^1|^2 - V_1
    viol_k1 = float(Vs[1] + 2*lams[1]*np.dot(ds[1], ys[1]-ystar)
                    + ths[1]*np.sum((ys[1]-ybars[1])**2) - Vs[0])
    # (iii) ergodic bound:  2 (sum_{k>=2} lam_k)(F(Y^k)-F*) <= V_2
    lw = np.array(lams[2:N]); yw = np.array(ys[2:N])
    Yk = np.average(yw, axis=0, weights=lw)
    erg_lhs = 2*lw.sum()*(F(Yk)-Fstar); erg_rhs = Vs[1]
    # (iv) unconditional stepsize lower bound (Lemma 4.2)
    rho_ = 1/phi + 1/phi**2
    lam_under = min(rho_*lam0, phi/(4*lam0*LF**2), lam_bar,
                    phi**2/(4*lam_bar*LF**2))
    lam_min_obs = np.nanmin(np.array(lams[1:]))
    # (v) Bregman identity (Lemma 4.4)
    fstar = P_L(-A.T @ ystar)
    rng = np.random.default_rng(1); worst = 0.0; worst_neg = np.inf
    for _ in range(5000):
        y = ystar + rng.normal(size=6)
        z = -A.T @ y; p = P_L(z)
        resid = (F(y)-Fstar) - (0.5*np.sum((p-fstar)**2))
        rem = -np.dot(fstar, z - p)
        worst = max(worst, abs(resid - rem)); worst_neg = min(worst_neg, rem)
    # (vi) identification
    def sig(y):
        v = -A.T @ y; out = []
        for i in range(nf):
            z = v[3*i:3*i+3]; x = z[:2]; s = z[2]; nx = np.linalg.norm(x)
            out.append('I' if nx <= mus[i]*s-1e-9
                       else ('V' if nx <= -s/mus[i]+1e-9 else 'B'))
        return tuple(out)
    sigstar = sig(ystar); Kid = None
    for k in range(N):
        if all(sig(ys[j]) == sigstar for j in range(k, N)): Kid = k; break
    # (vii) a-posteriori certified contraction (Theorem 4.8)
    lam_t = np.array(lams[Kid:]); th_t = np.array(ths[Kid:])
    lam_m = np.nanmin(lam_t); lam_p = np.nanmax(lam_t)
    th_m = np.nanmin(th_t)
    # strong-monotonicity modulus verified along the tail of the trajectory
    mu_traj = np.inf
    for k in range(Kid, N-1):
        e = ys[k]-ystar; ne2 = np.sum(e**2)
        if ne2 > 1e-24:
            mu_traj = min(mu_traj, np.dot(ds[k], e)/ne2)
    mu_cert = min(muT, mu_traj)
    beta = (phi-1)/phi; s_p = lam_p*LF
    a_c = 2*phi/(phi-1) + 1 + 1/phi
    b_c = 2*phi/(phi-1)*(1+beta*s_p)**2 + (1+1/phi)*s_p**2
    eta = min(th_m/a_c, 2*lam_m*mu_cert/b_c)
    Vt = Vs[Kid-1:]
    ratios = Vt[1:]/np.maximum(Vt[:-1], 1e-300)
    mask = np.array([np.linalg.norm(ds[Kid+j]) > 1e-9 for j in range(len(ratios))])
    mx = np.nanmax(ratios[mask & np.isfinite(ratios)])
    r_ = np.array([np.linalg.norm(ds[k]) for k in range(1, N)])
    kk = np.arange(Kid+30, N-1)
    emp = np.exp(np.polyfit(kk, np.log(r_[kk-1]), 1)[0])
    # a-priori (unconditional) constants of Theorem 4.8
    th_minus = phi*lam_under/lam_bar; t_minus = lam_under*muT/2
    s_plus = lam_bar*LF
    b_c0 = 2*phi/(phi-1)*(1+beta*s_plus)**2 + (1+1/phi)*s_plus**2
    eta0 = min(th_minus/a_c, 2*t_minus/b_c0)
    # (viii) explicit local constants (Lemma 4.13/4.14) and the sharpened
    # a-priori rate (Corollary 4.18): stepsize floor m for k >= K_lam,
    # theta_asy = phi*m/lam_bar, explicit quadratic growth mu_QG.
    V2 = float(Vs[1])
    ec = explicit_constants(A, mus, wext, ystar, V2, phi, lam0, lam_bar,
                            name)
    eta_asy = min(ec['th_asy']/a_c, 2*ec['m']*0.9*muT/b_c0)
    # global a-priori rate via explicit QG: convexity gives
    # <grad F(y), y-y*> >= F(y)-F* >= (mu_QG/2)||y-y*||^2 on S
    eta_glob = min(ec['th_asy']/a_c, ec['m']*ec['muQG']/b_c0)
    eps_half = (ec['d_half']/(2*(np.sqrt((phi-1)/phi)
                                 + 1/np.sqrt(ec['th_asy']))))**2
    Kid_bound = ec['Klam'] + int(np.ceil(
        np.log(V2/eps_half)/np.log(1+eta_glob)))
    print('  sharpened a-priori rate (Cor. 4.18): eta_asy(nu=0.1) = %.3e,'
          '  1/(1+eta_asy) = %.6f,  eta_asy*kappa_T = %.4f'
          % (eta_asy, 1/(1+eta_asy), eta_asy*LF/muT))
    print('  explicit identification bound: K_id <= %.3e  (observed %d)'
          % (Kid_bound, Kid))
    print('[%s]  (tol = 1e-12, lam0 = %g, lam_bar = %g)' % (name, lam0, lam_bar))
    print('  [Lemma 4.1] stepsize-inequality violations: %d    '
          'theta_k in (0,1+1/phi]: %s    [Lemma 4.3] energy violations: %d'
          % (v24, th_ok, v34))
    print('  [energy at k = 1] observed violation of the (invalid) k=1 case: '
          '%.3e' % viol_k1)
    print('  ergodic: 2*sum_{k>=2}(lam)*(F(Y)-F*) = %.4f <= V_2 = %.4f   OK=%s'
          % (erg_lhs, erg_rhs, erg_lhs <= erg_rhs*(1+1e-12)+1e-12))
    print('  stepsize bound: lam_min theory %.6f <= observed %.6f   OK=%s'
          % (lam_under, lam_min_obs, lam_under <= lam_min_obs + 1e-12))
    print('  Bregman identity max residual %.2e;  min cross term %.2e (>= 0)'
          % (worst, worst_neg))
    print('  identification at K_id = %d  (status %s)' % (Kid, sigstar))
    print('  a-posteriori certificate: lam_m = %.6f, th_m = %.4f, '
          'mu_traj = %.4f, mu_cert = %.4f' % (lam_m, th_m, mu_traj, mu_cert))
    print('  eta = %.3e,  certified V-ratio 1/(1+eta) = %.6f,  '
          'observed max %.6f   OK=%s' % (eta, 1/(1+eta), mx,
                                         mx <= 1/(1+eta)+1e-9))
    print('  a-priori eta_0 = %.3e (unconditional);  eta*kappa_T = %.3f;  '
          'empirical R-factor %.5f' % (eta0, eta*LF/muT, emp))
    return dict(v24=v24, v34=v34, Kid=Kid, eta=eta, eta0=eta0,
                lam_under=lam_under, lam_min_obs=lam_min_obs,
                mu_traj=mu_traj, mu_cert=mu_cert, emp=emp,
                erg_ok=bool(erg_lhs <= erg_rhs*(1+1e-12)+1e-12),
                contract_ok=bool(mx <= 1/(1+eta)+1e-9),
                eta_asy=eta_asy, eta_glob=eta_glob, Kid_bound=Kid_bound,
                ec=ec, Vs=Vs, ys=ys, ds=ds, lams=lams)

def check_sensitivity_identity():
    """Lemma 6.1: ||d/d mu P_{L_mu}(z)||^2 = ||z||^2/(1+mu^2)^2 on the
    boundary sheet.  Verified symbolically (sympy) and by finite differences."""
    import sympy as sp
    mu, r, s = sp.symbols('mu r s', positive=True)
    w = (mu*r + s)/(1 + mu**2); g = mu*w
    expr = sp.simplify(sp.diff(g, mu)**2 + sp.diff(w, mu)**2
                       - (r**2 + s**2)/(1 + mu**2)**2)
    sym_res = sp.simplify(expr)
    rng = np.random.default_rng(23); worst = 0.0
    for _ in range(2000):
        mu_ = rng.uniform(0.2, 0.95)
        xh = rng.normal(size=2); xh /= np.linalg.norm(xh)
        w_ = rng.uniform(0.5, 2.0); nu = rng.uniform(0.05, 1.0)
        nn = np.array([xh[0], xh[1], -mu_])/np.sqrt(1+mu_**2)
        rvec = np.array([mu_*xh[0], mu_*xh[1], 1.0])/np.sqrt(1+mu_**2)
        z = w_*np.sqrt(1+mu_**2)*rvec + nu*nn   # boundary-region point
        h = 1e-7
        dP = (proj_circ(z, mu_+h) - proj_circ(z, mu_-h))/(2*h)
        lhs = np.sum(dP**2); rhs = np.sum(z**2)/(1+mu_**2)**2
        worst = max(worst, abs(lhs-rhs)/max(rhs, 1e-300))
    print('[Lemma 6.1] sensitivity identity: symbolic residual = %s, '
          'FD residual = %.2e' % (sym_res, worst))
    return sym_res == 0 and worst < 1e-6

def check_robustness_constant():
    rng = np.random.default_rng(7); Cmax = 0.0
    for _ in range(200000):
        z = rng.normal(size=3)
        mu1 = rng.uniform(0.2, 1.0); mu2 = rng.uniform(0.2, 1.0)
        d = np.linalg.norm(proj_circ(z,mu1)-proj_circ(z,mu2)) \
            / (abs(mu1-mu2)*np.linalg.norm(z))
        Cmax = max(Cmax, d)
    print('[Lemma 6.1] empirical C_mu = %.4f <= 1/(1+mu_min^2) = %.4f'
          % (Cmax, 1/1.04))
    return Cmax <= 1/1.04 + 1e-9

# ------------------------------------------------------------ numerics: core
import time

def cold_start(A, mus, wext, LF, phi, name, tol=1e-4, repeats=5):
    """Cold-start comparison: aGRAAL (parameter-free and capped),
    DGD with oracle stepsize 1/L_F, FISTA with backtracking,
    DMPC (PC method on the dual equation) and FOPC (PC method on the
    KKT system), both with oracle-tuned initial step nu/L (generous)."""
    F, gradF, P_L = make_instance(A, mus, wext)
    nf = len(mus)
    Mop = np.block([[np.eye(3*nf), A.T], [-A, np.zeros((6, 6))]])
    LT = np.linalg.norm(Mop, 2)
    y0 = np.zeros(6); y1 = 0.05*np.ones(6)
    f0 = np.zeros(3*nf); lam0 = np.zeros(6)
    rows = []
    def timeit(fn):
        ts = []
        out = None
        for _ in range(repeats):
            t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter()-t0)
            out = r
        return out, np.median(ts)*1e3   # ms
    out, ms = timeit(lambda: run_agraal(gradF, y0, y1, 1.0, phi, 100.0, tol))
    rows.append(('aGRAAL-PF (lam0=1, lam_bar=100)', len(out[0])-2, out[5], 0, ms))
    out, ms = timeit(lambda: run_agraal(gradF, y0, y1, 1/LF, phi, 1/LF, tol))
    rows.append(('aGRAAL-capped (lam0=lam_bar=1/L_F)', len(out[0])-2, out[5], 0, ms))
    out, ms = timeit(lambda: run_dgd(gradF, y0, LF, tol))
    rows.append(('DGD oracle 1/L_F', len(out[0])-1, out[1], 0, ms))
    out, ms = timeit(lambda: run_fista_bt(F, gradF, y0, 1.0, 2.0, tol))
    rows.append(('FISTA backtracking', len(out[0])-1, out[1], out[2], ms))
    out, ms = timeit(lambda: run_dmpc(gradF, y0, LF, tol=tol))
    rows.append(('DMPC (dual eq., oracle beta0=nu/L_F)', out[0], out[1], 0, ms))
    out, ms = timeit(lambda: run_fopc(A, mus, wext, f0, lam0, LT, tol=tol))
    rows.append(('FOPC (KKT system, oracle beta0=nu/||M||)', out[0], out[1], 0, ms))
    print('\n[%s] cold start, tol = %g  (Python 3.12 prototype timings, ms)'
          % (name, tol))
    print('  ||M|| (KKT operator norm) = %.6f' % LT)
    for r in rows:
        print('  %-38s iters %6d   grad/T-evals %6d   func-evals %5d   %8.1f ms'
              % r)
    return rows

def warm_trajectory(A, mus, wext, LF, phi, name, T=500, tol=1e-4, tol2=1e-3):
    """Warm-started tracking of a slowly oscillating wrench
    (peak relative change 0.19% per step, 15% peak amplitude).
    Compares aGRAAL-PF with DMPC and FOPC on the identical wrench
    sequence; DMPC/FOPC restart from their oracle beta0 each step
    (generous to the baselines).
    Returns dict of (mean iters at tol, mean iters at tol2, median ms)."""
    rng = np.random.default_rng(11)
    u = rng.normal(size=6); u = u/np.linalg.norm(u)
    nf = len(mus)
    Mop = np.block([[np.eye(3*nf), A.T], [-A, np.zeros((6, 6))]])
    LT = np.linalg.norm(Mop, 2)
    wrenches = [wext + 0.15*np.linalg.norm(wext)*np.sin(2*np.pi*t/T)*u
                for t in range(T)]
    results = {}

    def sweep(step_fn, tol_):
        its = []; ts = []
        state = None
        for t in range(T):
            F, gradF, _ = make_instance(A, mus, wrenches[t])
            t0 = time.perf_counter()
            k, state = step_fn(F, gradF, wrenches[t], state, tol_)
            ts.append((time.perf_counter()-t0)*1e3)
            its.append(k)
        return np.mean(its), float(np.median(ts)), max(its)

    def step_agraal(F, gradF, w, state, tol_):
        yprev, lam_prev = state if state is not None else (np.zeros(6), 1/LF)
        ys, ybars, lams, ths, ds, _ = run_agraal(gradF, yprev, yprev,
                                                 lam_prev, phi, 100.0, tol_)
        return len(ys)-2, (ys[-1], lams[-1])

    def step_dmpc(F, gradF, w, state, tol_):
        yprev = state if state is not None else np.zeros(6)
        k, ge, y = run_dmpc(gradF, yprev, LF, tol=tol_)
        return ge, y            # count oracle calls (gradient evaluations)

    def step_fopc(F, gradF, w, state, tol_):
        uprev = state if state is not None else np.zeros(3*nf+6)
        k, Te, u_ = run_fopc(A, mus, w, uprev[:3*nf], uprev[3*nf:], LT,
                             tol=tol_)
        return Te, u_           # count oracle calls (T evaluations)

    for tag, fn in [('aGRAAL-PF', step_agraal), ('DMPC', step_dmpc),
                    ('FOPC', step_fopc)]:
        m1, ms1, M1 = sweep(fn, tol)
        m2, ms2, M2 = sweep(fn, tol2)
        results[tag] = (m1, m2, ms1, ms2)
        print('[%s] warm trajectory, T = %d, %-9s: mean iters/step %.1f '
              '(tol %g), %.1f (tol %g); max %d / %d; median wall %.3f / '
              '%.3f ms' % (name, T, tag, m1, tol, m2, tol2, M1, M2, ms1, ms2))
    return results

def misestimation(A, mus, wext, LF, phi, name, tol=1e-4):
    """Effect of overestimating the Lipschitz constant: the PC baselines
    DMPC/FOPC with beta0 = nu/L_est degrade (almost linearly) in the
    overestimation factor, while aGRAAL-PF uses no Lipschitz input."""
    F, gradF, _ = make_instance(A, mus, wext)
    nf = len(mus)
    Mop = np.block([[np.eye(3*nf), A.T], [-A, np.zeros((6, 6))]])
    LT = np.linalg.norm(Mop, 2)
    y0 = np.zeros(6); y1 = 0.05*np.ones(6)
    out, _ = None, None
    ys = run_agraal(gradF, y0, y1, 1.0, phi, 100.0, tol)
    print('[%s] L-misestimation (tol = %g): aGRAAL-PF iters %d '
          '(no Lipschitz input)' % (name, tol, len(ys[0])-2))
    for c in [1, 3, 10]:
        kd, ged, _ = run_dmpc(gradF, y0, c*LF, tol=tol)
        kf, Tef, _ = run_fopc(A, mus, wext, np.zeros(3*nf), np.zeros(6),
                              c*LT, tol=tol)
        print('   L_est = %2d x L:  DMPC iters %6d (ge %6d),  '
              'FOPC iters %6d (Te %6d)' % (c, kd, ged, kf, Tef))

def robustness(casename, A, mus, wext, make_geo, geo):
    """Re-solve perturbed problems; report relative optimal-force error."""
    _, _, P_L = make_instance(A, mus, wext)
    ystar, _ = solve_dual(A, mus, wext); fstar = P_L(-A.T @ ystar)
    nf = len(mus)
    print('[%s] robustness (relative change of optimal force, re-solved):'
          % casename)
    for desc, mus_p, geo_p in [
            ('friction +10%%', [m*1.1 for m in mus], scale_geo(geo, 1.0)),
            ('friction -10%%', [m*0.9 for m in mus], scale_geo(geo, 1.0)),
            ('geometry  +1%%', mus, scale_geo(geo, 1.01)),
            ('geometry  -1%%', mus, scale_geo(geo, 0.99))]:
        Ap, _ = make_geo(nf, *geo_p)
        _, _, P_Lp = make_instance(Ap, mus_p, wext)
        yp, _ = solve_dual(Ap, mus_p, wext); fp = P_Lp(-Ap.T @ yp)
        print('    %-16s  ||f*-f*_p||/||f*|| = %.2f%%'
              % (desc, 100*np.linalg.norm(fstar-fp)/np.linalg.norm(fstar)))

def scale_geo(args, factor):
    # scale all geometric parameters (radius/zoff or R/zlevels) by `factor`
    return [a*factor if np.isscalar(a) else [v*factor for v in a]
            for a in args]

def phi_sensitivity(A, mus, wext, name, tol=1e-4):
    _, gradF, _ = make_instance(A, mus, wext)
    y0 = np.zeros(6); y1 = 0.05*np.ones(6)
    print('[%s] phi-sensitivity (aGRAAL-PF, cold start, tol %g):' % (name, tol))
    for phiv in [1.2, 1.4, 1.5, (1+np.sqrt(5))/2]:
        ys = run_agraal(gradF, y0, y1, 1.0, phiv, 100.0, tol)
        print('    phi = %.3f : %d iterations' % (phiv, len(ys[0])-2))

# ------------------------------------------------------------------ figures
def make_figures(A_A, A_B, phi, LF_A, LF_B, resA, resB, outdir='.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, A, mus, wext, LF in [('A', A_A, MUS_A, WEXT_A, LF_A),
                                   ('B', A_B, MUS_B, WEXT_B, LF_B)]:
        F, gradF, _ = make_instance(A, mus, wext)
        ystar, Fstar = solve_dual(A, mus, wext)
        nf = len(mus)
        Mop = np.block([[np.eye(3*nf), A.T], [-A, np.zeros((6, 6))]])
        LTf = np.linalg.norm(Mop, 2)
        y0 = np.zeros(6); y1 = 0.05*np.ones(6)
        ag = run_agraal(gradF, y0, y1, 1.0, phi, 100.0, 1e-10)
        dg, _ = run_dgd(gradF, y0, LF, 1e-10)
        fi, _, _ = run_fista_bt(F, gradF, y0, 1.0, 2.0, 1e-10)
        _, _, _, dm = run_dmpc(gradF, y0, LF, tol=1e-10, hist=True)
        _, _, _, fo = run_fopc(A, mus, wext, np.zeros(3*nf), np.zeros(6),
                               LTf, tol=1e-10, hist=True)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ys_, lab, sty in [(ag[0], 'aGRAAL-PF', 'b-'),
                              (dg, 'DGD oracle $1/L_F$', 'g--'),
                              (fi, 'FISTA backtracking', 'r-.'),
                              (dm, 'DMPC (dual eq.)', 'm:'),
                              (fo, 'FOPC (KKT)', 'c--')]:
            gap = [max(F(y)-Fstar, 1e-17) for y in ys_]
            ax[0].loglog(np.arange(len(gap)), gap, sty, label=lab,
                         markevery=max(len(gap)//12,1), lw=1.4)
        kk = np.arange(10, 2000)
        ax[0].loglog(kk, gap[5]*(10/kk)**1.0, 'k:', lw=1.0,
                     label='$O(1/k)$ reference')
        ax[0].set_xlabel('iteration $k$'); ax[0].set_ylabel('$F(y^k)-F^*$')
        ax[0].legend(fontsize=8); ax[0].grid(True, which='both', alpha=0.3)
        res = [np.linalg.norm(gradF(y)) for y in ag[0]]
        ax[1].semilogy(np.arange(len(res)), res, 'b-', lw=1.2)
        # identification time of the plotted (parameter-free) trajectory
        mus_ = MUS_A if name == 'A' else MUS_B
        def sig_(y):
            v = -A.T @ y; out = []
            for i in range(len(mus_)):
                zz = v[3*i:3*i+3]; nx = np.linalg.norm(zz[:2])
                out.append('I' if nx <= mus_[i]*zz[2]-1e-9
                           else ('V' if nx <= -zz[2]/mus_[i]+1e-9 else 'B'))
            return tuple(out)
        sigs = sig_(ystar); Kid = next(
            (k for k in range(len(ag[0]))
             if all(sig_(ag[0][j]) == sigs for j in range(k, len(ag[0])))),
            len(ag[0]))
        print('[figures] Case %s: identification of plotted PF run at k = %d'
              % (name, Kid))
        ax[1].axvline(Kid, color='magenta', ls='--', lw=1.2,
                      label='identification $K_{\\rm id}=%d$' % Kid)
        ax[1].set_xlabel('iteration $k$')
        ax[1].set_ylabel('$\\|\\nabla F(y^k)\\|$')
        ax[1].legend(fontsize=9); ax[1].grid(True, which='both', alpha=0.3)
        fig.suptitle('Case %s' % name)
        fig.tight_layout()
        fig.savefig('%s/fig_case%s_convergence.png' % (outdir, name), dpi=180)
        plt.close(fig)
    # per-gradient-evaluation cost vs number of fingers: use the compiled-C
    # measurements (c_periter.dat, produced by gfo_solvers.c) when available;
    # otherwise fall back to Python prototype timings (labeled as such).
    nfs = [3, 5, 8, 12, 20, 30]
    compiled = None
    try:
        with open('%s/c_periter.dat' % outdir) as fh:
            rows = [l.split() for l in fh if l.strip() and not
                    l.startswith('#')]
            compiled = {int(r[0]): (float(r[1]), float(r[3]))
                        for r in rows}
    except OSError:
        compiled = None
    if compiled is not None:
        cost = [compiled[n][0] for n in nfs]
        costT = [compiled[n][1] for n in nfs]
        src = 'compiled C'
    else:
        cost = []; costT = None
        rng = np.random.default_rng(5)
        for nf in nfs:
            th = np.linspace(0, 2*np.pi, nf, endpoint=False)
            A = np.zeros((6, 3*nf))
            for i in range(nf):
                e3 = -np.array([np.cos(th[i]), np.sin(th[i]), 0.0])
                e1 = np.array([-np.sin(th[i]), np.cos(th[i]), 0.0])
                e2 = np.array([0,0,1.0]); Q = np.column_stack([e1,e2,e3])
                p = np.array([np.cos(th[i]), np.sin(th[i]),
                              0.3*rng.normal()])
                A[:,3*i:3*i+3] = np.vstack([Q, skew(p)@Q])
            mus = [0.4]*nf; wext = rng.normal(size=6)
            _, gradF, _ = make_instance(A, mus, wext)
            y = rng.normal(size=6)
            gradF(y)  # warmup
            t0 = time.perf_counter()
            for _ in range(2000): gradF(y)
            cost.append((time.perf_counter()-t0)/2000*1e6)   # microseconds
        src = 'Python prototype'
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(nfs, cost, 'bs-', lw=1.4, label='aGRAAL gradient evaluation')
    if costT is not None:
        ax.plot(nfs, costT, 'rs--', lw=1.4, label='FOPC $T$-evaluation')
        ax.legend(fontsize=9)
    ax.set_xlabel('number of fingers $n_f$')
    ax.set_ylabel(r'wall time per oracle evaluation ($\mu$s, %s)' % src)
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig('%s/fig_scalability.png' % outdir, dpi=180); plt.close(fig)
    print('[figures] per-oracle-evaluation cost (%s): ' % src +
          ', '.join('nf=%d: %.1f us' % (n, c) for n, c in zip(nfs, cost)))

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    ok = check_projection() & check_jacobian_fd() & check_sensitivity_identity()
    phi = 1.5
    A_A, POS_A = grasp_cylinder(5, 1.6, 2.0)
    A_B, POS_B = grasp_sphere(3, 3.75, ZLEV_B)
    LF_A = np.linalg.norm(A_A, 2)**2; LF_B = np.linalg.norm(A_B, 2)**2
    res = {}
    for name, A, mus, wext, LF, POS in [
            ('Case A', A_A, MUS_A, WEXT_A, LF_A, POS_A),
            ('Case B', A_B, MUS_B, WEXT_B, LF_B, POS_B)]:
        ystar, Fstar = solve_dual(A, mus, wext)
        Fres, Gres, _ = make_instance(A, mus, wext)
        print('  reference solve: ||grad F(y*)|| = %.2e' %
              np.linalg.norm(Gres(ystar)))
        Jstar, stat, gamma, muT = jacobian_star(A, mus, ystar)
        sA = np.linalg.svd(A, compute_uv=False)
        print('\n%s: ||A|| = %.4f, kappa_2(A) = %.2f, L_F = ||A||^2 = %.4f,'
              ' mu_T = %.4f, kappa_T = %.1f, gamma = %.4f, status = %s'
              % (name, np.linalg.norm(A,2), sA[0]/sA[-1], LF, muT, LF/muT,
                 gamma, stat))
        print('  contact positions (cm):\n' + np.array2string(
            POS, precision=4, suppress_small=True))
        print('  mu = %s,  omega_ext = %s' % (mus, np.round(wext, 3)))
        res[name[-1]] = verify_theory(A, mus, wext, ystar, Fstar, muT, LF,
                                      phi, 1/LF, 1/LF, name)
    ok &= check_robustness_constant()
    for k_, r_ in res.items():
        ok &= (r_['v24'] == 0) & (r_['v34'] == 0) & r_['erg_ok'] \
              & r_['contract_ok'] & r_['ec']['ok'] & bool(r_['lam_under']
                                         <= r_['lam_min_obs'] + 1e-12)

    print('\n================ performance tables ================')
    cold_start(A_A, MUS_A, WEXT_A, LF_A, phi, 'Case A')
    cold_start(A_B, MUS_B, WEXT_B, LF_B, phi, 'Case B')
    warm_trajectory(A_A, MUS_A, WEXT_A, LF_A, phi, 'Case A')
    warm_trajectory(A_B, MUS_B, WEXT_B, LF_B, phi, 'Case B')
    misestimation(A_A, MUS_A, WEXT_A, LF_A, phi, 'Case A')
    misestimation(A_B, MUS_B, WEXT_B, LF_B, phi, 'Case B')
    phi_sensitivity(A_B, MUS_B, WEXT_B, 'Case B')
    robustness('Case A', A_A, MUS_A, WEXT_A, grasp_cylinder, (1.6, 2.0))
    robustness('Case B', A_B, MUS_B, WEXT_B, grasp_sphere, (3.75, ZLEV_B))
    make_figures(A_A, A_B, phi, LF_A, LF_B, res['A'], res['B'], outdir='.')
    print('\nAll checks completed.  Overall OK = %s' % ok)
    if not ok:
        raise SystemExit('CHECK FAILED: at least one theoretical inequality '
                         'was violated; see output above.')
