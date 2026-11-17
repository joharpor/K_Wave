# Clamp-on V path, axial slice: 2D elastic wave simulation (finite differences).
# Self-contained: needs only numpy, numba, matplotlib  (pip install numpy numba matplotlib)
# Run once per PZT length (change Lpzt) and compare the printed peaks in dB.
#
# Model: wedge pod / steel wall / water / steel wall, air (vacuum) elsewhere.
# The rubber couplant is left out (uniform layer in this slice, same for every size).
# x = depth (rows, downwards), y = along the pipe (columns)
import numpy as np
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- inputs (edit these) ----
f0   = 2.15e6     # frequency [Hz]
ncyc = 7          # burst cycles
Lpzt = 18e-3      # PZT length along the pipe axis [m]
ang  = 33.15      # wedge angle [deg]
h    = 10e-3      # PZT centre height above the pipe surface [m]
wall = 8.56e-3    # pipe wall [m]   (4" Sch80)
ID   = 97.2e-3    # inner diameter [m]
dx   = 0.1e-3     # grid step [m] (about 7 points per wavelength in water)
c_set = 1480.0    # water sound speed at which the PZT spacing was set [m/s]
tag  = "run"      # name used for the output files

#      compression, shear, density
wdg = (2500.0, 1100.0, 1300.0)   # wedge (shear and density assumed, PEEK-like)
stl = (5900.0, 3230.0, 7850.0)   # steel
wat = (1480.0,    0.0, 1000.0)   # water

# ---- ray geometry (Snell's law) ----
a = np.radians(ang)
p = np.sin(a) / wdg[0]
a_stl = np.arcsin(p * stl[1])            # shear angle in steel
a_wat = np.arcsin(p * wat[0])            # angle in water
S = 2*h*np.tan(a) + 2*wall*np.tan(a_stl) + 2*ID*np.tan(np.arcsin(p*c_set))   # PZT spacing (fixed)
t_fluid = 2*(h/np.cos(a))/wdg[0] + 2*(wall/np.cos(a_stl))/stl[1] + 2*(ID/np.cos(a_wat))/wat[0]

# ---- grid ----
nsp = 60                                  # absorbing border [cells]
pad = nsp*dx + 2e-3
xc  = pad + Lpzt/2*np.sin(a)              # depth of both PZT centres
xs  = xc + h                              # depth of the pipe outer surface
y1  = pad + Lpzt/2*np.cos(a) + 3e-3       # transmitter centre
y2  = y1 + S                              # receiver centre
Nx  = int(round((xs + 2*wall + ID + 2e-3) / dx))
Ny  = int(round((y2 + Lpzt/2*np.cos(a) + 3e-3 + pad) / dx))
X = (np.arange(Nx) * dx)[:, None] * np.ones((1, Ny))
Y = np.ones((Nx, 1)) * (np.arange(Ny) * dx)[None, :]

# ---- materials (0 everywhere = vacuum, used for air) ----
cp = np.zeros((Nx, Ny)); cs = np.zeros((Nx, Ny)); rho = np.zeros((Nx, Ny))
def fill(mask, m):
    cp[mask], cs[mask], rho[mask] = m
foot = (Lpzt/2)/np.cos(a) + 4e-3                         # half footprint + margin
gap  = (Y > y1 + h*np.tan(a) + foot) & (Y < y2 - h*np.tan(a) - foot)   # air between pods
fill((X < xs) & ~gap, wdg)
fill((X >= xs) & (X < xs + wall), stl)
fill((X >= xs + wall) & (X < xs + wall + ID), wat)
fill((X >= xs + wall + ID) & (X < xs + 2*wall + ID), stl)

mu  = rho * cs**2
lam = rho * cp**2 - 2*mu
ra = 0.5*(rho[:-1, :] + rho[1:, :]); bx = np.zeros((Nx, Ny)); bx[:-1, :] = np.where(ra > 0, 1/np.maximum(ra, 1e-9), 0)
ra = 0.5*(rho[:, :-1] + rho[:, 1:]); by = np.zeros((Nx, Ny)); by[:, :-1] = np.where(ra > 0, 1/np.maximum(ra, 1e-9), 0)
m4 = np.stack([mu[:-1, :-1], mu[1:, :-1], mu[:-1, 1:], mu[1:, 1:]])
muxy = np.zeros((Nx, Ny))
muxy[:-1, :-1] = np.where(m4.min(0) > 0, 4/np.sum(1/np.maximum(m4, 1e-9), 0), 0)   # 0 next to fluid/air

# ---- absorbing border (not on the bottom: that is air below the far wall) ----
ix = np.arange(Nx)[:, None]; iy = np.arange(Ny)[None, :]
d = np.minimum(np.minimum(ix, iy), Ny - 1 - iy) * np.ones((Nx, Ny))
damp = np.exp(-(0.035 * np.maximum(nsp - d, 0) / nsp * 4)**2)

# ---- time ----
dt = 0.7 * dx / (stl[0] * np.sqrt(2) * (9/8 + 1/24))
nt = int((t_fluid + ncyc/f0 + 8e-6) / dt)
t  = np.arange(nt) * dt
burst = np.sin(2*np.pi*f0*t) * np.sin(np.pi*f0*t/ncyc)**2 * (t < ncyc/f0)

# ---- PZTs as tilted lines in the wedge ----
s = np.arange(-Lpzt/2, Lpzt/2, dx/4)
tx = np.zeros((Nx, Ny)); rx = np.zeros((Nx, Ny))
np.add.at(tx, (np.round((xc - s*np.sin(a))/dx).astype(int), np.round((y1 + s*np.cos(a))/dx).astype(int)), 1)
np.add.at(rx, (np.round((xc + s*np.sin(a))/dx).astype(int), np.round((y2 + s*np.cos(a))/dx).astype(int)), 1)
tx_i = np.nonzero(tx); tx_w = tx[tx_i] / tx.sum() * (Lpzt/dx)
rx_i = np.nonzero(rx); rx_w = rx[rx_i] / rx.sum()

# ---- one time step (staggered grid, 4th order in space) ----
@nb.njit(parallel=True, fastmath=True)
def step(vx, vy, sxx, syy, sxy, lam, mu, muxy, bx, by, damp, k):
    c1, c2 = 9/8, -1/24
    Nx, Ny = vx.shape
    for i in nb.prange(2, Nx-2):
        for j in range(2, Ny-2):
            vx[i, j] = damp[i, j] * (vx[i, j] + k*bx[i, j]*(
                c1*(sxx[i+1, j]-sxx[i, j]) + c2*(sxx[i+2, j]-sxx[i-1, j]) +
                c1*(sxy[i, j]-sxy[i, j-1]) + c2*(sxy[i, j+1]-sxy[i, j-2])))
            vy[i, j] = damp[i, j] * (vy[i, j] + k*by[i, j]*(
                c1*(sxy[i, j]-sxy[i-1, j]) + c2*(sxy[i+1, j]-sxy[i-2, j]) +
                c1*(syy[i, j+1]-syy[i, j]) + c2*(syy[i, j+2]-syy[i, j-1])))
    for i in nb.prange(2, Nx-2):
        for j in range(2, Ny-2):
            exx = c1*(vx[i, j]-vx[i-1, j]) + c2*(vx[i+1, j]-vx[i-2, j])
            eyy = c1*(vy[i, j]-vy[i, j-1]) + c2*(vy[i, j+1]-vy[i, j-2])
            exy = (c1*(vx[i, j+1]-vx[i, j]) + c2*(vx[i, j+2]-vx[i, j-1]) +
                   c1*(vy[i+1, j]-vy[i, j]) + c2*(vy[i+2, j]-vy[i-1, j]))
            sxx[i, j] = damp[i, j] * (sxx[i, j] + k*((lam[i, j]+2*mu[i, j])*exx + lam[i, j]*eyy))
            syy[i, j] = damp[i, j] * (syy[i, j] + k*(lam[i, j]*exx + (lam[i, j]+2*mu[i, j])*eyy))
            sxy[i, j] = damp[i, j] * (sxy[i, j] + k*muxy[i, j]*exy)

# ---- run ----
f32 = lambda z: z.astype(np.float32)
lam, mu, muxy, bx, by, damp = map(f32, (lam, mu, muxy, bx, by, damp))
vx, vy, sxx, syy, sxy = (np.zeros((Nx, Ny), np.float32) for _ in range(5))
rec = np.zeros(nt)
snap_dt = 10e-6                           # snapshot interval [s]
snap_at = set(range(int(round(snap_dt/dt)), nt, int(round(snap_dt/dt))))
snaps, snap_t = [], []
ds = 2                                    # keep every 2nd cell (every 4th aliases the water wavelength)
print(f"grid {Nx} x {Ny}, {nt} steps, spacing {S*1e3:.1f} mm, expected arrival {t_fluid*1e6:.1f} us")
for n in range(nt):
    sxx[tx_i] -= tx_w * burst[n]          # pressure source on the transmitter face
    syy[tx_i] -= tx_w * burst[n]
    step(vx, vy, sxx, syy, sxy, lam, mu, muxy, bx, by, damp, np.float32(dt/dx))
    rec[n] = np.sum(rx_w * -(sxx[rx_i] + syy[rx_i]) / 2)   # face-averaged pressure
    if n in snap_at:
        snaps.append(-(sxx + syy)[::ds, ::ds] / 2)
        snap_t.append(n*dt)

# ---- results ----
win = (t > t_fluid - 3e-6) & (t < t_fluid + ncyc/f0 + 5e-6)
fluid_pk = np.abs(rec[win]).max()
wall_pk  = np.abs(rec[t < t_fluid - 3e-6]).max()
t_pk = t[win][np.abs(rec[win]).argmax()]
print(f"Lpzt = {Lpzt*1e3:.0f} mm")
print(f"Fluid signal peak        : {fluid_pk:.4g}  (at {t_pk*1e6:.1f} us)")
print(f"Earlier (wall-borne) peak: {wall_pk:.4g}")
print(f"Fluid / wall             : {20*np.log10(fluid_pk/wall_pk):.1f} dB")
np.savez(f"{tag}.npz", t=t, rec=rec, fluid_pk=fluid_pk, wall_pk=wall_pk)
np.savez_compressed(f"{tag}_snaps.npz", snaps=np.array(snaps, np.float32), snap_t=np.array(snap_t),
                    dx=dx*ds, t_fluid=t_fluid, y1=y1, y2=y2, xc=xc,
                    layers=np.array([xs, xs + wall, xs + wall + ID, xs + 2*wall + ID]))

plt.figure(figsize=(7, 3)); plt.plot(t*1e6, rec); plt.axvline(t_fluid*1e6, color="r", ls="--")
plt.xlabel("time [us]"); plt.ylabel("face-averaged pressure"); plt.title(f"Received signal, Lpzt = {Lpzt*1e3:.0f} mm")
plt.tight_layout(); plt.savefig(f"{tag}_signal.png", dpi=110)

from plot_snaps import plot_grid          # snapshot matrix every 10 us -> {tag}_field_grid.png
plot_grid(f"{tag}_snaps.npz", f"{tag}_field_grid.png")
