# Clamp-on V path, pipe CROSS-SECTION: shows how the curved wall focuses the beam
# and whether more PZT WIDTH helps. 2D scalar wave simulation (finite differences).
# Self-contained: needs only numpy, numba, matplotlib.
# Run once per width (change Wpzt) and compare the printed peak in dB.
#
# Approximations:
#  - The real beam is tilted along the pipe axis. Each material therefore gets an
#    effective speed c/cos(angle), which is correct inside the cross-section plane.
#  - Steel is a scalar medium carrying the shear wave, so waves running around the
#    wall are not modelled correctly.
#  - Transmitter and receiver are at the same place in this view, so one line is
#    used for both. The echo from the far wall is the V-path signal.
# x = depth (rows, downwards), y = across the pipe (columns)
import numpy as np
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- inputs (edit these) ----
f0   = 2.15e6     # frequency [Hz]
ncyc = 7          # burst cycles
Wpzt = 13e-3      # PZT width around the pipe [m]
ang  = 33.15      # wedge angle [deg]
h    = 10e-3      # PZT centre height above the pipe surface [m]
OD   = 114.3e-3   # pipe outer diameter [m]
wall = 8.56e-3    # pipe wall [m]   (4" Sch80)
t0   = 1.5e-3     # rubber thickness, uncompressed [m]
tr   = 1.05e-3    # rubber thickness under the centre (30% compressed) [m]
dx   = 0.1e-3     # grid step [m]
tag  = "circ"     # name used for the output files

#      speed, density
wdg = (2500.0, 1300.0)   # wedge, compression (density assumed)
rub = (1000.0, 1100.0)   # rubber (assumed, silicone-like)
stl = (3230.0, 7850.0)   # steel, shear
wat = (1480.0, 1000.0)   # water

# ---- effective in-plane speeds: c/cos(angle), angle from Snell's law ----
p = np.sin(np.radians(ang)) / wdg[0]
eff = lambda c: c / np.cos(np.arcsin(p * c))

# ---- grid ----
nsp = 60                                  # absorbing border [cells]
pad = nsp*dx + 2e-3
Ro = OD/2; Ri = Ro - wall
x_pzt  = pad                              # depth of the PZT line
x_top  = x_pzt + h                        # depth of the pipe crown
x_base = x_top - tr                       # depth of the flat wedge base
xc = x_top + Ro                           # pipe centre depth
Nx = int(round((x_top + OD + 2e-3) / dx))
Ny = int(round((OD + 2*pad) / dx))
yc = Ny*dx/2
X = (np.arange(Nx) * dx)[:, None] * np.ones((1, Ny))
Y = np.ones((Nx, 1)) * (np.arange(Ny) * dx)[None, :]
r = np.hypot(X - xc, Y - yc)

# ---- materials (0 everywhere = air, pressure stays zero there) ----
y_touch = np.sqrt(2*Ro*(t0 - tr))         # half-width where the rubber still touches
print(f"Rubber touches the pipe over a width of {2*y_touch*1e3:.1f} mm")
c = np.zeros((Nx, Ny)); rho = np.zeros((Nx, Ny))
def fill(mask, m):
    c[mask] = eff(m[0]); rho[mask] = m[1]
fill(X < x_base, wdg)
fill((X >= x_base) & (X < xc) & (r > Ro) & (np.abs(Y - yc) < y_touch), rub)
fill((r <= Ro) & (r > Ri), stl)
fill(r <= Ri, wat)

K = rho * c**2
ra = 0.5*(rho[:-1, :] + rho[1:, :]); bx = np.zeros((Nx, Ny)); bx[:-1, :] = np.where(ra > 0, 1/np.maximum(ra, 1e-9), 0)
ra = 0.5*(rho[:, :-1] + rho[:, 1:]); by = np.zeros((Nx, Ny)); by[:, :-1] = np.where(ra > 0, 1/np.maximum(ra, 1e-9), 0)

# ---- absorbing border ----
ix = np.arange(Nx)[:, None]; iy = np.arange(Ny)[None, :]
d = np.minimum(np.minimum(ix, Nx - 1 - ix), np.minimum(iy, Ny - 1 - iy))
damp = np.exp(-(0.035 * np.maximum(nsp - d, 0) / nsp * 4)**2)

# ---- time: echo from the far inner wall ----
t_echo = 2*(h - tr)/eff(wdg[0]) + 2*tr/eff(rub[0]) + 2*wall/eff(stl[0]) + 2*(2*Ri)/eff(wat[0])
dt = 0.7 * dx / (c.max() * np.sqrt(2) * (9/8 + 1/24))
nt = int((t_echo + ncyc/f0 + 8e-6) / dt)
t  = np.arange(nt) * dt
burst = np.sin(2*np.pi*f0*t) * np.sin(np.pi*f0*t/ncyc)**2 * (t < ncyc/f0)

# ---- PZT: one flat line, used as transmitter and receiver ----
i_pzt = int(round(x_pzt/dx))
j_pzt = np.nonzero(np.abs(np.arange(Ny)*dx - yc) <= Wpzt/2)[0]

@nb.njit(parallel=True, fastmath=True)
def step(pr, vx, vy, K, bx, by, damp, k):
    c1, c2 = 9/8, -1/24
    Nx, Ny = pr.shape
    for i in nb.prange(2, Nx-2):
        for j in range(2, Ny-2):
            vx[i, j] = damp[i, j] * (vx[i, j] - k*bx[i, j]*(c1*(pr[i+1, j]-pr[i, j]) + c2*(pr[i+2, j]-pr[i-1, j])))
            vy[i, j] = damp[i, j] * (vy[i, j] - k*by[i, j]*(c1*(pr[i, j+1]-pr[i, j]) + c2*(pr[i, j+2]-pr[i, j-1])))
    for i in nb.prange(2, Nx-2):
        for j in range(2, Ny-2):
            div = (c1*(vx[i, j]-vx[i-1, j]) + c2*(vx[i+1, j]-vx[i-2, j]) +
                   c1*(vy[i, j]-vy[i, j-1]) + c2*(vy[i, j+1]-vy[i, j-2]))
            pr[i, j] = damp[i, j] * (pr[i, j] - k*K[i, j]*div)

# ---- run ----
f32 = lambda z: z.astype(np.float32)
K, bx, by, damp = map(f32, (K, bx, by, damp))
pr, vx, vy = (np.zeros((Nx, Ny), np.float32) for _ in range(3))
rec = np.zeros(nt)
snap_at = [int(f*t_echo/dt) for f in (0.2, 0.5, 0.75, 0.97)]
snaps = []
print(f"grid {Nx} x {Ny}, {nt} steps, expected echo at {t_echo*1e6:.1f} us")
for n in range(nt):
    pr[i_pzt, j_pzt] += burst[n]
    step(pr, vx, vy, K, bx, by, damp, np.float32(dt/dx))
    rec[n] = pr[i_pzt, j_pzt].mean()       # face-averaged pressure
    if n in snap_at:
        snaps.append(pr[::4, ::4].copy())

# ---- results ----
win = (t > t_echo - 2e-6) & (t < t_echo + ncyc/f0 + 4e-6)
echo_pk = np.abs(rec[win]).max()
print(f"Wpzt = {Wpzt*1e3:.0f} mm")
print(f"Far-wall echo peak (face average): {echo_pk:.4g}")
print(f"Echo peak x width (total charge) : {echo_pk*Wpzt*1e3:.4g}")
np.savez(f"{tag}.npz", t=t, rec=rec, echo_pk=echo_pk)

plt.figure(figsize=(7, 3)); plt.plot(t*1e6, rec); plt.axvline(t_echo*1e6, color="r", ls="--")
plt.ylim(-3*echo_pk, 3*echo_pk)
plt.xlabel("time [us]"); plt.ylabel("face-averaged pressure"); plt.title(f"Cross-section echo, Wpzt = {Wpzt*1e3:.0f} mm")
plt.tight_layout(); plt.savefig(f"{tag}_signal.png", dpi=110)

fig, ax = plt.subplots(1, len(snaps), figsize=(3.2*len(snaps), 3.6))
for k, sn in enumerate(snaps):
    m = np.abs(sn).max()*0.3
    ax[k].imshow(sn, cmap="seismic", vmin=-m, vmax=m, extent=[0, Ny*dx*1e3, Nx*dx*1e3, 0])
    ax[k].set_title(f"{snap_at[k]*dt*1e6:.0f} us"); ax[k].set_xlabel("across pipe [mm]")
ax[0].set_ylabel("depth [mm]")
plt.tight_layout(); plt.savefig(f"{tag}_field.png", dpi=110)
