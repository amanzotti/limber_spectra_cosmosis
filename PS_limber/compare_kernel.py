import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
zpower = np.loadtxt('output/matter_power_nl/z.txt')
kpower = np.loadtxt('output/matter_power_nl/k_h.txt')
powerarray = np.loadtxt('output/matter_power_nl/p_k.txt')
powerarray = np.loadtxt('output/matter_power_nl/p_k.txt').reshape([np.size(zpower), np.size(kpower)]).T
rbs = RectBivariateSpline(kpower, zpower, powerarray)

omega_m = 0.3

h0 = 0.7


h = np.loadtxt('output/distances/h.txt')
tmp = h[::-1]
h = tmp

xlss = 13615.317054155654
zdist = np.loadtxt('output/distances/z.txt')
tmp = zdist[::-1]
zdist = tmp

d_m = np.loadtxt('output/distances/d_m.txt')
tmp = d_m[::-1]
d_m = tmp

d_m *= h0
h /= h0
xlss *= h0

chispline = interp1d(zdist, d_m)
z_chi_spline = interp1d(d_m, zdist)
hspline = interp1d(zdist, h)

dndz_filename = 'data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt'
dndz_des = np.loadtxt(dndz_filename)
dndzfun = interp1d(dndz_des[:, 0], dndz_des[:, 1])
norm = scipy.integrate.quad(dndzfun, dndz_des[0, 0], dndz_des[-1, 0])[0]
# normalize
dndzfun_des = interp1d(dndz_des[:, 0], dndz_des[:, 1] / norm)

dndz_filename = 'data_input/DESI/DESI_dndz.txt'
dndz = np.loadtxt(dndz_filename)
dndz[:, 1] = np.sum(dndz[:, 1:], axis=1)
norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
# normalize
dndzfun_desi = interp1d(dndz[:, 0], dndz[:, 1] / norm)


nu = 353e9
zs = 2.
b = 1.
j2k = 1.e-6 / np.sqrt(83135.)  # for 353
lkern = kappa_kernel.kern(zdist, omega_m, h0, xlss)
des = gals_kernel.kern(dndz_des[:, 0], dndzfun, hspline, omega_m, h0)
cib = cib_hall.ssed_kern(h0, zdist, chispline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})
desi = gals_kernel.kern(dndz[:, 0], dndzfun_desi, hspline, omega_m, h0)


l = 30
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z) / hspline(z)

z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z) / hspline(z)

np.savetxt('output/limber_spectra/desi_kernel_l30.txt', np.vstack((z_desi, w_desi)).T)
np.savetxt('output/limber_spectra/des_kernel_l30.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l30.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l30.txt', np.vstack((z_kappa, w_kappa)).T)

l = 30
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z)
# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z)

z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z)

np.savetxt('output/limber_spectra/desi_kernel_l30_h.txt', np.vstack((z_desi, w_desi)).T)
# plt.plot(z_des,w_des,label='des')


np.savetxt('output/limber_spectra/des_kernel_l30_h.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l30_h.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l30_h.txt', np.vstack((z_kappa, w_kappa)).T)


# ========================================================================


l = 100
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z) / hspline(z)


z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z) / hspline(z)

np.savetxt('output/limber_spectra/desi_kernel_l100.txt', np.vstack((z_desi, w_desi)).T)

np.savetxt('output/limber_spectra/des_kernel_l100.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l100.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l100.txt', np.vstack((z_kappa, w_kappa)).T)


l = 100
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z)
# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z)

# plt.plot(z_des,w_des,label='des')


z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z)

np.savetxt('output/limber_spectra/desi_kernel_l100_h.txt', np.vstack((z_desi, w_desi)).T)

np.savetxt('output/limber_spectra/des_kernel_l100_h.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l100_h.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l100_h.txt', np.vstack((z_kappa, w_kappa)).T)

# ========================================================================


l = 500
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z) / hspline(z)

# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z) / hspline(z)


z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z) / hspline(z)

np.savetxt('output/limber_spectra/desi_kernel_l500.txt', np.vstack((z_desi, w_desi)).T)
np.savetxt('output/limber_spectra/des_kernel_l500.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l500.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l500.txt', np.vstack((z_kappa, w_kappa)).T)

l = 500
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = lkern.w_lxz(l, x, z)

# plt.plot(z_kappa,w_kappa,label='cmb kappa')

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    w_cib[i] = cib.w_lxz(l, x, z)
# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = des.w_lxz(l, x, z)

z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = desi.w_lxz(l, x, z)

np.savetxt('output/limber_spectra/desi_kernel_l500_h.txt', np.vstack((z_desi, w_desi)).T)
np.savetxt('output/limber_spectra/des_kernel_l500_h.txt', np.vstack((z_des, w_des)).T)
np.savetxt('output/limber_spectra/cib_kernel_l500_h.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt('output/limber_spectra/kappa_kernel_l500_h.txt', np.vstack((z_kappa, w_kappa)).T)
