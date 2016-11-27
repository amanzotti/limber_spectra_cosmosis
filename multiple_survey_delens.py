import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import ConfigParser


# cosmosis_dir = '/home/manzotti/my_version_cosmosis/'
# inifile = 'des_cib.ini'
# n_slices = 13


# Config_ini = ConfigParser.ConfigParser()
# values = ConfigParser.ConfigParser()
# Config_ini.read(inifile)
# values_file = Config_ini.get('pipeline', 'values')
# output_dir = Config_ini.get('test', 'save_dir')

# values.read(cosmosis_dir + values_file)



# ============================================================
# FIRST EXERCISE

# JUST GET RHO FOR DIFFERENT Z

# ============================================================

# cgk = np.zeros((n_slices, np.size(lbins)))
# cgg = np.zeros((n_slices, np.size(lbins)))
# rho = np.zeros((n_slices, np.size(lbins)))
# ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk.txt')

# for i, z_bin in enumerate(np.arange(1, n_slices + 1)):

#     cgk[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(z_bin) + '.txt')
#     cgg[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(z_bin) + '.txt')

# rho = cgk/np.sqrt(cgg*ckk)

# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(rho)
# cbar = fig.colorbar(cax)
# ax.set_aspect('auto')
# plt.savefig('try2.pdf')


# ============================================================
# THIRD EXERCISE
# ============================================================


# cgk = np.zeros((n_slices, np.size(lbins)))
# cgg = np.zeros((n_slices, n_slices, np.size(lbins)))
# ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk.txt')

# for i in np.arange(0, n_slices ):
#     cgk[i,:] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(i+1) + '.txt')

#     for j in np.arange(i, n_slices):
#         cgg[i, j, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(i+1) + '.txt')
#         cgg[j,i, :] =cgg[i,j ,:]

# cgk[i,j :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(z_bin1)+str(z_bin2) + '.txt')
# cgg[i, j :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(z_bin) +str(z_bin2)  + '.txt')


# rho = cgk/np.sqrt(cgg*ckk)

# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(rho)
# cbar = fig.colorbar(cax)
# ax.set_aspect('auto')
# plt.savefig('try2.pdf')

'''
To add the cmb contribution just add a ckg with value ckk and a cgg with value cgg+noise
'''


# ============================================================
# DES CIB EXERCISE
# ============================================================


cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = 'des_cib.ini'
n_slices = 2


Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')
output_dir = 'modules/limber/cib_des_delens/'
values.read(cosmosis_dir + values_file)


lbins = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/ells_des.txt')


cgk = np.zeros((3, np.size(lbins)))
cgg = np.zeros((3, 3, np.size(lbins)))

ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clkdes.txt')

noise_cl = np.nan_to_num(np.loadtxt(
    '/home/manzotti/cosmosis/modules/likelihoods/spt_cosmosis/quicklens/TT_noise_9.0muk_1beam.txt'))
# # check the first ell is the same
# # assert (ells_cmb[0] == noise_cl[0, 0])
ell_clkk = np.arange(len(noise_cl))
noise_cl = noise_cl * ell_clkk ** 4 /4. # because the power C_kk is l^4/4 C_phi
noise_fun = interp1d(ell_clkk, noise_cl)
ckk_noise = np.zeros_like(ckk)
ckk_noise = noise_fun(lbins)


rho_comb = np.zeros((np.size(lbins)))
rho_cib_des = np.zeros((np.size(lbins)))
rho_des = np.zeros((np.size(lbins)))
rho_cib = np.zeros((np.size(lbins)))

labels = ['des', 'cib']

for i in np.arange(0, n_slices):
    cgk[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + labels[i] + 'k_delens.txt')

    for j in np.arange(i, n_slices):
        cgg[i, j, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + labels[i] + labels[j] + '_delens.txt')
        cgg[j, i, :] = cgg[i, j, :]


# add cmb lensing

cgk[2, :] = ckk
cgg[2, : , :] =  cgk[:, :]
cgg[:, 2 , :] = cgg[2, : , :]
cgg[2, 2, :] = ckk + ckk_noise


rho_des = cgk[0, :] / np.sqrt(ckk[:] * cgg[0, 0, :])
rho_cib = cgk[1, :] / np.sqrt(ckk[:] * cgg[1, 1, :])
rho_cmb = cgk[2, :] / np.sqrt(ckk[:] * cgg[2, 2, :])


for i, ell in enumerate(lbins):
    rho_comb[i] = np.dot(cgk[:, i], np.dot(np.linalg.inv(cgg[:, :, i]), cgk[:, i])) / ckk[i]
    rho_cib_des[i] = np.dot(cgk[:2, i], np.dot(np.linalg.inv(cgg[:2, :2, i]), cgk[:2, i])) / ckk[i]


np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_cib.txt', np.vstack((lbins, rho_cib)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_des.txt', np.vstack((lbins, rho_des)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_cmb.txt', np.vstack((lbins, rho_cmb)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_comb_des_cib_cmb.txt', np.vstack((lbins, np.sqrt(rho_comb))).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_comb_des_cib.txt', np.vstack((lbins, np.sqrt(rho_cib_des))).T)

sys.exit()
# rho = cgk/np.sqrt(cgg*ckk)

