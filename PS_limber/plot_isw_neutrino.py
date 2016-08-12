import numpy as np
import isw_kernel
import gals_kernel
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import matplotlib.pylab as plt

kgrowth = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/kfnu_z_01.txt')
zgrowth = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/zfnu_z_01.txt')
growth = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/d2_zfnu_z_01.txt')
growth_fun = RectBivariateSpline(kgrowth, zgrowth, growth)

zgrowth01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/zfnu_z_01.txt')
kgrowth01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/kfnu_z_01.txt')
growth01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/d2_zfnu_z_01.txt')
growth_fun01 = RectBivariateSpline(kgrowth01, zgrowth01, growth01)


ctg01_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_iswg_fnu_z_01.txt')
ctg01_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_iswg_fnu_z_1.txt')
ctg01_z_2 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_iswg_fnu_z_2.txt')
ctg01_z_3 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_iswg_fnu_z_3.txt')

ctg0_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_iswg_fnu_z_01.txt')
ctg0_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_iswg_fnu_z_1.txt')
ctg0_z_2 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_iswg_fnu_z_2.txt')
ctg0_z_3 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_iswg_fnu_z_3.txt')

cisw01_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_isw_fnu_z_01.txt')
cisw01_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_isw_fnu_z_1.txt')


cisw0_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_isw_fnu_z_01.txt')
cisw0_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_isw_fnu_z_1.txt')


ell = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/ells_fnu_z_3.txt')

cg0_z_3 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_g_fnu_z_3.txt')
cg0_z_2 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_g_fnu_z_2.txt')
cg0_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_g_fnu_z_1.txt')
cg0_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/limber_spectra/cl_g_fnu_z_01.txt')

cg01_z_3 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_g_fnu_z_3.txt')
cg01_z_2 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_g_fnu_z_2.txt')
cg01_z_1 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_g_fnu_z_1.txt')
cg01_z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/limber_spectra/cl_g_fnu_z_01.txt')

k_0 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/matter_power_lin/k_h.txt')
z_0 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/matter_power_lin/z.txt')
Pk_0 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_0/matter_power_lin/p_k.txt')
Pk_0 = Pk_0.reshape([np.size(z_0), np.size(k_0)]).T
Pk0 = RectBivariateSpline(k_0, z_0, Pk_0)

k_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/matter_power_lin/k_h.txt')
z_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/matter_power_lin/z.txt')
Pk_01 = np.loadtxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output_mari_isw_fnu_01/matter_power_lin/p_k.txt')
Pk_01 = Pk_01.reshape([np.size(z_01), np.size(k_01)]).T
Pk01 = RectBivariateSpline(k_01, z_01, Pk_01)


