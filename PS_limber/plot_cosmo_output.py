import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline

import ConfigParser

cosmosis_dir = '/home/manzotti/my_version_cosmosis/'
Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
inifile = 'test.ini'
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

values.read(cosmosis_dir + values_file)


zpower = np.loadtxt(cosmosis_dir + output_dir + '/matter_power_nl/z.txt')
kpower = np.loadtxt(cosmosis_dir + output_dir + '/matter_power_nl/k_h.txt')
powerarray = np.loadtxt(cosmosis_dir + output_dir + '/matter_power_nl/p_k.txt')
powerarray = np.loadtxt(
    cosmosis_dir + output_dir + '/matter_power_nl/p_k.txt').reshape([np.size(zpower), np.size(kpower)]).T
rbs = RectBivariateSpline(kpower, zpower, powerarray)

omega_m = values.getfloat('cosmological_parameters', 'omega_m')
h0 = values.getfloat('cosmological_parameters', 'h0')
h = np.loadtxt(cosmosis_dir+output_dir + '/distances/h.txt')

tmp = h[::-1]
h = tmp

xlss = 13615.317054155654
zdist = np.loadtxt(cosmosis_dir + output_dir + '/distances/z.txt')
tmp = zdist[::-1]
zdist = tmp

d_m = np.loadtxt(cosmosis_dir + output_dir + '/distances/d_m.txt')
tmp = d_m[::-1]
d_m = tmp

d_m *= h0
h /= h0
xlss *= h0

chispline = interp1d(zdist, d_m)
z_chi_spline = interp1d(d_m, zdist)
hspline = interp1d(zdist, h)


zbins= np.linspace(0.1,5,100)
lbins= np.arange(2,2000)
P_plot= np.zeros((np.size(lbins),np.size(zbins)))

for i,ell in enumerate(lbins):
    for j,z in enumerate(zbins):
        P_plot[i,j]=rbs( (ell+0.5)/chispline(z), z )

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(P_plot_log)
ax.set_aspect('auto')




