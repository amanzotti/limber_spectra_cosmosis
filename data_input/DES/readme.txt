D.E.S redshift distribtion taken from https://cdcvs.fnal.gov/redmine/projects/des-lss/wiki/Bench_mask_TPZv3

The files are the redshift distributions in five photo-z bins using TPZv3 and WAVG_SPREAD_MODEL > 0.003 as SG separation




####   COMMENTS   ####

[] This catalogue corresponds to a bench-mark selection based on mag_auto magnitudes corrected by the new SLR
maps (which we call new photometry). This magnitude is given in the 5th field.

[] Weird colors selection is also done with updated SLR magnitudes

[] For SG separation do

wavg_spread_model > 0.003
tpz_sg_class < 0.14

NOTE: tpz_sg_class was not re-run with the new photometry therefore
tpz_sg_class_v3 is the same as tpz_sg_class_v2.

[] tpc and tpc_mag are preliminary

####   SELECTION   ####

( 60 < ra < 95.)
( -61 < dec < -40.)

( -18 <mag_auto_i < 22.5)

(mag_detmodel_g - mag_detmodel_r) BETWEEN -1. and 3.
(mag_detmodel_r  - mag_detmodel_i) BETWEEN -1. and 2.
(mag_detmodel_i  - mag_detmodel_z) BETWEEN -1. and 2.

####   FIELDS    ####

GENERAL

1 coadd_objects_id
2 ra
3 dec
4 mag_auto_i
5 mag_auto_i_new (THIS FIELD WAS THE ONE USED FOR MAG CUT < 22.5)

PHOTOZ

6 tpz_v2
7 tpz_v3
8 skynet_z_max_pdf
9 zb

SG_CLASS

10 wavg_spread_model
11 wavg_spreaderr_model
12 tpz_sg_class_v2
13 tpz_sg_class_v3 (== tpz_sg_class_v2)
14 tpc             (optimization of tpz_sg, uses colors and wavg_spreaderr_model)
15 tpc_mag         (optimization of tpz_sg, uses colors, wavg_spreaderr_model, and magnitudes)

########################

