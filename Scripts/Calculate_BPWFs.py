import bbmaster as mst
import numpy as np
import pymaster as nmt
import healpy as hp
#import pickle as pkl
import toast

nside = 128

# bins
bins_edges = np.loadtxt('data/bins.txt')
bins_edges_min = [] ; bins_edges_max = []
for n in range(len(bins_edges)-1):
	bins_edges_min.append(int(bins_edges[n]))
	bins_edges_max.append(int(bins_edges[n+1]))
bins = nmt.NmtBin.from_edges(bins_edges_min, bins_edges_max, is_Dell=False)
#bins = nmt.NmtBin.from_nside_linear(nside, nlb=10, is_Dell=False)

msk = hp.read_map("data/mask_SAT_FirstDayEveryMonth_apo5.0_fpthin8_nside512.fits")
msk = hp.ud_grade(msk,nside)

filt = {'mask': msk}
dsim = {'stats': 'Z2'}

obsmat = toast.ObsMat('output/obsmat_coadd-full.npz')
print('obsmat loaded')

#indir = '/home/chervias/CMBwork/SimonsObs/BBMASTER/output/'
#in_name = '/map_tqu_nside64_%s_ell%04i_seed%i_map%i_TOAST-obsmat-filtered.fits'

#bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=1, lmin=20, lmax=110, nsim_per_ell=100, save_maps=False, pure=0, mode=2, indir=indir, in_name=in_name, seed0=1000 )
#bpw_num_z2 = bc_z2.gen_Bbl_all()

#np.save('output/BPWFs_Z2_100sims_nside0064_TOAST-obsmat-filtered', bpw_num_z2)
#np.save('output/Std_Z2_100sims_nside0064_TOAST-obsmat-filtered',bc_z2.errors)

bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=False, lmin=80, lmax=110, nsim_per_ell=10, save_maps=False, mode=3, seed0=1000, obsmat=obsmat, nside_high=128, beam=0.5 )
bpw_num_z2 = bc_z2.gen_Bbl_all()

np.save('output/BPWFs_Z2_10sims_nside128_obsmat-filtered', bpw_num_z2)