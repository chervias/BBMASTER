import bbmaster as mst
import numpy as np
import pymaster as nmt
import healpy as hp
import toast

nside = 128

# bins
bins_edges = np.loadtxt('data/bins_single.txt')
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

bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=False, lmin=80, lmax=110, nsim_per_ell=10, save_maps=False, mode=3, seed0=1000, obsmat=obsmat, nside_high=128, beam=0.5 )
bpw_num_z2 = bc_z2.gen_Bbl_all()

np.save('output/BPWFs_Z2_10sims_nside128_obsmat-filtered', bpw_num_z2)