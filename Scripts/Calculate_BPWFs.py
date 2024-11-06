import bbmaster as mst
import numpy as np
import pymaster as nmt
import healpy as hp
import toast
from mpi4py import MPI
import pickle

comm = MPI.COMM_WORLD

nside = 128

if False:
    bins_edges = np.loadtxt('data/bins.txt')
    bins_edges_min = [] ; bins_edges_max = []
    for n in range(len(bins_edges)-1):
    	bins_edges_min.append(int(bins_edges[n]))
    	bins_edges_max.append(int(bins_edges[n+1]))
    bins = nmt.NmtBin.from_edges(bins_edges_min, bins_edges_max, is_Dell=False)
else:
    lmax = 2*nside-1
    bins = nmt.NmtBin.from_lmax_linear(lmax, nlb=10, )

msk = hp.read_map("data/mask_SAT_FirstDayEveryMonth_apo5.0_fpthin8_nside512.fits")
msk = hp.ud_grade(msk, nside)

filt = {'mask': msk}
dsim = {'stats': 'Z2'}

obsmat = toast.ObsMat('output/obsmat_coadd-full.npz')
print('obsmat loaded')

bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=False, lmin=2, lmax=lmax, nsim_per_ell=200, save_maps=False, mode=3, seed0=1000, obsmat=obsmat, beam=0.5, mcut=None, comm=comm)
#bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=False, lmin=20, lmax=170, nsim_per_ell=100, save_maps=False, mode=3, seed0=1000, nside_high=nside, mcut=30, comm=comm)
#bc_z2 = mst.DeltaBbl(nside, dsim, filt, bins, pol=True, interp_ells=False, lmin=20, lmax=170, nsim_per_ell=100, save_maps=False, mode=0, seed0=2000, comm=comm)
bpw_num_z2 = bc_z2.gen_Bbl_all()

if comm.rank == 0:
    np.save('output/BPWFs_Z2_200sims_nside128_obsmat-filtered', bpw_num_z2)
    np.save('output/cb_ell_bc_Z2_200sims_nside128_obsmat-filtered', bc_z2.cb_final)
    #np.save('output/BPWFs_Z2_100sims_nside128_mode0', bpw_num_z2)
    #np.save('output/cb_ell_bc_Z2_100sims_nside128_mode0', bc_z2.cb_final)
