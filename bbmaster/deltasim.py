import numpy as np
import healpy as hp
import pymaster as nmt
import copy
import sys
import os
from mpi4py import MPI

class DeltaBbl(object):
    def __init__(self, nside, dsim, filt, bins, lmin=2, lmax=None, pol=False, 
                 nsim_per_ell=10, interp_ells=None, seed0=1000, n_iter=3, mode=0, save_maps = False,
                 outdir=None, indir=None, in_name=None, obsmat=None, beam=None, mcut=None, comm=None, bin_mask=None): 
        # input beam in degrees.
        if not isinstance(dsim, dict):
            raise TypeError("For now delta simulators can only be "
                            "specified through a dictionary.")
    
        if not isinstance(filt, dict):
            raise TypeError("For now filtering operations can only be "
                            "specified through a dictionary.")
    
        if not isinstance(bins, nmt.NmtBin):
            raise TypeError("`bins` must be a NaMaster NmtBin object.")
        self.dsim_d = copy.deepcopy(dsim)
        self.dsim = self._dsim_default
        self.filt_d = copy.deepcopy(filt)
        self.filt = self._filt_default
        self.lmin = lmin
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        if lmax is None:
            lmax = 3*self.nside-1
        self.lmax = lmax
        self.bins = bins
        self.n_bins = self.bins.get_n_bands()
        self.n_ells = lmax-lmin+1
        self.nsim_per_ell = nsim_per_ell
        self.interp_ells = interp_ells
        self.seed0 = seed0
        self.n_iter = n_iter
        self.pol = pol
        self._prepare_filtering()
        self.alm_ord = hp.Alm()
        self._sqrt2 = np.sqrt(2.)
        self._oosqrt2 = 1/self._sqrt2
        self.mode = mode
        self.save_maps = save_maps
        self.outdir = outdir
        self.indir = indir
        self.in_name = in_name
        if self.pol:
            self.cb_ell = np.zeros((4,4,self.lmax+1)) # here we will accumulate per ell in each MPI job
            self.cb_final = np.zeros((4,4,self.lmax+1,self.lmax+1)) # here we will keep the total 
        self.errors = None
        self.obsmat = obsmat
        if beam is not None:
	        self.beam = np.radians(beam)
        else:
            self.beam = None
        self.mcut = mcut
        self.comm = comm
        self.bin_mask = bin_mask
    
    def _prepare_filtering(self):
        # Match pixel resolution
        self.filt_d['mask'] = hp.ud_grade(self.filt_d['mask'],nside_out=self.nside)
    def _gen_gaussian_map_old(self, ell):
        if self.pol:
            # We want to excite E (B) modes in Gaussian maps using healpy,
            # so we have to excite power spectra in EE and TE (BB and TB).
            # map_out contains 2x2 maps with (EE,EB,BE,BB)_in, (Q,U)_out
            map_out = np.zeros((4,2,self.npix))
            for ipol_in, p in enumerate([[1,3],[1,2,3,4,5],[1,2,3,4,5],[2,5]]):
                cl = np.zeros((6,3*self.nside), dtype='float64')
                for q in p:
                    cl[q, ell] = 1
                # synfast: Input Cls TT,EE,BB,TE,EB,TB. Output maps T,Q,U.
                map_out[ipol_in] = hp.synfast(cl, self.nside, pol=True, 
                                              new=False)[1:,:]
        else:
            cl = np.zeros(3*self.nside)
            cl[ell] = 1
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            map_out = hp.synfast(cl, self.nside)
        return map_out
    
    def _rands_iterator(self):
        # Loops over polarization pair (ip), map (im) and yields two sets random
        # numbers to pick from (pk) for that specific combination. Example:
        # For EE (ip==0), pick same numbers (0) for E modes and different ones
        # for B (1,2)
        for ip, p in enumerate([[[0,1],[0,2]], [[0,1],[2,0]],
                                 [[0,1],[1,2]], [[0,1],[2,1]]]):
            for im, pk in enumerate(p):
                yield im, ip, pk
    
    def _gen_gaussian_map(self, ell):
        idx = self.alm_ord.getidx(3*self.nside-1, ell, np.arange(ell+1))
        if self.pol:
            # shape (map1,map2; EE,EB,BE,BB; Q,U; ipix)
            map_out = np.zeros((2,4,2,self.npix)) 
            alms = np.zeros((2,4,3,self.alm_ord.getsize(3*self.nside-1)),
                            dtype='complex128')
            # We only need to pick from three independent sets of alms
            rans = np.random.normal(0, self._oosqrt2,
                                    size=6*(ell+1)).reshape([3,2,ell+1])
            rans[:, 0, 0] *= self._sqrt2
            rans[:, 1, 0] = 0
    
            for im, ip, pk in self._rands_iterator():
                alms[im,ip,1,idx] = rans[pk[0],0] + 1j*rans[pk[0],1]
                alms[im,ip,2,idx] = rans[pk[1],0] + 1j*rans[pk[1],1]
                map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside)[1:]
        else:
            alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),dtype='complex128')
            rans = np.random.normal(0, self._oosqrt2, size=2*(ell+1)).reshape([2,ell+1])
            rans[0, 0] *= self._sqrt2
            rans[1, 0] = 0
            alms[idx] = rans[0] + 1j*rans[1]
            # TODO: we can save time on the SHT massively, since in this case 
            # there is no sum over ell!
            map_out = hp.alm2map(alms, self.nside)
        return map_out
    
    def _gen_Z2_map(self, ell):
        # Analogous to Gaussian
        if self.mode in [0,1]:
            idx = self.alm_ord.getidx(3*self.nside-1, ell, np.arange(ell+1))
            if self.pol:
                # shape (map1,map2; EE,EB,BE,BB; T,Q,U; ipix)
                map_out = np.zeros((2,4,3,self.npix))
                alms = np.zeros((2,4,3,self.alm_ord.getsize(3*self.nside-1)),dtype='complex128')
                # We only need to pick from three independent sets of alms
                rans = self._oosqrt2*(2*np.random.binomial(1,0.5,size=6*(ell+1))-1).reshape([3,2,ell+1])
                rans[:, 0, 0] *= self._sqrt2
                rans[:, 1, 0] = 0
                for im, ip, pk in self._rands_iterator():
                    alms[im,ip,1,idx] = rans[pk[0],0] + 1j*rans[pk[0],1]
                    alms[im,ip,2,idx] = rans[pk[1],0] + 1j*rans[pk[1],1]                
                    map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside)#[1:] we comment this because we need a TQU map and not a QU map.
            else:
                rans = self._oosqrt2*(2*np.random.binomial(1,0.5,size=2*(ell+1))-1).reshape([2,ell+1])
                # Correct m=0 (it should be real and have twice as much variance)
                rans[0, 0] *= self._sqrt2
                rans[1, 0] = 0
                # Populate alms and transform to map
                alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),dtype='complex128')
                alms[idx] = rans[0] + 1j*rans[1]
                # TODO: we can save time on the SHT massively, since in this case 
                # there is no sum over ell!
                map_out = hp.alm2map(alms, self.nside)
        elif self.mode==3:
            idx = self.alm_ord.getidx(3*self.nside-1, ell, np.arange(ell+1))
            if self.pol:
                # shape (map1,map2; EE,EB,BE,BB; Q,U; ipix)
                map_out = np.zeros((2,4,3,self.npix))
                alms = np.zeros((2,4,3,self.alm_ord.getsize(3*self.nside-1)),dtype='complex128')
                # We only need to pick from three independent sets of alms
                rans = self._oosqrt2*(2*np.random.binomial(1,0.5,size=6*(ell+1))-1).reshape([3,2,ell+1])
                rans[:, 0, 0] *= self._sqrt2
                rans[:, 1, 0] = 0
                for im, ip, pk in self._rands_iterator():
                    alms[im,ip,1,idx] = rans[pk[0],0] + 1j*rans[pk[0],1]
                    alms[im,ip,2,idx] = rans[pk[1],0] + 1j*rans[pk[1],1]
                    if self.beam is not None:
                        map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside, pixwin=True, fwhm=self.beam)
                    else:
                        map_out[im,ip] = hp.alm2map(alms[im,ip], self.nside)
            else:
                rans = self._oosqrt2*(2*np.random.binomial(1,0.5,size=2*(ell+1))-1).reshape([2,ell+1])
                # Correct m=0 (it should be real and have twice as much variance)
                rans[0, 0] *= self._sqrt2
                rans[1, 0] = 0
                # Populate alms and transform to map
                alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),dtype='complex128')
                alms[idx] = rans[0] + 1j*rans[1]
                # TODO: we can save time on the SHT massively, since in this case 
                # there is no sum over ell!
                if self.beam is not None:
                    map_out = hp.alm2map(alms, self.nside, pixwin=True, fwhm=self.beam)
                else:
                    map_out = hp.alm2map(alms, self.nside,)
        return map_out
    
    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        if self.dsim_d['stats'] == 'Gaussian':
            return self._gen_gaussian_map(ell)
        elif self.dsim_d['stats'] == 'Z2':
            return self._gen_Z2_map(ell)
        else:
            raise ValueError("Only Gaussian and Z2 sims implemented")
    
    def _filt_default(self, mp_true):
        if self.mode == 0:
            if self.pol:
                assert(mp_true.shape==(2,4,3,self.npix))
                map_out = self.filt_d['mask'][None,None,None,:]*mp_true
            else:
                map_out = self.filt_d['mask']*mp_true
        elif self.mode == 3:
            if self.pol:
                assert(mp_true.shape==(2,4,3,self.npix))
                map_out = np.zeros((2,4,3,self.npix))
                if self.obsmat is not None:
                    # filter with the obsmat
                    for ii in range(2):
                        for jj in range(4):
                            if self.bin_mask is not None:
                                mp_true_masked = mp_true[ii,jj] * self.bin_mask
                            else:
                                mp_true_masked = mp_true[ii,jj]
                            mp_true_nest = hp.reorder(mp_true_masked, r2n=True)
                            mp_filt_nest = self.obsmat.apply(mp_true_nest)
                            if self.bin_mask is not None:
                                mp_filt = hp.reorder(mp_filt_nest, n2r=True) * self.bin_mask
                            else:
                                mp_filt = hp.reorder(mp_filt_nest, n2r=True)
                            map_out[ii,jj] = self.filt_d['mask'][None,None,None,:] * mp_filt
                elif self.mcut is not None:
                    # filter with mcut
                    for ii in range(2):
                        for jj in range(4):
                            if self.bin_mask is not None:
                                mp_true_masked = mp_true[ii,jj] * self.bin_mask
                            else:
                                mp_true_masked = mp_true[ii,jj]
                            alms_ = hp.map2alm(mp_true_masked, lmax=self.lmax, pol=True)
                            n_modes_to_filter = (self.mcut + 1) * (self.lmax + 1) - ((self.mcut + 1) * self.mcut) // 2
                            alms_[:, :n_modes_to_filter] = 0.
                            if self.bin_mask is not None:
                                mp_filt = hp.alm2map(alms_, nside=self.nside, lmax=self.lmax, pol=True) * self.bin_mask
                            else:
                                mp_filt = hp.alm2map(alms_, nside=self.nside, lmax=self.lmax, pol=True)
                            map_out[ii,jj] = self.filt_d['mask'][None,None,None,:] * mp_filt
            else: # this is spin 0
                if self.mcut is not None:
                    if self.bin_mask is not None:
                        mp_true_masked = mp_true * self.bin_mask
                    else:
                        mp_true_masked = mp_true
                    alms_ = hp.map2alm(mp_true_masked, lmax=self.lmax, pol=False)
                    n_modes_to_filter = (self.mcut + 1) * (self.lmax + 1) - ((self.mcut + 1) * self.mcut) // 2
                    alms_[:n_modes_to_filter] = 0.
                    if self.bin_mask is not None:
                        mp_filt = hp.alm2map(alms_, nside=self.nside, lmax=self.lmax, pol=False, mmax=self.lmax) * self.bin_mask
                    else:
                        mp_filt = hp.alm2map(alms_, nside=self.nside, lmax=self.lmax, pol=False, mmax=self.lmax)
                    map_out = mp_filt * self.filt_d['mask']
        elif self.mode == 2:
            if self.pol:
                assert(mp_true.shape==(2,4,3,self.npix))
                map_out = self.filt_d['mask'][None,None,None,:]*mp_true
            else:
                map_out = self.filt_d['mask']*mp_true
        else:
            raise ValueError("Error")
        return map_out
    
    def gen_deltasim(self, seed, ell):
        if self.mode in [0,3]: # mode 0 is the default
            dsim_true = self.dsim(seed, ell)
            # dsim_true is the map for seed and ell
        elif self.mode==1:
            dsim_true = self.dsim(seed, ell)
            if self.save_maps:
                # we save the map
                if self.outdir is None:
                    raise ValueError("You want to save the maps but forgot to define an output")
                outdir = self.outdir + '/ell%04i/'%ell
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                output = outdir + 'map_tqu_nside%i_%s_ell%04i_seed%0i_map%i.fits'
                for im, ip, pk in self._rands_iterator():
                    map_ = np.zeros((3,12*self.nside**2))
                    map_[1] = dsim_true[im,ip,0] # this is Q
                    map_[2] = dsim_true[im,ip,1] # this is U
                    if ip==0:
                        label = 'EE'
                    elif ip==1:
                        label = 'EB'
                    elif ip==2:
                        label = 'BE'
                    elif ip==3:
                        label = 'BB'
                    hp.write_map(output%(self.nside,label,ell,seed,im+1),map_,dtype=np.single, overwrite=True,column_units='K')
            dsim_filt = None
        if self.mode in [0,3]:
            dsim_filt = self.filt(dsim_true)
        elif self.mode==1:
            pass
        elif self.mode==2:
            dsim_true = np.zeros((2,4,2,self.npix))
            for im, ip, pk in self._rands_iterator():
                if ip==0:
                    label = 'EE'
                elif ip==1:
                    label = 'EB'
                elif ip==2:
                    label = 'BE'
                elif ip==3:
                    label = 'BB'
                name_file = self.indir + '/ell%04i/'%ell + self.in_name%(label,ell,seed,im+1)
                dsim_true[im,ip] = hp.read_map(name_file,field=(1,2))
            dsim_filt = self.filt(dsim_true)
        else:
            raise ValueError("Mode parameter invalid")
        return dsim_filt 
    
    def gen_deltasim_bpw(self, seed, ell):
        if self.mode in [0,3]:
            dsim = self.gen_deltasim(seed, ell)
            if self.pol:
                assert(dsim.shape==(2,4,3,self.npix))
                # cb has shape (EE,EB,BE,BB)_out, bpw_out, (EE,EB,BE,BB)_in
                cb = np.zeros((4,self.n_bins,4))
                pols = [(0,0),(0,1),(1,0),(1,1)]
                for ipol_in, (ip1,ip2) in enumerate(pols):
                    #tqu1 = np.concatenate((np.zeros((1,self.npix)), dsim[0,ipol_in]))
                    #tqu2 = np.concatenate((np.zeros((1,self.npix)), dsim[1,ipol_in]))
                    # anafast only outputs EB, not BE for 2 polarized input maps. We
                    # can generalize this using map2alm + alm2cl instead.
                    alm1 = hp.map2alm(dsim[0,ipol_in])[1:] # alm_EB
                    alm2 = hp.map2alm(dsim[1,ipol_in])[1:] # alm_EB
                    for ipol_out, (iq1,iq2) in enumerate(pols):
                        cells_ = hp.alm2cl(alm1[iq1], alm2[iq2],lmax=(3*self.nside-1))
                        cb[ipol_out, :, ipol_in] = self.bins.bin_cell(cells_[:self.lmax+1])
                        self.cb_ell[ipol_out,ipol_in,:] += cells_[:self.lmax+1] 
            else:
                cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter)[:self.lmax+1])
        elif self.mode==1 and self.gen_deltasim(seed, ell)==None:
            dsim = self.gen_deltasim(seed, ell)
            cb = None
        elif self.mode==2:
            dsim = self.gen_deltasim(seed, ell)
            if self.pol:
                assert(dsim.shape==(2,4,3,self.npix))
                # cb has shape (EE,EB,BE,BB)_out, bpw_out, (EE,EB,BE,BB)_in
                cb = np.zeros((4,self.n_bins,4))
                pols = [(0,0),(0,1),(1,0),(1,1)]
                for ipol_in, (ip1,ip2) in enumerate(pols):
                    tqu1 = np.concatenate((np.zeros((1,self.npix)), dsim[0,ipol_in]))
                    tqu2 = np.concatenate((np.zeros((1,self.npix)), dsim[1,ipol_in]))
                    # anafast only outputs EB, not BE for 2 polarized input maps. We
                    # can generalize this using map2alm + alm2cl instead.
                    alm1 = hp.map2alm(tqu1)[1:] # alm_EB
                    alm2 = hp.map2alm(tqu2)[1:] # alm_EB
                    for ipol_out, (iq1,iq2) in enumerate(pols):
                        cells_ = hp.alm2cl(alm1[iq1], alm2[iq2],lmax=(3*self.nside-1))
                        cb[ipol_out, :, ipol_in] = self.bins.bin_cell(cells_[:self.lmax+1])
                        self.cb_ell[ipol_out,ipol_in,:] += cells_[:self.lmax+1]
            else:
                cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        return cb
    
    def gen_Bbl_at_ell(self, ell):
        #remember to clear self.cb_ell when starting with a new ell
        if self.pol:
            self.cb_ell *= 0.0
        if self.mode in [0,2,3]:
            if self.comm is not None:
                if self.pol:
                    # Bbl has shape pol_out, bpw_out, pol_in
                    Bbl = np.zeros((int(self.nsim_per_ell/self.comm.size),4,self.n_bins,4))
                    Bbl_final = None # this is the receiving buffer
                else:
                    Bbl = np.zeros((int(self.nsim_per_ell/self.comm.size),self.n_bins))
                    Bbl_final = None # this is the receiving buffer
                counter = 0
                for i in range(self.comm.rank*self.nsim_per_ell//self.comm.size,(self.comm.rank+1)*self.nsim_per_ell//self.comm.size):
                    if self.comm.rank == 0:
                        sys.stdout.write(f'\rell={ell}/{self.lmax}: sim {i+1} of {self.nsim_per_ell}\n')
                        sys.stdout.flush()
                    seed = self.seed0 + ell*self.nsim_per_ell + i
                    cb = self.gen_deltasim_bpw(seed, ell)
                    Bbl[counter] = cb
                    counter += 1
                if self.pol:
                    # Here I have to gather Bbl and reduce cb_ell
                    if self.comm.rank == 0:
                        Bbl_final = np.zeros((self.comm.size, int(self.nsim_per_ell/self.comm.size),4,self.n_bins,4))
                    self.comm.Gather(Bbl, Bbl_final, root=0)
                    if self.comm.rank == 0:
                        Bbl = np.reshape(Bbl_final, (self.nsim_per_ell,4,self.n_bins,4))
                    #Bbl /= self.nsim_per_ell
    
                    cb_ell_ = np.zeros_like(self.cb_ell)
                    self.comm.Reduce(self.cb_ell, cb_ell_, op=MPI.SUM, root=0)
                    if self.comm.rank == 0:
                        self.cb_final[:,:,ell,:] = cb_ell_ / self.nsim_per_ell
                else:
                    # Here I have to gather Bbl and reduce cb_ell
                    if self.comm.rank == 0:
                        Bbl_final = np.zeros((self.comm.size, int(self.nsim_per_ell/self.comm.size),self.n_bins))
                    self.comm.Gather(Bbl, Bbl_final, root=0)
                    if self.comm.rank == 0:
                        Bbl = np.reshape(Bbl_final, (self.nsim_per_ell,self.n_bins))
                return np.mean(Bbl,axis=0), np.std(Bbl,axis=0)
            else:
                if self.pol:
                    # Bbl has shape pol_out, bpw_out, pol_in
                    Bbl_final = np.zeros((int(self.nsim_per_ell),4,self.n_bins,4))
                else:
                    Bbl_final = np.zeros((int(self.nsim_per_ell),self.n_bins))
                counter = 0
                for i in range(self.nsim_per_ell):
                    if i==0:
                        sys.stdout.write(f'\rell={ell}/{self.lmax}: sim {i+1} of {self.nsim_per_ell}\n')
                        sys.stdout.flush()
                    seed = self.seed0 + ell*self.nsim_per_ell + i
                    cb = self.gen_deltasim_bpw(seed, ell)
                    Bbl_final[counter] = cb
                    counter += 1
                if self.pol:
                    self.cb_final[:,:,ell,:] = self.cb_ell / self.nsim_per_ell
                #Bbl_final = np.reshape(Bbl_final, (self.nsim_per_ell,4,self.n_bins,4))
                #Bbl /= self.nsim_per_ell
                return np.mean(Bbl_final,axis=0), np.std(Bbl_final,axis=0)
        elif self.mode==1:
            for i in range(self.nsim_per_ell):
                sys.stdout.write(f'\rell={ell}/{self.lmax}: sim {i+1} of {self.nsim_per_ell}')
                sys.stdout.flush()
                seed = self.seed0 + ell*self.nsim_per_ell + i
                cb = self.gen_deltasim_bpw(seed, ell) # we only need to run this in mode=1
            return None, None
    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)
    def gen_Bbl_all(self):
        if self.mode in [0,2,3]:
            if self.interp_ells != False:
                # determine if interp_ells is either an integer or an array
                if hasattr(self.interp_ells, '__iter__'):
                    ls_sampled = []
                    il_sampled = []
                    for il, l in enumerate(self.get_ells()):
                        if l in self.interp_ells:
                            ls_sampled.append(l)
                            il_sampled.append(il)
                    ls_sampled = np.array(ls_sampled)
                    il_sampled = np.array(il_sampled)
                else:
                    ipl = self.interp_ells
                    ls_sampled = []
                    il_sampled = []
                    for il, l in enumerate(self.get_ells()):
                        if il%ipl == 0:
                            ls_sampled.append(l)
                            il_sampled.append(il)
                    ls_sampled = np.array(ls_sampled)
                    il_sampled = np.array(il_sampled)
            else:
                ls_sampled = self.get_ells()
                il_sampled = np.arange(len(ls_sampled))
            bpw_sampled = np.array([self.gen_Bbl_at_ell(l)[0] for l in ls_sampled])
            self.errors = {}
            #for l in ls_sampled:
            #    self.errors[l] = self.gen_Bbl_at_ell(l)[1]
            if self.comm is None:
                if self.pol:
                    # arr_out has shape pol_out, bpw_out, pol_in, ell_in
                    arr_out = np.zeros((4,self.n_bins,4,self.n_ells))
                    for idx, ils in enumerate(il_sampled):
                        arr_out[:,:,:,ils] = bpw_sampled[idx]   
                    if self.interp_ells != False:
                        for il, l in enumerate(self.get_ells()):
                            if l not in ls_sampled:
                                for ip in range(4):
                                    for ib in range(self.n_bins):
                                        for iq in range(4):
                                            #print('iq', iq, 'ib', ib, 'ip', ip, 'il', il)
                                            arr_out[iq,ib,ip,il] = np.interp(l, ls_sampled, bpw_sampled[:,iq,ib,ip])
                else: 
                    arr_out = np.zeros((self.n_bins, self.n_ells))
                    for idx, ils in enumerate(il_sampled):
                        arr_out[:,ils] = bpw_sampled[idx,:]
                    if self.interp_ells != False:
                        for il, l in enumerate(self.get_ells()):
                            if l not in ls_sampled:
                                for ib in range(self.n_bins):
                                    arr_out[ib,il] = np.interp(l, ls_sampled, bpw_sampled[:,ib])
            elif self.comm.rank == 0:
                if self.pol:
                    # arr_out has shape pol_out, bpw_out, pol_in, ell_in
                    arr_out = np.zeros((4,self.n_bins,4,self.n_ells))
                    for idx, ils in enumerate(il_sampled):
                        arr_out[:,:,:,ils] = bpw_sampled[idx]   
                    if self.interp_ells != False:
                        for il, l in enumerate(self.get_ells()):
                            if l not in ls_sampled:
                                for ip in range(4):
                                    for ib in range(self.n_bins):
                                        for iq in range(4):
                                            #print('iq', iq, 'ib', ib, 'ip', ip, 'il', il)
                                            arr_out[iq,ib,ip,il] = np.interp(l, ls_sampled, bpw_sampled[:,iq,ib,ip])
                else: 
                    arr_out = np.zeros((self.n_bins, self.n_ells))
                    for idx, ils in enumerate(il_sampled):
                        arr_out[:,ils] = bpw_sampled[idx,:]
                    if self.interp_ells != False:
                        for il, l in enumerate(self.get_ells()):
                            if l not in ls_sampled:
                                for ib in range(self.n_bins):
                                    arr_out[ib,il] = np.interp(l, ls_sampled, bpw_sampled[:,ib])
            else:
                arr_out = None
        elif self.mode==1:
            if self.interp_ells:
                ipl = self.interp_ells
                ls_sampled = []
                il_sampled = []
                for il, l in enumerate(self.get_ells()):
                    if il%ipl == 0:
                        ls_sampled.append(l)
                        il_sampled.append(il)
                ls_sampled = np.array(ls_sampled)
                il_sampled = np.array(il_sampled)
            else:
                ls_sampled = self.get_ells()
                il_sampled = np.arange(len(ls_sampled))
            bpw_sampled = np.array([self.gen_Bbl_at_ell(l) for l in ls_sampled]) # We only need to run this for mode=1
            arr_out = None
        return arr_out
