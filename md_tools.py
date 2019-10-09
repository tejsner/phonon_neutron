import numpy as np
import pylab as plt
import os
import pickle
import time
from numpy.fft import fft, ifft, fftshift, ifftshift
from matplotlib.colors import LogNorm

# masses from VASP pseudopotentials (since these are the ones used in simulation)
MASS = {'Cu': 63.546, 'La': 138.900, 'O': 16.000, 'Sr': 87.620}

# coherent neutron scattering lengths
BCOH = {'La': 8.24, 'Cu': 7.718, 'O': 5.803, 'Sr': 7.02}

# coherent scattering cross sections
SIGMA_COH = {'La': 8.53, 'Cu': 7.485, 'O': 4.2320, 'Sr': 6.19}
SIGMA_INC = {'La': 1.13, 'Cu': 0.550, 'O': 0.0008, 'Sr': 0.06}
SIGMA_TOT = {'La': 9.66, 'Cu': 8.03, 'O': 4.232, 'Sr': 6.25}

# THz to eV conversion
PlanckConstant = 4.13566733e-15 # [eV s]
THzToEv = PlanckConstant * 1e12 # [eV]

# constants
# https://physics.nist.gov/cuu/Constants/
kb = 1.380649e-23         # [J/K]
amu = 1.66053906660e-27   # [kg]

def dist(r1, r2, cell=None):
    d = r2 - r1
    d[d < -0.5] += 1
    d[d > 0.5] -= 1
    if cell is not None:
        d = np.dot(d, cell)    
    return d

def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

def linear_convolution(x, signal, sigma, sigma_lin=0, sigma_range=3):
    """
    Convolutes a signal with a gaussian a width that depends on the x-axis
    sigma_new = sigma + sigma_lin*x
    Requires equally spaced data
    Parameters:
        signal: data.
        x: x-axis
        sigma: width of gaussian
        sigma_lin: linear parameter. Default behaviour is no linear dependance.
        sigma_range: cutoff for the Gaussian function. Default is 3 sigma.
    Returns:
        x, y_new: x, new y-values
    """
    print("... Computing Convolution ...")
    dx = x[1]-x[0] # assume equally spaced data
    if sigma_lin == 0:
        print("\tNormal convolution (no linear term). Length of signal: {}".format(len(signal)))
        print("\tsigma = {:.3f} meV ({:.3f} meV FWHM)".format(sigma, 2.3548*sigma))
        gx = np.arange(-sigma_range*sigma, sigma_range*sigma, dx)
        gaussian = 1/np.sqrt(2.0*np.pi)/sigma*np.exp(-0.5*(gx/sigma)**2)
        return x, np.convolve(signal, gaussian, mode='same')*dx
    else:
        # convolute with a different gaussian at each point
        # much more expensive, but still fast even with several thousand points
        sigma_min = sigma + sigma_lin*min(x)
        sigma_max = sigma + sigma_lin*max(x)
        print("\tLinear Convolution. Length of signal: {}".format(len(signal)))
        print("\tsigma_min = {:.3f} meV ({:.3f} meV FWHM)".format(sigma_min, 2.3548*sigma_min))
        print("\tsigma_max = {:.3f} meV ({:.3f} meV FWHM)".format(sigma_max, 2.3548*sigma_max))
        y_new = np.zeros(len(x))
        for i, x_val in enumerate(x):
            sig = sigma + sigma_lin*x_val
            gx = np.arange(-sigma_range*sig, sigma_range*sig, dx)
            gaussian = 1/np.sqrt(2.0*np.pi)/sig*np.exp(-0.5*(gx/sig)**2)
            y_new[i] = np.convolve(signal, gaussian, mode='same')[i]*dx
        return x, y_new

class VaspMD:
    def __init__(self, xdatcar, dt=1, frames=None, is_poscar=False):
        print('... Loading XDATCAR ...')
        self.read_xdatcar(xdatcar, dt=dt)

        if frames is not None:
            self.position = self.position[frames,:,:]
        
        self.nframes = self.position.shape[0]

        self.velocity = None
        self.vacf = None
        self.dos = None

        # we have some methods that are useful for POSCAR files
        # so we allow loading them eventhough most methods will ot work
        if not is_poscar:
            self.__compute_axes()

    def prime_check(self):
        print('... Running primality check ...')
        found = False
        skip_frames = 0
        while not found:
            if not (is_prime(self.nframes-1-skip_frames) or is_prime(2*(self.nframes-1-skip_frames)-1)):
                found = True
            else:
                skip_frames += 1

        print('\tskip {} frames to avoid slow FFTs'.format(skip_frames))

    def append(self, other):
        if self.dt != other.dt:
            print('... WARNING: different time steps ...')

        self.position = np.append(self.position, other.position, axis=0)
        self.nframes = self.position.shape[0]

        # recompute axes and delete information about vacf, velocity and DOS
        self.__compute_axes()
        self.vacf = None
        self.velocity = None
        self.dos = None

    def skip_frames(self, skip=0):
        self.position = self.position[skip:,:,:]
        self.nframes = self.position.shape[0]
        self.__compute_axes()
        
    def read_xdatcar(self, file, dt=1):
        """
        Open XDATCAR file from VASP
        timestep dt is supplied in fs
        """
        with open(file) as f:
            self.dt = dt # femtoseconds
            self.name = f.readline()
            scale = float(f.readline())
            a1 = [float(i) for i in f.readline().split()]
            a2 = [float(i) for i in f.readline().split()]
            a3 = [float(i) for i in f.readline().split()]
            self.cell = np.vstack((a1, a2, a3))*scale # angstrom

            symbols = f.readline().split()
            num_atoms = [int(i) for i in f.readline().split()]
            self.nions = sum(num_atoms)

            self.volume = np.linalg.det(self.cell)
            self.rho = self.nions / self.volume

            self.symbol = []
            self.num_atoms = dict()
            for s, n in zip(symbols, num_atoms):
                self.symbol += [s]*n
                self.num_atoms[s] = n

            self.mass = []
            self.bcoh = []
            self.sigma_coh = []
            self.sigma_inc = []
            self.sigma_tot = []
            self.num_atom_type = []

            for s in self.symbol:
                self.mass.append(MASS[s])
                self.bcoh.append(BCOH[s])
                self.sigma_coh.append(SIGMA_COH[s])
                self.sigma_inc.append(SIGMA_INC[s])
                self.sigma_tot.append(SIGMA_TOT[s])
                self.num_atom_type.append(self.symbol.count(s))

            self.mass = np.array(self.mass)
            self.bcoh = np.array(self.bcoh)
            self.sigma_coh = np.array(self.sigma_coh)
            self.sigma_inc = np.array(self.sigma_inc)
            self.sigma_tot = np.array(self.sigma_tot)
            self.num_atom_type = np.array(self.num_atom_type)
            self.symbol = np.array(self.symbol)

            self.position = []
            line = f.readline()
            while line:
                if 'Direct' in line:
                    pos = np.zeros((self.nions, 3))
                    for i in range(self.nions):
                        pos[i,:] = [float(i) for i in f.readline().split()]
                    self.position.append(pos)

                line = f.readline()

            self.position = np.array(self.position)

    def compute_velocity(self, method=2):
        """
        Calculate velocities given that we know positions and time step
        Uses the definition of velocities from the Verlet algorithm
        """
        print('... Computing Velocity ...')
        self.nvel = self.nframes-1
        self.velocity = np.zeros((self.nvel, self.nions, 3))

        # from 0 to nframes-2 (high prec)
        dist = self.position[0:-2,:,:] - self.position[2:,:,:]
        dist[dist < -0.5] += 1
        dist[dist > 0.5] -= 1
        self.velocity[:-1,:,:] = np.dot(dist, self.cell)/2/self.dt

        # nframes-1 (low prec)
        dist = self.position[-1,:,:] - self.position[-2,:,:]
        dist[dist < -0.5] += 1
        dist[dist > 0.5] -= 1
        self.velocity[-1,:,:] = np.dot(dist, self.cell)/self.dt

        self.__compute_axes()
    
    def compute_temperature(self, tebeg=300, method=1):
        if self.velocity is None:
            self.compute_velocity()

        print('... Computing Temperature ...')
        self.temperature = (self.mass[None,:]*(self.velocity*self.velocity).sum(axis=2)).sum(axis=1)
        self.temperature *= amu*1e10/3/kb/(self.nions-1)
        self.__compute_axes()

    def compute_vacf(self, method=3, weight='mass'):
        if self.velocity is None:
            self.compute_velocity()
        
        start = time.time()
        print('... Computing VACF (weight = {}) ...'.format(weight))

        # number of velocities is one less than the number of frames
        self.vacf = np.zeros((self.nvel,), dtype=float)
        
        if weight == 'equal':
            m = np.ones((self.nions,), dtype=float)
        elif weight == 'mass':
            m = self.mass/np.mean(self.mass)
        elif weight == 'neutron_bcoh':
            m = self.bcoh/np.mean(self.bcoh)
        else:
            print('\tWeight not understood, defaulting to "equal"')
            m = np.ones((self.nions,), dtype=float)

        if method == 1:
            # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
            # direct method, slow
            for t in range(self.nvel):
                for j in range(self.nvel-t):
                    for i in range(self.nions):
                        self.vacf[t] += np.dot(self.velocity[j,i,:],self.velocity[j+t,i,:])*m[i]/(self.nvel-t)
                self.vacf[t] = self.vacf[t]/3/self.nions
        elif method == 2:
            # (xx, natoms, 3) * (1, natoms, 1) -> (xx, natoms, 3)
            # direct method, significantly faster (uses numpy only for each time step)
            for t in range(self.nvel):
                self.vacf[t] = (self.velocity[:(self.nvel-t),:,:]*self.velocity[t:,:,:]*m[None,:,None]).mean()
        elif method == 3:
            # FCA method from MDANSE using FFTs
            # Significant speedup and identical spectrum
            print('\tLength of FFT spectrum: {} (can be slow if prime)'.format(len(self.velocity)))
            X = np.fft.fft(self.velocity, 2*self.nvel, axis=0)
            corr = np.fft.ifft(np.conjugate(X)*X, axis=0)[0:self.nvel,:,:]
            corr = np.real(corr)
            norm = self.nvel - np.arange(self.nvel)
            corr = corr/norm[:,None,None]
            # apply weights
            for i in range(self.nions):
                corr[:,i,:] *= m[i]
            
            # partial VACF (per atom)
            # averaging over directions is equivalent to performing a dot product
            # maybe you can do a projected VACF here? Or do we project earlier?
            self.pvacf = np.mean(corr, axis=2)
            
            # total VACF 
            self.vacf = np.mean(self.pvacf, axis=1)
        
        self.__compute_axes()
        end = time.time()
        print('\tComputed VACF in {} seconds'.format(end-start))
        
    def compute_dos(self, sigma=0.3, weight='neutron_sqw', method=1):
        if self.vacf is None:
            self.compute_vacf(weight='mass')
        
        start = time.time()
        print('... Computing DOS (weight = {}) ...'.format(weight))

        # symmetrize partial VACF
        # S_ab(t) = S*_ba(-t)
        signal = np.concatenate((self.pvacf[-1:0:-1,:], self.pvacf), axis=0)
        
        if sigma is not None:
            # gaussian window (input sigma is meV in frequency domain)
            sigma_f = sigma/THzToEv/1000 # [THz] = [1/ps]
            sigma_t = 1/np.pi/2/(sigma_f*0.001) # [fs]
            x = np.arange(-(self.nvel-1), self.nvel, 1)*self.dt
            print('\tLength of FFT spectrum: {} (can be slow if prime)'.format(len(x)))
            # has to be normalized such that window=1 at t=0 (rather than a Gaussian normalized to area 1)
            window = np.exp(-0.5*(x/sigma_t)**2)
            self.window = window
            print('\tDOS smoothed by sigma = {:.2f} ps | {:.2f} THz | {:.2f} meV ({:.2f} meV FWHM)'.format(sigma_t*1e-3, sigma_f, sigma, sigma*2.3548))
        else:
            # no smoothing
            window = np.ones(2*self.nvel-1)
            print('\tNo smoothing of DOS')
        
        # do the FFT
        fftSignal = self.dt/2/np.pi*fftshift(fft(ifftshift(signal*window[:,None],axes=0),axis=0),axes=0)
        self.pdos = fftSignal.real
        self.pdos /= self.nions

        # weigh the pdos 
        if weight == 'equal':
            w = np.ones((self.nions,), dtype=float)
        elif weight == 'mass':
            w = self.mass/np.mean(self.mass)
        elif weight == 'neutron':
            w = self.bcoh/np.mean(self.bcoh)
        elif weight == 'neutron_coh_over_mass':
            w = self.sigma_coh/self.mass / np.mean(self.sigma_coh/self.mass)
        elif weight == 'neutron_coh':
            w = self.sigma_coh / np.mean(self.sigma_coh)
        elif weight == 'neutron_sqw':
            # this is the correct one to compare with tof data
            w = self.sigma_tot/self.mass

        self.pdos = self.pdos * w[None,:]
        self.dos = np.sum(self.pdos, axis=1)

        self.__compute_axes()
        end = time.time()
        print('\tComputed DOS in {} seconds'.format(end-start))

        # genrate pdos dictionary for easier plotting
        self.pdos_dict = {}
        for i, s in enumerate(self.symbol):
            if s in self.pdos_dict:
                self.pdos_dict[s] += self.pdos[:,i]
            else:
                self.pdos_dict[s] = self.pdos[:,i]

        self.pdos_dict['omega'] = self.omega
        self.pdos_dict['total_dos'] = self.dos

    def get_atomic_dos_dict(self, sigma=None):
        return self.pdos_dict
    
    def __compute_axes(self):
        """
        Computes time and frequency axes used for plotting
        """
        self.time = np.arange(self.nframes)*self.dt
        self.vtime = self.time[1:] # we only have n-1 velocities
        
        dw = 1/2/(self.nframes-1)/self.dt # 1/fs
        w = np.arange(self.nframes-1)*dw
        self.omega = np.concatenate((-w[-1:0:-1], w))
        self.omega *= THzToEv*1000*1000 #meV

    def print_units(self):
        print("... Units ...")
        print("\ttime: fs")
        print("\tdistance: angstrom")
        print("\tfrequency: meV")
        print("\tVACF: angstrom^2/fs (multiply by 10000 to get MDANSE units)")
        print("\tDOS: angstrom^2/fs^2 (multiply by 10 to get MDANSE units)")

    def compute_pdf(self, nbins=1000, pdf_range=(0,12), weight='neutron_norm'):
        """
        Computes the pair distribution function
        """
        start = time.time()
        print('... Computing PDF (weight = {}) ...'.format(weight))

        # create structure containing information about each pair
        # shape: (npairs, [b_a,b_b,N_a,N_b,c_a,c_b])
        # npairs = nions*(nions-1)/2
        pair_indices = []
        pair_partial = []
        for i in range(self.nions):
            for j in range(i+1, self.nions):
                pair_indices.append([i,j])
                # we sort the two atomic species such that the partials are unique
                partial = sorted([self.symbol[i], self.symbol[j]])
                pair_partial.append(partial[0] + partial[1])

        pair_indices = np.array(pair_indices, dtype=int)
        pair_partial = np.array(pair_partial, dtype='U4')
        num_pairs = pair_indices.shape[0]

        # build dictionary of partial factors,
        # where each key is a partial string and the value is a list:
        # [N_a, N_b, b_a, b_b, c_a, c_b]
        pf = {}
        partial_label, partial_index = np.unique(pair_partial, return_index=True)

        for p, i in zip(partial_label, partial_index):
            ii = pair_indices[i]
            b_a, b_b = BCOH[self.symbol[ii[0]]], BCOH[self.symbol[ii[1]]]
            N_a, N_b = self.num_atoms[self.symbol[ii[0]]], self.num_atoms[self.symbol[ii[1]]]
            c_a, c_b = N_a/self.nions, N_b/self.nions
            pf[p] = [N_a, N_b, b_a, b_b, c_a, c_b]

        # compute b_avg_squared for normalizataion
        b_avg_squared = 0
        for s in np.unique(self.symbol):
            b_avg_squared += BCOH[s]*self.num_atoms[s]/self.nions
        b_avg_squared = b_avg_squared**2

        # find all distances for all frames
        # this is the time-consuming part of the code
        distances = np.zeros((self.nframes, num_pairs))
        
        for t in range(self.nframes):
            d = self.position[t,pair_indices[:,1],:] - self.position[t,pair_indices[:,0],:]
            d[d < -0.5] += 1
            d[d > 0.5] -= 1
            d = np.dot(d, self.cell)
            d = np.linalg.norm(d, axis=1)
            distances[t,:] = d

        # histogram distances depending on their partials (alpha,beta)
        g_ab = {}
        g_total = np.zeros(nbins)

        for partial in partial_label:
            distances_partial = distances[:,pair_partial == partial].ravel()
            g_ab[partial], bin_edges = np.histogram(distances_partial, bins=nbins, range=pdf_range)
            r = (bin_edges[:-1] + bin_edges[1:]) / 2
            dr = bin_edges[1] - bin_edges[0]
            Vs = 4*np.pi*r**2*dr
            
            N_a, N_b = pf[partial][0], pf[partial][1]
            b_a, b_b = pf[partial][2], pf[partial][3]
            c_a, c_b = pf[partial][4], pf[partial][5]
            
            g_ab[partial] = g_ab[partial]*2*self.volume/(N_a*N_b)/self.nframes/Vs
            
            if weight == 'neutron':
                # Eq (10) in Keen et al. (2000)
                g_ab[partial] = (g_ab[partial]-1)*c_a*c_b*b_a*b_b
                g_total += g_ab[partial]
            elif weight == 'neutron_norm':
                # Eq (17) in Keen et al. (2000)
                g_ab[partial] = (g_ab[partial]-1)*c_a*c_b*b_a*b_b/b_avg_squared
                g_total += g_ab[partial]
            elif weight == 'equal':
                g_ab[partial] = g_ab[partial]*c_a*c_b
                g_total += g_ab[partial]
            elif weight == 'none':
                g_total += g_ab[partial]
            else:
                print('\tWeight not recogniced, defaulting to equal')
                g_ab[partial] = g_ab[partial]*c_a*c_b
                g_total += g_ab[partial]
        
        self.ppdf = g_ab
        self.pdf = g_total
        self.pdf_x = r # PDF x-axis

        end = time.time()
        print('\tComputed PDF in {} seconds'.format(end-start))

    def find_pairs(self, l1, l2, r, t=0, verbose=True):
        """ 
        finds pairs of l1 and l2 within a distance 
        d < r where d is a three dimensional vector
        performed only at time step t
        returns an array with pairs of atomic indices
        """
        ii = np.where(self.symbol == l1)[0]
        jj = np.where(self.symbol == l2)[0]
        list_pairs = []

        for i in ii:
            for j in jj:
                if (l1 == l2) and (j > i):
                    # inefficient way of dealing with equal labels
                    # for the situations when dealing with identical atoms
                    break

                # find distance
                d = self.position[t,j,:] - self.position[t,i,:]
                d[d < -0.5] += 1
                d[d > 0.5] -= 1
                d = np.dot(d, self.cell)
                                
                # check if d < r along all three components
                success = True
                for k in range(3):
                    if np.abs(d[k]) > r[k]:
                        success = False

                if success:
                    list_pairs.append([i, j])

        if verbose:
            print('Found {} occurences of {}-{} pairs'.format(len(list_pairs), l1, l2))
        
        return np.array(list_pairs)

    def get_distances(self, pairs, projection='equal', return_3d=False, print_avg=True):
        """
        Get distances between pairs of atoms. Can be used to generate histograms
        """
        num_pairs = len(pairs)
        distances = np.zeros((num_pairs, self.nframes, 3))
        for t in range(self.nframes):
            for i, p in enumerate(pairs):
                d = self.position[t,p[1],:] - self.position[t,p[0],:]
                d[d < -0.5] += 1
                d[d > 0.5] -= 1
                d = np.dot(d, self.cell)
                distances[i,t,:] = d

        # for each pair subtract the average distance
        if projection == 'prm':
            for p in range(num_pairs):
                distances[p,:,:] = distances[p,:,:] - np.average(distances[p,:,:], axis=0)

        distance_histogram_3d = distances.reshape((self.nframes*num_pairs, 3))
        distance_histogram_1d = np.linalg.norm(distance_histogram_3d, axis=1)
        
        if print_avg:
            mean = np.mean(distance_histogram_1d)
            std = np.std(distance_histogram_1d)
            print('AVG: {} angstrom, STD: {} angstrom'.format(mean, std))

        if return_3d:
            return distance_histogram_3d
        else:
            return distance_histogram_1d
        

    def get_segmented_path(self, atom, segments=10, axis='xy',):
        """
        Get the path of a single atom, sampled in a number of segments.
        Currently projects to xy plane.
        Useful for looking at diffusion.
        """
        traj = self.position[:,atom,:]

        x = traj[:,0]
        y = traj[:,1]

        s_x = []
        s_y = []
        l = self.nframes
        for i in range(segments):
            s_x.append(x[l*i//segments])
            s_y.append(y[l*i//segments])
    
        s_x.append(x[-1])
        s_y.append(y[-1])

        return s_x, s_y

def find_octahedra(md, t=0, a1='Cu', a2='O', r1=[2.1,2.1,1], r2=[1,1,2.7]):
    # find pairs of cu-oeq and cu-oap
    cu_oeq = md.find_pairs(a1, a2, r=r1, t=t)
    cu_oap = md.find_pairs(a1, a2, r=r2, t=t)

    # cu indices
    cu_indices = np.unique(cu_oeq[:,0])

    # our octahedra are defined by 7 indices
    # 0: Cu
    # 1-4: O-eq in order (-1,-1), (-1,+1), (+1,+1), (+1,-1)
    # 5-6: O-ap in order (-1), (+1)
    octahedra = np.zeros((len(cu_indices), 7), dtype=int)

    for i, cu_pos in enumerate(cu_indices):
        octahedra[i,0] = cu_pos
    
        # check for equatorial oxygen
        for row in cu_oeq:
            if cu_pos == row[0]:
                oeq_pos = row[1]
                d = dist(md.position[t,cu_pos,:], md.position[t,oeq_pos,:])
                d_sign = (np.sign(d[0]), np.sign(d[1]))
                
                if d_sign == (-1, -1):
                    octahedra[i,1] = oeq_pos
                elif d_sign == (-1, +1):
                    octahedra[i,2] = oeq_pos
                elif d_sign == (+1, +1):
                    octahedra[i,3] = oeq_pos
                elif d_sign == (+1, -1):
                    octahedra[i,4] = oeq_pos

        # check for apical oxygen
        for row in cu_oap:
            if cu_pos == row[0]:
                oap_pos = row[1]
                d = dist(md.position[t,cu_pos,:], md.position[t,oap_pos,:])
                d_sign = np.sign(d[2])

                if d_sign == -1:
                    octahedra[i,5] = oap_pos
                elif d_sign == 1:
                    octahedra[i,6] = oap_pos
    
    return octahedra

def get_octahedral_tilts(md, octahedra, t=0, verbose=False, symmetry=None, summary=False):
    # (num_octahedra, [Q1*4 + Q2*4])
    Q_list = np.zeros((octahedra.shape[0], 8))

    for i, octa in enumerate(octahedra):
        cu = octa[0]   

        # Apical Oxygen
        oap_below = octa[5]
        oap_above = octa[6]
        
        d = dist(md.position[t,cu,:], md.position[t,oap_above,:], cell=md.cell)
        Q1_above = np.arcsin(d[0]/d[2])*180/np.pi
        Q2_above = np.arcsin(d[1]/d[2])*180/np.pi

        d = dist(md.position[t,cu,:], md.position[t,oap_below,:], cell=md.cell)
        Q1_below = np.arcsin(d[0]/d[2])*180/np.pi
        Q2_below = np.arcsin(d[1]/d[2])*180/np.pi

        # Equatorial Oxygen
        oeq_1, oeq_2, oeq_3, oeq_4 = octa[1:5]
        d1 = dist(md.position[t,cu,:], md.position[t,oeq_1,:], cell=md.cell)
        d2 = dist(md.position[t,cu,:], md.position[t,oeq_2,:], cell=md.cell)
        d3 = dist(md.position[t,cu,:], md.position[t,oeq_3,:], cell=md.cell)
        d4 = dist(md.position[t,cu,:], md.position[t,oeq_4,:], cell=md.cell)

        Q1_eq1 = -np.arcsin(d1[2]/d1[0]/2 + d2[2]/d2[0]/2)*180/np.pi
        Q1_eq2 = -np.arcsin(d3[2]/d3[0]/2 + d4[2]/d4[0]/2)*180/np.pi

        Q2_eq1 = -np.arcsin(d1[2]/d1[1]/2 + d4[2]/d4[1]/2)*180/np.pi
        Q2_eq2 = -np.arcsin(d2[2]/d2[1]/2 + d3[2]/d3[1]/2)*180/np.pi
        
        Q_list[i,:] = Q1_above, Q1_below, Q1_eq1, Q1_eq2, Q2_above, Q2_below, Q2_eq1, Q2_eq2
        
        if verbose:
            print("Cu {} | Q1eq: {:+.2f}, {:+.2f} | Q2eq: {:+.2f}, {:+.2f}".format(cu, Q1_eq1, Q1_eq2, Q2_eq1, Q2_eq2))
            print("Cu {} | Q1ap: {:+.2f}, {:+.2f} | Q2ap: {:+.2f}, {:+.2f}".format(cu, Q1_above, Q1_below, Q2_above, Q2_below))
            
    # use to determine symmetry
    if summary:
        print("(0:4)   Average Q1: {:+.5f} | Average Q2: {:+.5f}".format(Q_list[0:4,0:4].mean(), Q_list[0:4,4:8].mean()))
        print("(4:8)   Average Q1: {:+.5f} | Average Q2: {:+.5f}".format(Q_list[4:8,0:4].mean(), Q_list[4:8,4:8].mean()))
        print("(8:12)  Average Q1: {:+.5f} | Average Q2: {:+.5f}".format(Q_list[8:12,0:4].mean(), Q_list[8:12,4:8].mean()))
        print("(12:16) Average Q1: {:+.5f} | Average Q2: {:+.5f}".format(Q_list[12:16,0:4].mean(), Q_list[12:16,4:8].mean()))

    if symmetry is None:
        return Q_list
    else:
        return Q_list*symmetry

def get_octahedral_tilt_histogram(md, octahedra, sym=None):
    Q1 = []
    Q2 = []
    for t in range(md.nframes):
        tilts = get_octahedral_tilts(md, octahedra, t=t, verbose=False, symmetry=sym)
        Q1.append(tilts[:,0:4])
        Q2.append(tilts[:,4:8])

    Q1 = np.array(Q1)
    Q2 = np.array(Q2)

    Q1 = Q1.mean(axis=2)
    Q2 = Q2.mean(axis=2)

    return Q1.ravel(), Q2.ravel()

def plot_pdos(vaspmd_object, ax=None, colors=None, total_color='xkcd:grey', total_alpha=0.5, in_nm2_ps2=False):
    dos = vaspmd_object.get_atomic_dos_dict()

    if colors is not None:
        if len(colors) != len(np.unique(vaspmd_object.symbol)):
            print("Length of color array inconsistent with number of partials. Reverting to default.")
            colors = None

    if colors is None:
        colors = []
        for i, s in enumerate(np.unique(vaspmd_object.symbol)):
            colors.append('C{}'.format(i))

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot()
    
    if in_nm2_ps2:
        for i, s in enumerate(np.unique(vaspmd_object.symbol)):
            ax.plot(dos['omega'], dos[s]*10000, label=s, color=colors[i])

        ax.fill_between(dos['omega'], dos['total_dos']*10000, color=total_color, alpha=total_alpha)
    else:
        for i, s in enumerate(np.unique(vaspmd_object.symbol)):
            ax.plot(dos['omega'], dos[s], label=s, color=colors[i])

        ax.fill_between(dos['omega'], dos['total_dos'], color=total_color, alpha=total_alpha)
