import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import THzToEv
from phonopy.structure.atoms import atom_data
import pickle
import time

# coherent neutron scattering lengths
BCOH = {'La': 8.24, 'Cu': 7.718, 'O': 5.803, 'Sr': 7.02}

# coherent scattering cross sections
SIGMA_COH = {'La': 8.53, 'Cu': 7.485, 'O': 4.2320, 'Sr': 6.19}
SIGMA_INC = {'La': 1.13, 'Cu': 0.550, 'O': 0.0008, 'Sr': 0.06}
SIGMA_TOT = {'La': 9.66, 'Cu': 8.03, 'O': 4.232, 'Sr': 6.25}

def gauss(x, amp, x0, sigma):
    return amp / sigma / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x - x0)**2 / sigma**2)

class PhonopyNeutron():
    def __init__(self, phonopy_disp, force_sets, pa='auto', symprec=1e-4, escale='meV'):

        self.cell = phonopy.load(phonopy_disp, primitive_matrix=pa)
        # can we get the primitive matrix here?

        self.phonon = phonopy.Phonopy(self.cell.unitcell,
                                      self.cell.supercell_matrix,
                                      primitive_matrix=self.cell.primitive_matrix,
                                      symprec=symprec)

        self.phonon.dataset = phonopy.file_IO.parse_FORCE_SETS(filename=force_sets)
        self.phonon.produce_force_constants()

        if escale == 'meV':
            self.scale = THzToEv * 1000
        elif escale == 'THz':
            self.scale = 1

        self.path = None
        self.labels = None
        self.are_bands_computed = False
        self.are_neutron_bands_computed = False
        self.nions = len(self.phonon.primitive.symbols)

    def set_path(self, path, primitive=False):
        if not primitive:
            new_path = []
            for p in path:
                new_path += [np.dot(p, self.phonon.primitive_matrix)]

            self.path = [new_path]

        if self.labels is None:
            self.labels = []
            for p in self.path[0]:
                self.labels.append('({:.2f},{:.2f},{:.2f})'.format(p[0], p[1], p[2]))

    def set_labels(self, labels):
        self.labels = labels

    def compute_bands(self, numpoints=101):
        if self.path is None:
            assert False,  "Band structure path not set, aborting"

        qpoints, connections = get_band_qpoints_and_path_connections(self.path, npoints=numpoints)
        self.phonon.run_band_structure(qpoints, path_connections=connections, labels=self.labels)

        bs = self.phonon.get_band_structure_dict()

        self.high_symmetry_points = []
        for segment in bs['distances']:
            self.high_symmetry_points.append(min(segment))
        self.high_symmetry_points.append(max(segment))

        self.segments = bs['distances']
        self.q = np.hstack(bs['distances'])
        self.en = np.vstack(bs['frequencies']) * self.scale
        self.num_bands = self.en.shape[1]
        self.are_bands_computed = True

    def plot_bands(self, ax=None, fmt='C0-', lw=None, bands=None):
        if self.are_bands_computed == False:
            self.compute_bands()

        if ax is None:
            f, ax = plt.subplots()

        if bands is None:
            ax.plot(self.q, self.en, fmt, lw=lw, zorder=0)
        else:
            ax.plot(self.q, self.en[:,bands], fmt, lw=lw, zorder=0)

        ax.set_xticks(self.high_symmetry_points)
        ax.set_xticklabels(self.labels)
        ax.set_xlim([self.q[0], self.q[-1]])

    def compute_neutron_bands(self, T=1, mesh=20, verbose=True):
        if self.path is None:
            assert False,  "Band structure path not set, aborting"
        elif self.are_neutron_bands_computed == False:
            self.compute_bands()
        
        start = time.time()

        bs = self.phonon.get_band_structure_dict()
        Q_prim = np.vstack(bs['qpoints'])
        # change all gamma points to (1e-5, 1e-5, 1e-5)
        Q_prim[np.all(np.round(Q_prim,5) == 0, axis=1)] = np.array([1e-5, 1e-5, 1e-5])

        self.phonon.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        self.phonon.run_dynamic_structure_factor(Q_prim, T, scattering_lengths=BCOH)
        self.dsf = self.phonon.dynamic_structure_factor
        self.are_neutron_bands_computed = True

        print('... Computed S(q,w) in {:.3f} seconds ...'.format(time.time() - start))

    def plot_neutron_bands(self,
                           ax=None,
                           verbose=True,
                           logcolors=False,
                           vmin=None,
                           vmax=None,
                           bands=None,
                           plottype='lines',
                           sigma=0.2,
                           energy_numpoints=101,
                           energy_limits=None,
                           colormap='viridis'):
        if ax is None:
            f, ax = plt.subplots()

        start = time.time()

        if vmin is None:
            vmin = self.dsf.dynamic_structure_factors.min()

        if vmax is None:
            vmax = self.dsf.dynamic_structure_factors.max()

        if energy_limits is None:
            en_axis = np.linspace(np.min(self.en), np.max(self.en), energy_numpoints)
        else:
            en_axis = np.linspace(energy_limits[0], energy_limits[1], energy_numpoints)     

        if logcolors:
            norm = LogNorm(vmin, vmax)
        else:
            norm = plt.Normalize(vmin, vmax)

        if bands is None:
            # If not specified, show all bands
            bands = np.arange(0, self.num_bands)

        if plottype == 'lines':
            # quick fix for a specific example of overlapping bands
            bands = bands[::-1]            
            for band in bands:
                x = self.q
                y = self.en[:,band]
                c = self.dsf.dynamic_structure_factors[:,band]

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments, cmap=colormap, norm=norm)
                lc.set_array(c)
                lc.set_linewidth(2)
                im = ax.add_collection(lc)
            print('... Generated line plots in {:.3f} seconds ...'.format(time.time() - start))
        elif plottype == 'map':
            I = np.zeros((self.q.shape[0], energy_numpoints))
            for q in range(self.q.shape[0]):
                sqw = np.zeros(energy_numpoints)
                for band in range(self.num_bands):
                    sqw += gauss(en_axis, self.dsf.dynamic_structure_factors[q,band], self.en[q,band], sigma)
                I[q, :] = sqw
            
            I[I<vmin] = np.nan
            print('... Generated map in {:.3f} seconds ...'.format(time.time() - start))                
            xx, yy = np.meshgrid(self.q, en_axis)
            im = ax.pcolor(xx, yy, I.T, cmap=colormap, norm=norm)
        else:
            assert False,  "Plot type not understood. Aborting..."


        ax.set_xlim([self.q[0], self.q[-1]])
        ax.set_ylim([en_axis.min(), en_axis.max()])
        ax.set_xticks(self.high_symmetry_points)
        ax.set_xticklabels(self.labels)
        ax.set_facecolor('xkcd:grey')
        return im

    def compute_dos(self, mesh=20, partial=True, sigma=None, weight='equal'):
        start = time.time()

        if sigma is not None:
            sigma = sigma / self.scale

        result = {}

        if not partial:
            if weight != 'equal':
                print('... Weighted DOS must be run with partial=True. Aborting ...')
                return -1

            self.phonon.run_mesh(mesh)
            self.phonon.run_total_dos(sigma=sigma)
            dos_phonopy = self.phonon.get_total_dos_dict()
            result['total_dos'] = dos_phonopy['total_dos']
            result['en'] = dos_phonopy['frequency_points'] * self.scale
            print('... Computed TOTAL DOS in {:.3f} seconds ...'.format(time.time() - start))
        else:
            self.phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
            self.phonon.run_projected_dos(sigma=sigma)

            pdos_phonopy = self.phonon.get_projected_dos_dict()
            result['en'] = pdos_phonopy['frequency_points'] * self.scale
            result['total_dos'] = 0

            w = {}
            for s, n in zip(self.phonon.primitive.symbols, self.phonon.primitive.numbers):
                mass = atom_data[n][3]
                if weight == 'neutron':
                    w[s] = SIGMA_TOT[s] / self.nions / mass
                else:
                    w[s] = 1 / self.nions

            for i, s in enumerate(self.phonon.primitive.symbols):
                if s in result:
                    result[s] += pdos_phonopy['projected_dos'][i] * w[s]
                else:
                    result[s] = pdos_phonopy['projected_dos'][i] * w[s]

                result['total_dos'] += pdos_phonopy['projected_dos'][i] * w[s]
            
            print('... Computed PARTIAL DOS in {:.3f} seconds ...'.format(time.time() - start))

        result['total_dos_norm'] = result['total_dos'] / max(result['total_dos'])
        self.dos = result

    def plot_dos(self, ax=None, colors=None, partial=True, total_color='xkcd:grey', total_alpha=0.5, fmt='C0-'):
        if ax is None:
            f, ax = plt.subplots()

        if partial:
            if colors is None:
                for s in np.unique(self.phonon.primitive.symbols):
                    ax.plot(self.dos['en'], self.dos[s], label=s)
            else:
                for s, fmt in zip(np.unique(self.phonon.primitive.symbols), colors):
                    ax.plot(self.dos['en'], self.dos[s], fmt, label=s)

            ax.fill_between(self.dos['en'], self.dos['total_dos'], color=total_color, alpha=total_alpha)
        else:
            ax.plot(self.dos['en'], self.dos['total_dos'], fmt)

    def get_sqw_xy(self, q_limits, nx, ny, mesh=20, T=1):
        qx = np.linspace(q_limits[0], q_limits[1], nx)
        qy = np.linspace(q_limits[2], q_limits[3], ny)
        
        Q_arr = np.zeros((nx, ny, 3))
        for i, x in enumerate(qx):
            for j, y in enumerate(qy):
                Q_arr[i, j, :] = [x, y, 0]

        Q_list = Q_arr.reshape((nx * ny, 3))
        Q_prim = np.dot(Q_list, self.phonon.primitive_matrix)
        Q_prim[np.all(np.round(Q_prim,5) == 0, axis=1)] = np.array([1e-5, 1e-5, 1e-5])

        print('... Starting S(q,w) computation on {} q values ...'.format(nx * ny))
        start = time.time()
        self.phonon.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        self.phonon.run_dynamic_structure_factor(Q_prim, T, scattering_lengths=BCOH)
        print('... Finished S(q,w) computation in {:.3f} seconds ...'.format(time.time() - start))

        # put everything in a dict
        result = {}
        
        dsf = self.phonon.dynamic_structure_factor
        en = dsf.frequencies * self.scale
        sqw = dsf.dynamic_structure_factors
        num_bands = len(en[0])
        
        result['bands'] = en.reshape(nx, ny, num_bands)
        result['sqw'] = sqw.reshape(nx, ny, num_bands)
        result['qx'] = qx
        result['qy'] = qy
        result['nx'] = nx
        result['ny'] = ny
        return result
        
def get_xy_colormap(sqw, energy, sigma=0.5):
    print('... Generating colormap ...')
    
    I = np.zeros((sqw['nx'], sqw['ny']))
    start = time.time()
    for i in range(sqw['nx']):
        for j in range(sqw['ny']):
            sqw_at_e = 0
            for en0, sqw0 in zip(sqw['bands'][i, j], sqw['sqw'][i, j]):
                sqw_at_e += gauss(energy, sqw0, en0, sigma)

            I[i, j] = sqw_at_e
    
    xx, yy = np.meshgrid(sqw['qx'], sqw['qy'])
    print('... Generated colormap in {:.3f} seconds ...'.format(time.time() - start))
    return xx, yy, I.T

if __name__ == '__main__':
    import os
    os.chdir('C:/Users/ttejs/Dropbox/ILL/simulation/phonon/')
    plt.style.use('seaborn-paper')

    sim = 'lto'
    lto = PhonopyNeutron(sim + '/phonopy_disp.yaml', sim + '/FORCE_SETS')
    lto.compute_dos(partial=True, weight='equal', sigma=0.5)
    lto.plot_dos()
    #lto.compute_dos(partial=False, weight='equal', sigma=0.5)
    #plt.plot(lto.dos['en'], lto.dos['total_dos'], 'k-')
    #lto.compute_dos(partial=False)
    #plt.plot(lto.dos['en'], lto.dos['total_dos'])
    plt.show()

