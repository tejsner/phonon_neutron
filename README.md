# phonon_neutron
Tools to compare neutron scattering data with phonon calculations (MD or Frozen Phonons). Used in my [thesis](https://github.com/tejsner/thesis/releases) work.

## Requirements
* Python 3
* numpy
* matplotlib
* [phonopy](https://atztogo.github.io/phonopy/)

## Usage

During my thesis, a selection of Python classes were developed with the intention of generalizing some of the tasks required to get the correct neutron weights out of simulations. While software such as [MDANSE](https://mdanse.org/) does a good job with respect to molecular dynamics, I wanted something focussed on analyzing phonons specifically from different levels of theory (MD and "Frozen Phonons"). Two modules are found here:

### md_tools.py
This module is used to perform various tasks on molecular dynamics trajectories as obtained from VASP. VASP trajectories are saved in `XDATCAR` files, which can be analysed, for example, in the following way:

```
from md_tools import VaspMD
md_data = VaspMD('XDATCAR', dt=1)
md_data.compute_velocity()
md_data.compute_temperature()
md_data.compute_pdf()
md_data.compute_dos(sigma=0.5)
```

Line 1 imports the module, line 2 reads the VASP trajectory (specifiying the time step `dt`) and lines 3-6 computes velocity, temperature, the pair distribution function and the phonon density of states with a Gaussian smearing width of sigma = 0.5 meV. Everything is now saved in the `md_data` object, and can be plotted, for example, using matplotlib in the following way:

```
import matplotlib.pytplot as plt
# temperature
plt.figure()
plt.plot(md_data.vtime, md_data.temperature)
# PDF
plt.figure()
plt.plot(md_data.pdf_x, md_data.pdf)
# DOS
plt.figure()
plt.plot(md_data.omega, md_data.dos)
```

Many additional features are present in this module and can be found by inspecting the code. A current limitation is that it only contains neutron cross sections for the atomic species used in my thesis (La, Sr, Cu, O), but it is a fairly simple procedure to add scattering lengths at the top of `md_tools.py`.

### phonopy_tools.py
This module contains a number of helper functions used to manipulate output from a [Phonopy](https://atztogo.github.io/phonopy/) calculation. In particular, I wanted easy access to neutron-weighted band structure plotted in different ways as shown throughout my thesis (in particular chapter 6). In order for the code to load Phonopy data, we require two files from such a phonon calculation:

* `phonopy_disp.yaml`: Contains information about the input structure and phonon calculation.
* `FORCE_SETS`: Contains the force constants obtained from DFT calculations.

With those files, neutron weighted band structure plots can be generated in the following way:

```
from phonopy_tools import PhonopyNeutron
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ph_data = PhonopyNeutron('phonopy_disp.yaml', 'FORCE_SETS ')
ph_data.set_path([[0,0,0] ,[0.5,0.5,0] ,[1,0,0] ,[0,0,0]])
ph_data.set_labels (['$\Gamma$ ', 'X', 'M', '$\Gamma$'])
ph_data.compute_bands()
ph_data.compute_neutron_bands()
ph_data.plot_neutron_bands (ax , plotype='lines', sigma=0.2)
```
Line 1 imports the module, and line 4 imports the Phonopy data. In order to get the band structure, it is necessary to set the path in reciprocal space that you want to plot as shown in line 5. The coordinates are here with respect to the input cell, so not necessarily the primitive cell. Line 5 simply labels these paths, line 6 and 7 computes the bands and line 8 plots them.

A different option is to plot in 2 dimensions of Q at a selected energy. An example of how to get this plot is:

```
from phonopy_tools import PhonopyNeutron, get_xy_colormap
import matplotlib.pyplot as plt
ph_data = PhonopyNeutron('phonopy_disp.yaml', 'FORCE_SETS')
cmap_data = lco.get_sqw_xy ([1, 5, -1, 3], 100, 100)
x, y, I = get_xy_colormap(cmap_data, 9, sigma=1)
plt.pcolor (x, y, I)
```

We load the modules in line 1, and the Phonopy data in line 3. In line 4, we generate the S(Q, omega) in a grid where Qx ranges from 1 to 5 and Qy ranges from -1 to 3 with a grid size of 100 in each direction. The evaluation of this can be quite slow, so start with a small grid size. Finally, the `get_xy_colormap()` function uses this data to generate a colormap at a certain energy (here 9 meV) with some fixed Gaussian resolution `sigma` (in meV).




