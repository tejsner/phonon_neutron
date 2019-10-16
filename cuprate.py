# cuprate.py
from ase import spacegroup as sg
from ase.io import write
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import write_vasp

def tand(x):
    return np.tan(x * np.pi / 180)

def sind(x):
    return np.sin(x * np.pi / 180)

def cosd(x):
    return np.cos(x * np.pi / 180)

def ase_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions()
    )

class Cuprate:
    def __init__(self, a=3.78, c=13.007, eta=0, oap=0.184, laz=0.362, Q1=0, Q2=0, lax=0, lay=0, Q1_ap=None, Q2_ap=None):
        b = (1+eta)*a*np.sqrt(2)
        a = 1/(1+eta)*a*np.sqrt(2)

        if Q1_ap is None:
            dx = oap*sind(Q1)*c/a
        else:
            dx = oap*sind(Q1_ap)*c/a

        if Q2_ap is None:
            dy = oap*sind(Q2)*c/b
        else:
            dy = oap*sind(Q2_ap)*c/b
            
        
        dz_o1 = - 0.25*sind(Q1)*a/c + 0.25*sind(Q2)*b/c
        dz_o2 = - 0.25*sind(Q1)*a/c - 0.25*sind(Q2)*b/c

        symbols = ['Cu', 'La', 'O', 'O', 'O']
        cell = [a, b, c, 90, 90, 90]
        Cu = [0,0,0]
        La = [lax,lay,laz]
        O1 = [0.25, 0.75, dz_o1]
        O2 = [0.25, 0.25, dz_o2]
        O3 = [0-dx, 0.5+dy, 0.5-oap]
        basis = [Cu, La, O1, O2, O3]

        self.atoms = sg.crystal(symbols=symbols, basis=basis, cellpar=cell, spacegroup=56)
        self.atoms = ase_to_phonopy(self.atoms)

    def print_spacegroup(self):
        g = sg.get_spacegroup(self.atoms)
        print('Space group', g.no, g.symbol)

    def get_volume(self):
        return self.atoms.cell[0,0]*self.atoms.cell[1,1]*self.atoms.cell[2,2]

    def get_eta(self):
        a = self.atoms.cell[0,0]
        b = self.atoms.cell[1,1]
        return (b-a)/(b+a)

    def expand_volume(self, val, method='p'):
        # methods can be any string for percent or 'abs' for absolute
        if method == 'abs':
            val = val/self.get_volume()

        for i in range(3):
            self.atoms.cell[i,i] *= val**(1/3)

    def write(self, filename):
        # the file io from ASE does not label the atoms for display in VESTA
        # so use the phonopy version
        write_vasp(filename, self.atoms)
