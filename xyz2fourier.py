# Routines to prepare a coil to the simsopt format
import numpy as np
import matplotlib.pyplot as plt
from simsopt._core.optimizable import DOFs
from simsopt.geo import CurveXYZFourier
from simsopt.field import Coil, Current
import fourier_decomp


def xyz2curve(x,y,z, qpoints = 1000):
    # Returns a cuve object on the simsopt format
    phi = np.linspace(0,2*np.pi, len(x))

    xf = fourier_decomp.nft1d(phi,x)
    xf = xf.T.flatten()
    xf = np.delete(xf, 0)
    yf = fourier_decomp.nft1d(phi,y)
    yf = yf.T.flatten()
    yf = np.delete(yf, 0)
    zf = fourier_decomp.nft1d(phi,z)
    zf = zf.T.flatten()
    zf = np.delete(zf, 0)
    fcoefs = np.concatenate((xf, yf, zf))

    order = int((len(fcoefs)-3)/6)
    dofs = DOFs(fcoefs)
    return CurveXYZFourier(qpoints, order, dofs)

def file2coils(filename, qpoints = 1000):
    # Returns the coils and their associated curves (same but with no currents)
    with open(filename, "r") as file:
        data = file.readlines()[3:]
    curves = []
    coils = []
    coords = []
    for line in data:
        columns = line.strip().split()
        if columns[0] == "end":
            break
        # Coil current
        s = float(columns[3])

        if s != 0: # s = 0 signals end of filament
            coords.append([float(ord) for ord in columns[0:3]])
            curr = s
        else:
            coords = np.array(coords)
            curves.append(xyz2curve(coords[:,0], coords[:,1], coords[:,2], qpoints = qpoints))
            coils.append(Coil(curves[-1], Current(curr)))
            coords = []
    return coils, curves

def plotcurves(curves, ax = None):
    # Plot all curves in an assembly. MatPlotLib does not handle well the 3D plots, so no axes equal.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
    xmax, xmin, ymax, ymin, zmax, zmin = [0,0,0,0,0,0]
    for c in curves:
        c.plot(ax= ax, show = False)
        val = c.gamma()
        if max(val[:,0]) > xmax: xmax = max(val[:,0])
        if min(val[:,0]) < xmin: xmin = min(val[:,0])
        if max(val[:,1]) > ymax: ymax = max(val[:,1])
        if min(val[:,1]) < ymin: ymin = min(val[:,1])
        if max(val[:,2]) > zmax: zmax = max(val[:,2])
        if min(val[:,2]) < zmin: zmin = min(val[:,2])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    plt.show(block = False)
    
