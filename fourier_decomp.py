import numpy as np
import numba as nb

@nb.njit(cache=True, nogil=True)
def my_trapz(y: float, dx: float=1.0):
     """
     Internal trapezoidal rule to be used alongside with numba.

     Pablo Oyola - poyola@us.es

     :param y: values evenly spaced to integrate.
     :param dx: spacing between the points.
     """

     return dx/2.0*(y[..., 0] + y[..., -1] + 2.0*y[..., 1:-1].sum(axis=-1))


@nb.njit
def nft1d(phi: float, val: float, nmax: int=-1):
     """
     Decompose a field into the respective toroidal mode numbers (or equivalent).

     Pablo Oyola - poyola@us.es

     :param phi: axis along the phi direction. It must be evenly distributed.
     :param val: value to decompose. The phi direction must be the last one.
     :param nmax: maximum toroidal mode number. It will be chosen to be the
         minimum between the input value and the Nyquist frequency.
     """
     dphi = phi[1] - phi[0]
     phi_range = phi[-1] - phi[0]
     nphi = len(phi)

     phimode = float(int(2.0*np.pi/phi_range))

     ## Computing the Nyquist frequency.
     nNyquist = nb.int64(np.floor(nphi/(2.0*phi_range)))
     if nmax < 0:
         nmax = nNyquist+1
     else:
         nmax = nb.int64(min(nNyquist+1, nmax+1))

     ## Allocating output.
     output = np.zeros((2, nmax))
     ## Looping along all the dimensions.
     for ix in nb.prange(val.size):
         for n in range(nmax):
             output[0, n] = my_trapz(val*np.sin(n*phimode*phi), dx=dphi)/np.pi
             output[1, n] = my_trapz(val*np.cos(n*phimode*phi), dx=dphi)/np.pi

     output[0, 0] = 0.0
     output[1, 0] /= 2.0

     output *= phimode

     return output

@nb.njit(nogil=True, parallel=True, cache=True)
def rebuild_nft(phi: float, val_n: float):
     """
     Recomposes a field from the respective toroidal mode numbers (or equivalent).

     Pablo Oyola - poyola@us.es

     :param phi: axis along the phi direction. If must be evenly distributed.
     :param val_n: value to decompose. The phi direction must be the last one.
     """

     phi_range = phi[-1] - phi[0]
     phimode = float(int(2.0*np.pi/phi_range))

     # We now invert the decomposition.
     output = np.zeros((val_n.shape[1], val_n.shape[2], phi.size))
     for ix in nb.prange(val_n.shape[1]):
         for iy in range(val_n.shape[2]):
             for n in range(val_n.shape[-1]):
                 output[ix, iy, :] += val_n[0, ix, iy, n] * np.sin(n*phimode*phi) + \
                                      val_n[1, ix, iy, n] * np.cos(n*phimode*phi)

     return output