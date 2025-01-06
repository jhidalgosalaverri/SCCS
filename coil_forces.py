"""
Get the Lorentz forces on the given coils. 

Based on the coil_forces example of the simsopt library.
"""
import numpy as np
from simsopt.field.force import coil_force, coil_torque
from simsopt.field.selffield import regularization_circ
import xyz2fourier
import matplotlib.pyplot as plt

# --- INPUTS ---
filename = '/home/jhidalgo/stellarator_system_code/inputs/coils.example'
coils, curves = xyz2fourier.file2coils(filename, qpoints = 1000)

# --- SCRIPT ---
forces = np.zeros([curves[0].gamma().shape[0], 3, len(coils)])
torques = np.zeros_like(forces)
a = 0.05 # regularization term
for i in range(len(coils)):
    forces[:, :, i] = coil_force(coils[i], coils, regularization_circ(a))
    torques[:, :, i] = coil_torque(coils[i], coils, regularization_circ(a))

forces_norm = np.linalg.norm(forces, axis = 1).flatten()
torques_norm = np.linalg.norm(torques, axis = 1).flatten()
point_data = {"Pointwise_Forces": forces_norm, "Pointwise_Torques": torques_norm}

# Get the parallel and perpendicular components of the forces to the coil
forces_par = np.zeros_like(forces)
forces_perp = np.zeros_like(forces)
torques_par = np.zeros_like(forces)
torques_perp = np.zeros_like(forces)
for i in range(len(curves)):
    c = curves[i].gamma()
    for j in range(len(c)):
        dir_curve = c[0,:]-c[j,:] if j == len(c)-1 else c[j+1,:]-c[j,:]
        f = forces[j,:,i]
        t = torques[j,:,i]
        forces_par[j,:,i] = np.dot(f, dir_curve)/np.dot(dir_curve, dir_curve)*dir_curve
        forces_perp[j,:,i] = f-forces_par[j,:,i]
        torques_par[j,:,i] = np.dot(t, dir_curve)/np.dot(dir_curve, dir_curve)*dir_curve
        torques_perp[j,:,i] = t-torques_par[j,:,i]
forces_par_norm = np.linalg.norm(forces_par, axis = 1)
forces_perp_norm = np.linalg.norm(forces_perp, axis = 1)

# fig, ax = plt.subplots()
# ax.plot(forces_par_norm.flatten(), 'o', label ='Fpar')
# ax.plot(forces_perp_norm.flatten(), 'o', label ='Fperp')
# ax.legend()
# fig.tight_layout()
# plt.show(block = False)

dh = np.linalg.norm(curves[0].gamma()[1:, :]-curves[0].gamma()[0:-1, :], axis = 1)
f = np.linalg.norm(forces[:,:,0], axis = 1)
f = (f[1:]+f[0:-1])/2
t = np.linalg.norm(torques[:,:,0], axis = 1)
t = (t[1:]+t[0:-1])/2

# print(np.sum(f*dh))
print(np.sum(t*dh))
# print(np.sum(f))
# fig, ax = plt.subplots()
# ax.plot(f)
# plt.show(block = False)