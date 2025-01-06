# Define the cross section of a coil given the forces and current
import numpy as np
import xyz2fourier
from simsopt.field.force import coil_force, coil_torque
from simsopt.field.selffield import regularization_circ

# --- INPUTS ---
filecoils = '/home/jhidalgo/stellarator_system_code/inputs/coils.example'
material = 'HTS' # placeholder
sigma_cond = 50e6 # Pa, yield limit of pure copper, from ITER MPH (used for the entire conductor, SC and Cu)
sigma_steel = 150e6 # Pa, yield limit of SS316L, from ITER MPH

# --- SCRIPT ---
coils, curves = xyz2fourier.file2coils(filecoils)
current = np.array([c.current.current for c in coils])

j_sc = 500e6 # A/m2
j_cu = 120e6 # A/m2 
A_sc =  np.abs(current)/j_sc
A_cu = np.abs(current)/j_cu
A_cond = A_sc+A_cu
A_coolant = A_cond*0.3/(1-0.3) # 30% of the cable is for coolant
A_cable = A_cond+A_coolant

# Get the forces acting on the coils
forces = np.zeros([curves[0].gamma().shape[0], 3, len(coils)])
torques = np.zeros_like(forces)
a = 0.05 # regularization term
for i in range(len(coils)):
    # Forces and torques are given in N/m and Nm/m, respectively. The torque is converted back to Nm
    gamma = curves[i].gamma()
    gamma = np.append(gamma, np.reshape(gamma[0,:], (1,3)), axis = 0)
    dgamma = np.linalg.norm(gamma[1:,:]-gamma[0:-1,:], axis = 1)
    dgamma = np.reshape(dgamma, (len(dgamma),1))
    forces[:, :, i] = coil_force(coils[i], coils, regularization_circ(a))
    torques[:, :, i] = dgamma*coil_torque(coils[i], coils, regularization_circ(a))
# Get the parallel and perpendicular components of the forces to the coil. Forces are given in N/m
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
torques_par_norm = np.linalg.norm(torques_par, axis = 1)
torques_perp_norm = np.linalg.norm(torques_perp, axis = 1)

# Get the minimum steel case
# Initialize the values so the loop starts working
t_steel = np.zeros((len(coils)))
A_coil = np.zeros((len(coils)))
sigma_coil = np.ones_like(t_steel)*sigma_cond
sigma_vm_max = sigma_coil*2
for i in range(len(coils)):
    while sigma_coil[i] < sigma_vm_max[i]:
        t_steel[i] += 0.01
        A_coil[i] = (np.sqrt(A_cable[i])+2*t_steel[i])
        A_steel = A_coil[i]-A_cable[i]
        f_steel = A_steel/A_coil[i]
        f_cond = A_cond[i]/A_coil[i]
        sigma_coil[i] = f_steel*sigma_steel+f_cond*sigma_cond
        c = curves[i].gamma()
        sigma = np.zeros(len(c))
        tau = np.zeros(len(c))
        l = np.sqrt(A_cable[i])+t_steel[i]
        for j in range(len(c)):
            sigma[j] = forces_perp_norm[j,i]/l
            J = l**4/6
            tau[j] = torques_par_norm[j,i]*l/J
        # s_vm = np.sqrt((sigma**2+6*tau**2)/2)
        s_vm = sigma
        sigma_vm_max[i] = s_vm.max()
