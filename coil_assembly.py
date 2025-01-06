import numpy as np
import xyz2fourier
from simsopt.field.force import coil_force, coil_torque
from simsopt.field.selffield import regularization_circ

# --- References ---
# [1] X. Sarasola et al. http://idm.euro-fusion.org/?uid=2PG9KL
# [2] K. Tsuchiya et al., Cryogenics, 2017. https://doi.org/10.1016/j.cryogenics.2017.05.002
# [3] S.B.L. Chislett-McDonald et al., arXiv, 2022. https://doi.org/10.48550/arXiv.2205.04441 
# [4] ITER Material Properties Handbook. https://user.iter.org/default.aspx?uid=24L6RN 
# [5] ITER Material Properties Handbook. https://user.iter.org/default.aspx?uid=222TFS

# --- Units ---
# j in W/m2
# sigma in Pa
# cost in $/A*m

materials = {'NbTi': {'j': 1030e6, # [1] @7T and @4.75K
                      'sigma': 0, 
                      'cost': 1.7e-6}, # [3]
             'Nb3Sn': {'j': 2329e6, # [1] @7T and @4.75K
                      'sigma': 0, 
                      'cost': 1.7e-6}, # [3]
             'REBCO1': {'j': 1615e6, # [2] @7T
                      'sigma': 0, 
                      'cost': 80e-6}, # [3]
             'REBCO2': {'j': 1615e6, # [2] @7T
                      'sigma': 0, 
                      'cost': 30e-6}, # [3]
             'REBCO3': {'j': 1615e6, # [2] @7T
                      'sigma': 0, 
                      'cost': 10e-6}} # [3]

class c_assembly():

    def __init__(self, coils, curves, material, sigma_cu = 50e6, sigma_steel = 150e6,
                 j_cu = 120e6, sf_sigma = 1.2, sf_j = 1.2):
        # sigma_cu -> [4]
        # sigma_steel -> [5]
        # j_cu -> [1]
        
        if type(material) is str: mat = materials[material]
        else: mat = material

        c_assembly.coils = coils
        c_assembly.curves = curves
        c_assembly.sigma_cu_crit = sigma_cu
        c_assembly.sigma_cu_max = sigma_cu/sf_sigma
        c_assembly.sigma_sc_crit = mat['sigma']
        c_assembly.sigma_sc_max = mat['sigma']/sf_sigma
        c_assembly.sigma_steel_crit = sigma_steel
        c_assembly.sigma_steel_max = sigma_steel/sf_sigma
        c_assembly.j_sc_crit = mat['j']
        c_assembly.j_sc_max = mat['j']/sf_j
        c_assembly.j_cu_crit = j_cu
        c_assembly.j_cu_max = j_cu/sf_j
        c_assembly.sc_mat = mat

    def get_cross_section(self):
        curves = c_assembly.curves
        coils = c_assembly.coils
        # Get the cable (conductor + coolant) section
        current = np.array([c.current.current for c in c_assembly.coils])
        A_sc =  np.abs(current)/c_assembly.j_sc_max
        A_cu = np.abs(current)/c_assembly.j_cu_max
        A_cond = A_sc+A_cu
        A_coolant = A_cond*0.3/(1-0.3) # 30% of the cable is for coolant -> [1]
        A_cable = A_cond+A_coolant
        sigma_cond = (A_sc*c_assembly.sigma_sc_max+A_cu*c_assembly.sigma_cu_max)/A_cond
        sigma_cable = (A_sc*c_assembly.sigma_sc_max+A_cu*c_assembly.sigma_cu_max)/A_cable

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
        sigma_coil = np.ones_like(t_steel)*(c_assembly.sigma_cu_max*A_cu+c_assembly.sigma_sc_max*A_sc)/A_cable
        sigma_vm_max = sigma_coil*2 # this is just to start the iteration
        for i in range(len(coils)):
            while sigma_coil[i] < sigma_vm_max[i]:
                t_steel[i] += 0.01
                A_coil[i] = (np.sqrt(A_cable[i])+2*t_steel[i])
                A_steel = A_coil[i]-A_cable[i]
                f_steel = A_steel/A_coil[i]
                f_cond = A_cond[i]/A_coil[i]
                sigma_coil[i] = f_steel*c_assembly.sigma_steel_max+f_cond*sigma_cable[i]
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
        
        A_steel = A_coil-A_cable
        c_assembly.section = {'A_coil': A_coil,
                              'A_cu': A_cu,
                              'A_sc': A_sc,
                              'A_cable': A_cable,
                              'A_steel': A_steel, 
                              't_steel': t_steel}

    def get_cost(self, c_steel = 4.2, rho_steel = 7850, c_cu = 9.259, rho_cu = 8960): 
        # c_steel and c_cu in $/kg
        # rho_steel and rho_cu in kg/m3
        curves = c_assembly.curves
        current = np.array([c.current.current for c in c_assembly.coils])
        c_sc = c_assembly.sc_mat['cost']
        tcost_steel = 0
        tcost_cu = 0
        tcost_sc = 0

        for i, c in enumerate(curves):
            g = c.gamma()
            l = np.sum(np.sqrt((g[0:-1, 0]-g[1:, 0])**2+(g[0:-1, 1]-g[1:, 1])**2+(g[0:-1, 2]-g[1:, 2])**2))
            tcost_steel += c_steel*l*c_assembly.section['A_steel'][i]*rho_steel
            tcost_cu += c_cu*l*c_assembly.section['A_cu'][i]*rho_cu
            tcost_sc += c_sc*np.abs(current[i])*l

        c_assembly.cost ={'steel': tcost_steel,
                          'sc': tcost_sc,
                          'cu': tcost_cu,
                          'total': tcost_steel+tcost_sc+tcost_cu}




