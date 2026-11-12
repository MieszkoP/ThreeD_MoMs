import numpy as np
import ThreeD_MoMs

def mesh_creator2D(xmin, xmax, zmin, zmax, y_value):
    def create_mesh():
        N=200
        x = np.linspace(xmin, xmax, N)
        y = np.linspace(y_value, y_value+0.1, 1)#-0.2, 0.2, -0.2, 0.2
        z = np.linspace(zmin, zmax, N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        r = np.column_stack((
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ))
        return r
    return create_mesh

def mesh_creator2Dyz(ymin, ymax, zmin, zmax, x_value):
    def create_mesh():
        N=200
        z = np.linspace(zmin, zmax, N)
        x = np.linspace(x_value, x_value+0.1, 1)#-0.2, 0.2, -0.2, 0.2
        y = np.linspace(ymin, ymax, N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        r = np.column_stack((
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ))
        return r
    return create_mesh
    
def mesh_creator2Dxy(xmin, xmax, ymin, ymax, z_value):
    def create_mesh():
        N=200
        x = np.linspace(xmin, xmax, N)
        z = np.linspace(z_value, z_value+0.1, 1)#-0.2, 0.2, -0.2, 0.2
        y = np.linspace(ymin, ymax, N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        r = np.column_stack((
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ))
        return r
    return create_mesh

def mesh_creator3D(xmin, xmax, zmin, zmax, ymin, ymax):
    def create_mesh():
        N=40
        x = np.linspace(xmin, xmax, N)
        y = np.linspace(ymin, ymax, N)
        z = np.linspace(zmin, zmax, N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        r = np.column_stack((
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ))
        return r
    return create_mesh

def V_m_by_theta(vertices, faces, fn, fn_pm, freq, theta, pol): 
    '''
    polarization is either E_x   (pol = 'y') or parallel to zy plane (pol = 'xz')
    '''
    import ThreeD_MoMs
    if(pol == "y"):
        if(theta == 180):
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Pex") # polaryzacja -y (x times -z = y)
            V_E, V_nH = -V_E, -V_nH
        elif(theta == 360):
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Pex") # polaryzacja -y (x times z = -y)
        else:
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Pez") # polaryzacja -y (z times [sint 0 cost] -> -y)
    elif(pol=="xz"):
        if(theta == 180):
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Pax") # polaryzacja xz (-z times y = x)
            V_E, V_nH = -V_E, -V_nH
        elif(theta == 360):
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Pax") # polaryzacja xz (z times -y = x)
        else:
            theta = np.deg2rad(theta)
            V_E, V_nH = ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm,  np.array([np.sin(theta),0,np.cos(theta)]), freq, "Paz") # polaryzacja xz ([sint 0 cost] times -y = -x dla 180 i +x dla 360)
            #Aby uzgodnić niech będzie -x dla 180, +x dla 360
    else:
        print("wrong polarization")
        
    return V_E, V_nH

def E_i_by_theta(theta, pol, r, freq):
    import MoM
    k, E_i_vers = MoM.V_i_by_theta(theta, pol)
    k = k.reshape(-1,1)
    E_i = np.exp(-2j*np.pi*freq/299792458.0*(r@k).T[0]).reshape(-1,1)*E_i_vers
    return E_i
    
def V_i_by_theta(theta, pol): 
    '''
    polarization is either E_x   (pol = 'y') or parallel to zy plane (pol = 'xz')
    '''
    theta_rad = np.deg2rad(theta)
    k = np.array([np.sin(theta_rad),0,np.cos(theta_rad)])
    if(pol == "y"):
        if(theta == 180):
            E_i_vers = np.array([0,-1,0])
        elif(theta == 360):
            E_i_vers = np.array([0,-1,0])
        else:
            E_i_vers = np.cross(np.array([0, 0, 1]), k) / np.linalg.norm(np.cross(np.array([0, 0, 1]), k))
    elif(pol=="xz"):
        if(theta == 180):
            E_i_vers = np.array([-1,0,0])
        elif(theta == 360):
            E_i_vers = np.array([1,0,0])
        else:
            theta = np.deg2rad(theta)
            E_i_vers = np.cross(k, np.cross(np.array([0, 0, 1]), k)) / np.linalg.norm(np.cross(k, np.cross(np.array([0, 0, 1]), k)))
            #Aby uzgodnić niech będzie -x dla 180, +x dla 360
    else:
        print("wrong polarization")
        
    return k, E_i_vers

def E_i_by_k(k, pol, r, freq):
    if(pol == 'x'):
        e = np.cross(np.array([1.0,0.0,0.0]), k) #polaryzacja Pex lub Pez
    else:
        e = np.cross(np.array([0.0,0.0,1.0]), k)
            
    f = np.cross(k, e)
    e = e/np.linalg.norm(e)
    f = f/np.linalg.norm(f)
    k = k.reshape(-1,1)
        
    E_i_e = np.exp(-2j*np.pi*freq/299792458.0*(r@k).T[0]).reshape(-1,1)*e #polaryzacja Pex lub Pez
    E_i_f = np.exp(-2j*np.pi*freq/299792458.0*(r@k).T[0]).reshape(-1,1)*f #Polaryzacja Pax lub Paz
    return E_i_e, E_i_f

def V_m_by_k_creator(k, pol):
    import ThreeD_MoMs
    def V_m_fun(vertices, faces, fn, fn_pm, freq):
        return ThreeD_MoMs.V_m(vertices, faces, fn, fn_pm, k, freq, pol)
    return V_m_fun

def V_m_by_theta_creator(theta : int, pol : str):
    def V_m_fun(vertices, faces, fn, fn_pm, freq):
        return V_m_by_theta(vertices, faces, fn, fn_pm, freq, theta, pol)
    return V_m_fun

def MoM_exc(vertices, faces, fn, fn_pm, V_m_fun, freq, create_mesh, Z_CFIE_inv):
    epsilon = 8.8541878e-12;#// [F/m]
    mu = 1.2566371e-6; #[H/m]
    from scipy.sparse.linalg import LinearOperator, gmres
        
    V_E, V_nH = V_m_fun(vertices, faces, fn, fn_pm, freq)

    V_E = V_E[1:]
    V_nH = V_nH[1:]
    alpha = 0.5
    V_CFIE = (1-alpha)*V_E+alpha*np.sqrt(mu/epsilon)*V_nH

    I_E_CFIE = (Z_CFIE_inv@V_CFIE).reshape(-1,1)    
    I_E_CFIE = np.vstack((np.zeros((1, I_E_CFIE.shape[1])), I_E_CFIE))
    
    r = create_mesh()

    E_s, H_s = ThreeD_MoMs.E_s(r, I_E_CFIE, vertices, faces, fn, fn_pm, freq)
    return E_s, H_s