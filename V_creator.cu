#include "creator.cuh"
#include "cuda_runtime.h"
#include "memory_man.cuh"
#include "custom_operators.cuh"

__device__ __host__  complex V_D(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq, point k, polType P)
{
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result;
    for(uint p = 0; p<N_points_quad; p++)
    {   
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        point E_vers;
        point vers;
        if( P == polType::Pax || P == polType::Pex)
            vers = {1.0, 0.0, 0.0};
        if( P == polType::Pay || P == polType::Pey)
            vers = {0.0, 1.0, 0.0};
        if( P == polType::Paz || P == polType::Pez)
            vers = {0.0, 0.0, 1.0};

        if(P == polType::Pax || P  == polType::Pay || P == polType::Paz)
        {
            point surf_nrm = vec_vec_m(vers, k);
            E_vers = vec_vec_m(k, surf_nrm);
        }
        else
        {
            E_vers = vec_vec_m(vers, k);
        }
        E_vers = vec_times_sc(E_vers, 1.0/vec_len(E_vers));
        sc_point E = vec_times_sc((sc_point)E_vers, 1.0*exp(complex(0, -2*PI*freq*vec_sc_m(k, rp)/c_v)));
        result += seven_points_Q[p][3]*vec_sc_m((sc_point)rho_m_arr,E);
    }
    return result*fm.L/2.0;
}

__device__ __host__ complex V_nH_D(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq, point k, polType P)
{
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result;
    for(uint p = 0; p<N_points_quad; p++)
    {   
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};

        point H_vers;
        point vers;
        if( P == polType::Pax || P == polType::Pex)
            vers = {1.0, 0.0, 0.0};
        if( P == polType::Pay || P == polType::Pey)
            vers = {0.0, 1.0, 0.0};
        if( P == polType::Paz || P == polType::Pez)
            vers = {0.0, 0.0, 1.0};

        if(P == polType::Pax || P  == polType::Pay || P == polType::Paz)
        {
            point surf_nrm = vec_vec_m(vers, k);
            point E_vers = vec_vec_m(k, surf_nrm);
            H_vers = vec_vec_m(k, E_vers);
        }
        else
        {
            point E_vers = vec_vec_m(vers, k);
            H_vers = vec_vec_m(k, E_vers);
        }
        H_vers = vec_times_sc(H_vers, 1.0/vec_len(H_vers));

        sc_point H = vec_times_sc((sc_point)H_vers, 1.0*sqrt(epsilon/mu)*exp(complex(0, -2*PI*freq*vec_sc_m(k, rp)/c_v)));
        sc_point nH = vec_vec_m((sc_point)fm.normal, H);
        result += seven_points_Q[p][3]*vec_sc_m((sc_point)rho_m_arr,nH);
    }
    return result*fm.L/2.0;
}

py::tuple V_m(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, numpy_array_real k, double freq, const std::string& pol)
{
    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();
    auto buf_k = k.request();

    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);
    const double* _k = static_cast<double*>(buf_k.ptr);

    uint N = (uint)buf_edges.shape[0]-1;
    complex* V_E = (complex*)calloc((N+1), sizeof(complex));
    complex* V_nH = (complex*)calloc((N+1), sizeof(complex));

    point _kp = {_k[0],_k[1],_k[2]};
    polType polarization;

    if(pol == "Paz")
    {
        polarization = polType::Paz;
    }
    else if(pol == "Pez")
    {
        polarization = polType::Pez;
    }
    else if(pol == "Pax")
    {
        polarization = polType::Pax;
    }
    else if(pol == "Pex")
    {
        polarization = polType::Pex;
    }
    else if(pol == "Pey")
    {
        polarization = polType::Pey;
    }
    else
    {
        polarization = polType::Pay;
    }

    for(uint m = 1; m<=N; m++)
    {   
        V_E[m] = {0.0};
        V_nH[m] = {0.0};
        uint fm_m = index(_edges, m, 1, 4);
        uint fm_p = index(_edges, m, 0, 4);
        V_E[m] = V_D(fm_m, _verticies, _faces, _face_curr, freq,_kp,polarization ) + V_D(fm_p, _verticies, _faces, _face_curr, freq, _kp,polarization );
        V_nH[m] = V_nH_D(fm_m, _verticies, _faces, _face_curr, freq, _kp,polarization ) + V_nH_D(fm_p, _verticies, _faces, _face_curr, freq, _kp,polarization );
    }

    numpy_array V_E_numpy({N+1});
    numpy_array V_nH_numpy({N+1});

    std::memcpy(
    V_E_numpy.mutable_data(),
    V_E,
    (N+1) * sizeof(complex)
    );

    std::memcpy(
        V_nH_numpy.mutable_data(),
        V_nH,
        (N+1) * sizeof(complex)
    );

    free(V_E);
    free(V_nH);
    return py::make_tuple(V_E_numpy,V_nH_numpy);
}

py::tuple V_m_cuda(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    //Not yet done
    return py::make_tuple();
}