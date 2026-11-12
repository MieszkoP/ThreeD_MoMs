#include "creator.cuh"
#include "cuda_runtime.h"
#include "custom_operators.cuh"
#include "memory_man.cuh"

__device__ __host__ sc_point IL_DI(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr , double freq, point _point)
{
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    sc_point result = {0.0, 0.0, 0.0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        //point rho_m_arr;
        //Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        double w_p = seven_points_Q[p][3];
        double R = vec_len(vector_sum(_point, vec_times_sc(rp, -1.0)));
        complex nominator = exp(complex(0,-k_f(freq)*R))*(complex(1.0,k_f(freq)*R));

        sc_point under_sum = vec_times_sc((sc_point)vector_sum(_point, vec_times_sc(rp, -1.0)), w_p*nominator/cube(R));

        result = vector_sum(result, under_sum);

    }
    if(fm.isplus)
        return vec_times_sc(result, (complex)fm.L/(4.0*PI));
    else
        return vec_times_sc(result, (complex)fm.L/(-4.0*PI));
}

__device__ __host__ sc_point InK_D(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr , double freq, point _point)
{
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    sc_point result = {0.0, 0.0, 0.0};
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    for(uint p = 0; p<N_points_quad; p++)
    {
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);

        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        double w_p = seven_points_Q[p][3];
        double R = vec_len(vector_sum(_point, vec_times_sc(rp, -1.0)));
        complex nominator = exp(complex(0,-k_f(freq)*R))*(complex(1.0,k_f(freq)*R));

        sc_point under_sum_left = vec_times_sc((sc_point)vector_sum(_point, vec_times_sc(rp, -1.0)), w_p*nominator/cube(R));
        sc_point under_sum = vec_vec_m(under_sum_left, (sc_point)rho_m_arr);

        result = vector_sum(result, under_sum);

    }

    return vec_times_sc(result, (complex)fm.L/(-8.0*PI));
}

__device__ __host__ sc_point IL_DII(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr , double freq, point _point)
{
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    sc_point result = {0.0, 0.0, 0.0};
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    for(uint p = 0; p<N_points_quad; p++)
    {
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        double w_p = seven_points_Q[p][3];
        double R = vec_len(vector_sum(_point, vec_times_sc(rp, -1.0)));
        complex nominator = exp(complex(0,-k_f(freq)*R));

        sc_point under_sum = vec_times_sc((sc_point)rho_m_arr, w_p*nominator/R);
        result = vector_sum(result, under_sum);

    }
    return vec_times_sc(result, (complex)fm.L/(8.0*PI));
}

__device__ __host__ sc_point IL_D(uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr , double freq, point _point)
{
    sc_point IL_DI_el = vec_times_sc(IL_DI(fm_pm, _verticies, _faces, _face_curr, freq, _point), (complex)1/sq(k_f(freq)));
    return vector_sum(IL_DI_el, IL_DII(fm_pm, _verticies, _faces, _face_curr, freq, _point));
}

void __global__ E_s_kernel(int M, int N, complex* Es_r, complex* Hs_r, const complex* _I_E ,const uint* _edges, const double* _verticies, const double* _faces,const uint*  _face_curr, double freq, const double* _r)
{
    sc_point IL_D_sum = {0.0,0.0,0.0};
    sc_point nK_D_sum = {0.0,0.0,0.0};
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=N)
    {
        return;
    }
    point _point = {index(_r, i, 0,3), index(_r, i, 1,3), index(_r, i, 2,3)};

        //py::print("po odczytaniu punktu i = ", i,  py::arg("flush") = true);
    for(uint j =1; j<=M; j++)
    {
        uint fm_m = index(_edges, j, 1, 4);
        //py::print("87");
        sc_point IL_D_j = vec_times_sc(IL_D(fm_m, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
        sc_point InK_D_j = vec_times_sc(InK_D(fm_m, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
        //py::print("89");
        IL_D_sum = vector_sum(IL_D_sum, IL_D_j);
        nK_D_sum = vector_sum(nK_D_sum, InK_D_j);
        //py::print("91");

        uint fm_p = index(_edges, j, 0, 4);
        //py::print("94");
        IL_D_j = vec_times_sc(IL_D(fm_p, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
        InK_D_j = vec_times_sc(InK_D(fm_p, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
        //py::print("96");
        IL_D_sum = vector_sum(IL_D_sum, IL_D_j);
        nK_D_sum = vector_sum(nK_D_sum, InK_D_j);
        //py::print("98");
    }
    sc_point Es_r_point = vec_times_sc(IL_D_sum, complex(0, -mu*omega(freq)));

    sc_point Hs_r_point = nK_D_sum;
    //py::print("przed index(Es_r, i, 0, 3) = Es_r_point.x; i = ", i );
    index(Es_r, i, 0, 3) = Es_r_point.x;
    index(Es_r, i, 1, 3) = Es_r_point.y;
    index(Es_r, i, 2, 3) = Es_r_point.z;

    index(Hs_r, i, 0, 3) = Hs_r_point.x;
    index(Hs_r, i, 1, 3) = Hs_r_point.y;
    index(Hs_r, i, 2, 3) = Hs_r_point.z;
}

numpy_array E_s(numpy_array_real r, numpy_array I_E, numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    py::print("start!",  py::arg("flush") = true);
    auto buf_I_E = I_E.request(); 
    auto buf_r = r.request(); //r to tablica o rozmiarach [liczba_punktów, 3]

    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();

    const complex* _I_E= static_cast<complex*>(buf_I_E.ptr);
    const double* _r = static_cast<double*>(buf_r.ptr);
    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);
    

    uint M = (uint)buf_edges.shape[0]-1; //number of currents
    uint N = (uint)buf_r.shape[0]; //number of points

    complex* Es_r = (complex*)calloc(3*(N), sizeof(complex));
    complex* Hs_r = (complex*)calloc(3*(N), sizeof(complex));

    for(uint i = 0; i<N; i++)
    {   
        sc_point IL_D_sum = {0.0,0.0,0.0};
        sc_point nK_D_sum = {0.0,0.0,0.0};

        point _point = {index(_r, i, 0,3), index(_r, i, 1,3), index(_r, i, 2,3)};

        //py::print("po odczytaniu punktu i = ", i,  py::arg("flush") = true);
        for(uint j =1; j<=M; j++)
        {

            uint fm_m = index(_edges, j, 1, 4);
            //py::print("87");
            sc_point IL_D_j = vec_times_sc(IL_D(fm_m, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
            sc_point InK_D_j = vec_times_sc(InK_D(fm_m, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
            //py::print("89");
            IL_D_sum = vector_sum(IL_D_sum, IL_D_j);
            nK_D_sum = vector_sum(nK_D_sum, InK_D_j);
            //py::print("91");

            uint fm_p = index(_edges, j, 0, 4);
            //py::print("94");
            IL_D_j = vec_times_sc(IL_D(fm_p, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
            InK_D_j = vec_times_sc(InK_D(fm_p, _verticies, _faces, _face_curr, freq, _point), index(_I_E, j,0,1));
            //py::print("96");
            IL_D_sum = vector_sum(IL_D_sum, IL_D_j);
            nK_D_sum = vector_sum(nK_D_sum, InK_D_j);
            //py::print("98");
        }
        sc_point Es_r_point = vec_times_sc(IL_D_sum, complex(0, -mu*omega(freq)));

        sc_point Hs_r_point = nK_D_sum;
        //py::print("przed index(Es_r, i, 0, 3) = Es_r_point.x; i = ", i );
        index(Es_r, i, 0, 3) = Es_r_point.x;
        index(Es_r, i, 1, 3) = Es_r_point.y;
        index(Es_r, i, 2, 3) = Es_r_point.z;

        index(Hs_r, i, 0, 3) = Hs_r_point.x;
        index(Hs_r, i, 1, 3) = Hs_r_point.y;
        index(Hs_r, i, 2, 3) = Hs_r_point.z;
    }
    numpy_array Es_r_numpy({(int)N,3});

    std::memcpy(
    Es_r_numpy.mutable_data(),
    Es_r,
    (N) * 3 * sizeof(std_complex)
    );

    numpy_array Hs_r_numpy({(int)N,3});

    std::memcpy(
    Hs_r_numpy.mutable_data(),
    Hs_r,
    (N) * 3 * sizeof(std_complex)
    );

    free(Hs_r);
    free(Es_r);
    return py::make_tuple(Es_r_numpy,Hs_r_numpy);
} 

numpy_array E_s_cuda(numpy_array_real r, numpy_array I_E, numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    py::print("start!",  py::arg("flush") = true);
    auto buf_I_E = I_E.request(); 
    auto buf_r = r.request(); //r to tablica o rozmiarach [liczba_punktów, 3]

    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();

    const std_complex* _I_E= static_cast<std_complex*>(buf_I_E.ptr);
    const double* _r = static_cast<double*>(buf_r.ptr);
    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);
    

    uint M = (uint)buf_edges.shape[0]-1; //number of currents
    uint N = (uint)buf_r.shape[0]; //number of points

    std_complex* Es_r = (std_complex*)calloc(3*(N), sizeof(std_complex));
    

    complex* d_I_E;
    double* d_r ;
    double* d_verticies;
    double* d_faces ;
    uint* d_edges ;
    uint* d_face_curr ;
    complex* dEs_r;
    complex* dHs_r;

    allocateMemory((void**)&dEs_r, sizeof(complex)*(3*N));
    allocateMemory((void**)&dHs_r, sizeof(complex)*(3*N));

    allocateMemory((void**)&d_I_E, sizeof(complex)*(long uint)buf_I_E.size);
    allocateMemory((void**)&d_r, sizeof(double)*(long uint)buf_r.size);
    allocateMemory((void**)&d_verticies, sizeof(double)*(long uint)buf_verticies.size);
    allocateMemory((void**)&d_faces, sizeof(double)*(long uint)buf_faces.size);
    allocateMemory((void**)&d_edges, sizeof(uint)*(long uint)(buf_edges.size));
    allocateMemory((void**)&d_face_curr, sizeof(uint)*(long uint)(buf_face_curr.size));

    copyMemoryToGPU(d_verticies, _verticies, sizeof(double)*(long uint)buf_verticies.size);
    copyMemoryToGPU(d_faces, _faces, sizeof(double)*(long uint)buf_faces.size);
    copyMemoryToGPU(d_face_curr, _face_curr, sizeof(uint)*(long uint)buf_face_curr.size);
    copyMemoryToGPU(d_edges, _edges, sizeof(uint)*(long uint)buf_edges.size);
    copyMemoryToGPU(d_I_E, _I_E, sizeof(complex)*(long uint)buf_I_E.size);
    copyMemoryToGPU(d_r, _r, sizeof(double)*(long uint)buf_r.size);

    uint threads = 128.0;
    int blocks = int((N+1)/threads)+1;
    
    E_s_kernel<<<blocks, threads>>>(M,  N, dEs_r, dHs_r, d_I_E, d_edges, d_verticies, d_faces,d_face_curr,  freq,  d_r);

    Synchronize();
    numpy_array Es_r_numpy({(int)N,3});
    copyMemoryFromGPU(Es_r_numpy.mutable_data(), reinterpret_cast<std_complex*>(dEs_r), sizeof(complex)*(N*3));

    numpy_array Hs_r_numpy({(int)N,3});
    copyMemoryFromGPU(Hs_r_numpy.mutable_data(), reinterpret_cast<std_complex*>(dHs_r), sizeof(complex)*(N*3));

    //y::print("pokoniec!");


    cudaFree(d_verticies);
    cudaFree(d_faces);
    cudaFree(d_face_curr);
    cudaFree(d_edges);
    cudaFree(d_I_E);
    cudaFree(dHs_r);
    cudaFree(d_r);

    free(Es_r);
    return py::make_tuple(Es_r_numpy,Hs_r_numpy);
} 
