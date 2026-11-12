#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <format>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace std::complex_literals;
//using complex = cuDoubleComplex;

typedef std::complex<double> std_complex;
typedef std::vector<std::vector<std::vector<std_complex>>> vector3D;
typedef std::vector<std::vector<std::vector<double>>> vector3D_real;
typedef py::array_t<std_complex> numpy_array;
typedef py::array_t<double> numpy_array_real;
typedef py::array_t<int> numpy_array_int;

//Około 8 miliony wierzchołków jest ograniczeniem wynikającym z zaokrąglania double
py::tuple create_edge_tables(numpy_array_real u_n, numpy_array_real Delta_n, int num_of_handles)
{
    int w = 0;
    int s = 0;
    auto data_u_n = u_n.unchecked<2>();
    auto data_Delta_n = Delta_n.unchecked<2>();
    int G = num_of_handles;

    int N = (int)data_u_n.shape(0)-1; //number of vertices
    int S = (int)data_Delta_n.shape(0)-1; //number of faces
    numpy_array_int fn_pm({2*(3*N-6+6*G)+1, 3});
    numpy_array_int fn({3*N-6+6*G+1, 4});
    auto data_fn_pm= fn_pm.mutable_unchecked<2>();
    auto data_fn = fn.mutable_unchecked<2>();

    for(int j = 0; j<(3*N-6+6*G+1); j++)
    {
        data_fn(j,0) = -1;
        data_fn(j,1) = -1;
        data_fn(j,2) = -1;
        data_fn(j,3) = -1;
    }

    for(int i = 1; i<=S; i++)
    {
        for(int k =1; k<=3; k++)
        {
            w++;
            int v_1id = static_cast<int>(std::round(data_Delta_n(i,k%3)));
            int v_2id = static_cast<int>(std::round(data_Delta_n(i,(k+1)%3)));
            int v_n = static_cast<int>(std::round(data_Delta_n(i,k-1)));
            ///////////////CHECKING IF IT IS ALREAD PRESENT//////////////////
            bool present = false;
            int j = 1;
            for(; j<=s; j++)
            {
                if(data_fn(j,2)==v_1id && data_fn(j,3)==v_2id)
                {
                    present = true;
                    break;
                }

                if(data_fn(j,2)==v_2id && data_fn(j,3)==v_1id)
                {
                    present = true;
                    break;
                }
            }
            //////////////////////////////////////////////
            if(present)
            {
                data_fn_pm(w, 0) = i;
                data_fn_pm(w, 1) = v_n;
                data_fn_pm(w, 2) = 0;
                data_fn(j, 1) = w;
            }
            else
            {
                s++;
                data_fn(s, 2) = v_1id;
                data_fn(s, 3) = v_2id;
                data_fn_pm(w, 0) = i;
                data_fn_pm(w, 1) = v_n;
                data_fn_pm(w, 2) = 1;
                data_fn(j, 0) = w;
            }
        }
    }
    return py::make_tuple(fn, fn_pm);
}

//py::tuple L_mn(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);
py::tuple V_m(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, numpy_array_real k, double freq, const std::string& pol);
py::tuple E_s(numpy_array_real r, numpy_array I_E, numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);
py::tuple E_s_cuda(numpy_array_real r, numpy_array I_E, numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);

py::tuple L_mn(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);
py::tuple L_mn_cuda(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);
py::tuple L_mn_cuda_16_batches(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq);

numpy_array_real Test_LDNI(numpy_array_real v1, numpy_array_real v2, numpy_array_real v3, numpy_array_real r );

PYBIND11_MODULE(ThreeD_MoMs, m) {
    m.def("create_edge_tables", &create_edge_tables);
    m.def("L_mn", &L_mn);
    m.def("L_mn_cuda", &L_mn_cuda);
    m.def("L_mn_cuda_16_batches", &L_mn_cuda_16_batches);
    m.def("V_m", &V_m);
    m.def("E_s", &E_s);
    m.def("E_s_cuda", &E_s_cuda);
    m.def("Test_LDNI", &Test_LDNI);
}