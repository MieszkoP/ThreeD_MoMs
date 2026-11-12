#include "creator_d.cuh"

;
__host__ __device__ void WritePriximity(Proximity prox)
{
    switch(prox)
    {
        case Proximity::Far:
            //std::cout << "Far !\n";
            break;
        case Proximity::Near:
           // std::cout << "Near !\n";
            break;
        case Proximity::Self:
          //  std::cout << "Self !\n";
            break;

    }
}

__host__ __device__ void mean_point(unsigned int v1, unsigned int v2, unsigned int v3, point& mean, const double* _verticies)
{
    double v1x = index(_verticies, v1, 0, 3);
    double v1y = index(_verticies, v1, 1, 3);
    double v1z = index(_verticies, v1, 2, 3);

    double v2x = index(_verticies, v2, 0, 3);
    double v2y = index(_verticies, v2, 1, 3);
    double v2z = index(_verticies, v2, 2, 3);

    double v3x = index(_verticies, v3, 0, 3);
    double v3y = index(_verticies, v3, 1, 3);
    double v3z = index(_verticies, v3, 2, 3);

    mean.x = (v1x+v2x+v3x)/3;
    mean.y = (v1y+v2y+v3y)/3;
    mean.z = (v1z+v2z+v3z)/3;
}

__host__ __device__ double distance(double* v1, double* v2)
{
    return sqrt(pow((v1[0]-v2[0]),2)+pow((v1[1]-v2[1]),2)+pow((v1[2]-v2[2]),2));
}

__host__ __device__ double distance(uint v1, uint v2, const double* _verticies)
{
    double v1_arr[3];
    double v2_arr[3];
    v1_arr[0] = index(_verticies, v1, 0, 3);
    v1_arr[1] = index(_verticies, v1, 1, 3);
    v1_arr[2] = index(_verticies, v1, 2, 3);
    v2_arr[0] = index(_verticies, v2, 0, 3);
    v2_arr[1] = index(_verticies, v2, 1, 3);
    v2_arr[2] = index(_verticies, v2, 2, 3);
    return distance(v1_arr, v2_arr);
}

__host__ __device__ Proximity checkProximity(uint fm_pm, uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    uint face_m = index(_face_curr, fm_pm, 0, 3);
    uint face_n = index(_face_curr, fn_pm, 0, 3);
    if(face_m == face_n)
    {
        return Proximity::Self;
    }
    
    uint n_v1 = (uint)round(index(_faces, face_n, 0, 7));
    uint n_v2 = (uint)round(index(_faces, face_n, 1, 7));
    uint n_v3 = (uint)round(index(_faces, face_n, 2, 7));

    uint m_v1 = (uint)round(index(_faces, face_m, 0, 7));
    uint m_v2 = (uint)round(index(_faces, face_m, 1, 7));
    uint m_v3 = (uint)round(index(_faces, face_m, 2, 7));
    point mean_n;
    point mean_m;
    mean_point(n_v1, n_v2, n_v3, mean_n, _verticies);
    mean_point(m_v1, m_v2, m_v3, mean_m, _verticies);
    double _lambda = c_v/freq;
    //double D = 2*_lambda; //
    //if(distance(mean_n, mean_m) > 0.6*pow(D,2)/_lambda)
    if(distance(mean_n, mean_m) > 1.4)
    {
        return Proximity::Far;
    }
    else
    {
        return Proximity::Near;
    }

}

__host__ __device__ double GetLengthOfEdge(uint fn_pn, const uint* _face_curr, const double* _faces, const double* _verticies)
{
    uint triangle_n = index(_face_curr, fn_pn, 0, 3);
    uint v_n = index(_face_curr, fn_pn, 1, 3);

    uint v1 = (uint)round(index(_faces, triangle_n, 0, 7));
    uint v2 = (uint)round(index(_faces, triangle_n, 1, 7));
    uint v3 = (uint)round(index(_faces, triangle_n, 2, 7));
    double Ln;
    if(v1==v_n)
    {
        Ln = distance(v2, v3, _verticies);
    }

    if(v2==v_n)
    {
        Ln = distance(v1, v3, _verticies);
    }

    if(v3==v_n)
    {
        Ln = distance(v2, v1, _verticies);
    }
    return Ln;
}

__host__ __device__ void vector_sum(double* A, double* B, point &sum)
{
    sum.x = A[0]+B[0];
    sum.y = A[1]+B[1];
    sum.z = A[2]+B[2];
}

__host__ __device__ double vec_sc_m(double* A, double* B)
{
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

__host__ __device__ void vec_vec_m(double* A, double* B, double* result)
{
    result[0] = A[1]*B[2] - A[2]*B[1];
    result[1] = A[2]*B[0] - A[0]*B[2];
    result[2] = A[0]*B[1] - A[1]*B[0];
}

__host__ __device__ void get_points_from_triangle(uint v1, uint v2, uint v3, double* points_x, double* points_y, double* points_z, const double* _verticies)
{
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;

    double v1x = index(_verticies, v1, 0, 3);
    double v1y = index(_verticies, v1, 1, 3);
    double v1z = index(_verticies, v1, 2, 3);

    double v2x = index(_verticies, v2, 0, 3);
    double v2y = index(_verticies, v2, 1, 3);
    double v2z = index(_verticies, v2, 2, 3);

    double v3x = index(_verticies, v3, 0, 3);
    double v3y = index(_verticies, v3, 1, 3);
    double v3z = index(_verticies, v3, 2, 3);
    
    for(int p = 0; p<7; p++)
    {
        points_x[p] = v1x*seven_points_Q[p][0]+v2x*seven_points_Q[p][1]+v3x*seven_points_Q[p][2];
        points_y[p] = v1y*seven_points_Q[p][0]+v2y*seven_points_Q[p][1]+v3y*seven_points_Q[p][2];
        points_z[p] = v1z*seven_points_Q[p][0]+v2z*seven_points_Q[p][1]+v3z*seven_points_Q[p][2];
    }
}

__host__ __device__ double omega(double freq)
{
    return 2*PI*freq;
}

__host__ __device__ double k_f(double freq)
{
    double _lambda = c_v/freq;
    return 2*PI/_lambda;
}

__host__ __device__ fn_pm_params getParamsFromFn_pm(uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr)
{
    fn_pm_params output;
    output.triangle = index(_face_curr, fn_pm, 0, 3);
    output.v_n = index(_face_curr, fn_pm, 1, 3);
    output.isplus = index(_face_curr, fn_pm, 2, 3);

    output.v1 = (uint)round(index(_faces, output.triangle, 0, 7));
    output.v2 = (uint)round(index(_faces, output.triangle, 1, 7));
    output.v3 = (uint)round(index(_faces, output.triangle, 2, 7));
    output.L = GetLengthOfEdge(fn_pm, _face_curr, _faces, _verticies);
    output.A = index(_faces, output.triangle, 3, 7);
    output.normal.x = index(_faces, output.triangle, 4, 7);
    output.normal.y = index(_faces, output.triangle, 5, 7);
    output.normal.z = index(_faces, output.triangle, 6, 7);
    get_points_from_triangle(output.v1, output.v2, output.v3, output.points_x, output.points_y, output.points_z, _verticies);
    return output;
}

///////////////////////////////////////////////