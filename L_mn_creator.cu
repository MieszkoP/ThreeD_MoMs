#include "creator.cuh"
#include "cuda_runtime.h"
#include "custom_operators.cuh"
#include "memory_man.cuh"
//static int counter_recursivity = 0;


__device__ __host__ complex L_D_F(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result = {0.0, 0.0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        for(uint q = 0; q<N_points_quad; q++)
        {
            point rho_m_arr;
            Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
            point rho_n_arr;
            Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);

            double w_p_w_q = seven_points_Q[q][3]*seven_points_Q[p][3];
            complex left_side = vec_sc_m(rho_m_arr, rho_n_arr)*complex(0,mu*omega(freq)/4);
            complex right_side = complex(0, 1/(omega(freq)*epsilon));

            double r1[3] = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
            double r2[3] = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};

            double R = distance(r1, r2);
            complex Green = exp(complex(0,-k_f(freq)*R))/R;
            
            if(fn.isplus+fm.isplus==1)
                result = result + Green*w_p_w_q*(left_side + right_side);
            else
                result = result + Green*w_p_w_q*(left_side - right_side);
        }
    }
    return (fn.L*fm.L)/(4*PI)*result;
}

__device__ __host__ complex K_D_F(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result = {0.0, 0.0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        for(uint q = 0; q<N_points_quad; q++)
        {
            point rho_n_arr;
            Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);

            double w_p_w_q = seven_points_Q[q][3]*seven_points_Q[p][3];
            ////
            //(r -r ) x rho
            point r_min_r_prim = {fm.points_x[p] - fn.points_x[q], fm.points_y[p] - fn.points_y[q], fm.points_z[p] - fn.points_z[q]};

            point r_min_r_times_rho_n = vec_vec_m(r_min_r_prim, rho_n_arr);
            point inside = vec_vec_m(fm.normal, r_min_r_times_rho_n);

            double r1[3] = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
            double r2[3] = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};
            double R = distance(r1, r2);

            complex Green = (1.0+complex(0,k_f(freq)*R))/(pow(R,3))*exp(complex(0,-k_f(freq)*R));
            
            result = result + Green*w_p_w_q*vec_sc_m(inside, rho_m_arr);
            ////
        }
    }
    return result*(fn.L*fm.L)/(16*PI);
}

__device__ __host__ complex S_I(double R, double freq)
{
    double exponent = k_f(freq)*R;
    if(exponent<0.01)
    {
        return complex(0, -k_f(freq));
    }
    return exp(complex(0,-exponent))/R-1/R;
}

__device__ __host__ complex S_II(double R, double freq)
{
    double exponent = k_f(freq)*R;
    if(exponent<0.01)
    {
        return complex(0, -cube(k_f(freq)))/3;
    }
    return complex(1, k_f(freq)*R)*exp(complex(0, -k_f(freq)*R))/(cube(R))-complex(1+0.5*sq(k_f(freq))*sq(R), 0)/(cube(R));
}

__device__ __host__ NearD_Type generate_KN_N(uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr, point r, equationType eqType)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    point r_plus[3];
    point r_minus[3];
    double nr = vec_sc_m(fn.normal, r);
    point minus_n_nr = {-nr*fn.normal.x, -nr*fn.normal.y, -nr*fn.normal.z};
    point p = vector_sum(r, minus_n_nr);
    point r_n = {index(_verticies, fn.v_n, 0,3), index(_verticies, fn.v_n, 1,3), index(_verticies, fn.v_n, 2,3)};
    point p_n = vector_sum(r_n, vec_times_sc(vec_times_sc(fn.normal, vec_sc_m(fn.normal, r_n)),-1.0));

    point l_i[3];
    point u_i[3];
    double l_i_sc_plus[3];
    double l_i_sc_minus[3];
    double P_i_0[3];
    double R_i_0[3];
    double P_i_plus[3];
    double P_i_minus[3];
    double R_i_p[3];
    double R_i_m[3];
    double d;
    point P_i_0_vec[3];
    r_plus[0] = {index(_verticies, fn.v3, 0,3), index(_verticies, fn.v3, 1,3), index(_verticies, fn.v3, 2,3)};
    r_plus[1] = {index(_verticies, fn.v1, 0,3), index(_verticies, fn.v1, 1,3), index(_verticies, fn.v1, 2,3)};
    r_plus[2] = {index(_verticies, fn.v2, 0,3), index(_verticies, fn.v2, 1,3), index(_verticies, fn.v2, 2,3)};
    r_minus[0] = {index(_verticies, fn.v2, 0,3), index(_verticies, fn.v2, 1,3), index(_verticies, fn.v2, 2,3)};
    r_minus[1] = {index(_verticies, fn.v3, 0,3), index(_verticies, fn.v3, 1,3), index(_verticies, fn.v3, 2,3)};
    r_minus[2] = {index(_verticies, fn.v1, 0,3), index(_verticies, fn.v1, 1,3), index(_verticies, fn.v1, 2,3)};
    //std::cout<<"DUPA";
    for(int i = 0; i<3; i++)
    {
        d = vec_sc_m(fn.normal, vector_sum(r, vec_times_sc(r_plus[0], -1.0)));
        point p_plus = vector_sum(r_plus[i], vec_times_sc(vec_times_sc(fn.normal, vec_sc_m(fn.normal, r_plus[i])),-1.0));
        point p_minus = vector_sum(r_minus[i], vec_times_sc(vec_times_sc(fn.normal, vec_sc_m(fn.normal, r_minus[i])),-1.0));
        point l_i_numerator = vector_sum(p_plus, vec_times_sc(p_minus, -1.0));
        l_i[i] = vec_times_sc(l_i_numerator, 1/vec_len(l_i_numerator));
        u_i[i] = vec_vec_m(l_i[i], fn.normal);
        l_i_sc_plus[i] = vec_sc_m(vector_sum(p_plus, vec_times_sc(p, -1.0)), l_i[i]);
        l_i_sc_minus[i] = vec_sc_m(vector_sum(p_minus, vec_times_sc(p, -1.0)), l_i[i]);
        P_i_0[i] = abs(vec_sc_m(vector_sum(p_minus, vec_times_sc(p, -1.0)), u_i[i]));
        P_i_plus[i] = sqrt(pow(P_i_0[i], 2) + pow(l_i_sc_plus[i], 2.0));
        P_i_minus[i] = sqrt(pow(P_i_0[i], 2) + pow(l_i_sc_minus[i], 2.0));
        P_i_0_vec[i] = vector_sum(vector_sum(p_minus, vec_times_sc(p, -1.0)), vec_times_sc(l_i[i], -l_i_sc_minus[i]));
        P_i_0_vec[i] = vec_times_sc(P_i_0_vec[i], 1/P_i_0[i]);
        R_i_0[i] = sqrt(pow(P_i_0[i], 2) + pow(d, 2));
        //std::cout<<"R_i_0:\n";
        //std::cout<<fn_pm<<" "<<R_i_0[i] <<""<<r.x<<r.y<<r.z;
        //std::cout<<"\n";
        R_i_p[i] = sqrt(pow(P_i_plus[i], 2) + pow(d, 2));
        R_i_m[i] = sqrt(pow(P_i_minus[i], 2) + pow(d, 2));
        
        /*
        if(i==1)
        {
            point temp =vector_sum(p_minus, vec_times_sc(p, -1.0));
            py::print(
                "\np_minus =", p_minus.x, p_minus.y, p_minus.z,
                "\np =", p.x, p.y, p.z,
                "\nP_i_0[i] = ", P_i_0[i],
                "\nu_i[i] = ", u_i[i].x, u_i[i].y, u_i[i].z,
                "\nvector_sum(p_minus, vec_times_sc(p, -1.0)) = ", temp.x, temp.y, temp.z);
        }*/
    }
    point L_DNIA = {0,0,0};
    double L_DNIB = 0;

    for(int i = 0; i<3; i++)
    {
        {
        double under_brackets = (pow(R_i_0[i],2)*log((R_i_p[i]+l_i_sc_plus[i])/(R_i_m[i]+l_i_sc_minus[i]))+l_i_sc_plus[i]*R_i_p[i]-l_i_sc_minus[i]*R_i_m[i]);
        point under_sum = vec_times_sc(u_i[i], under_brackets);
        L_DNIA = vector_sum(L_DNIA, under_sum);
        }
        ///////////////////////
        {

        double under_brackets_arctan = atan((P_i_0[i]*l_i_sc_plus[i])/(pow(R_i_0[i],2)+abs(d)*R_i_p[i]))-atan((P_i_0[i]*l_i_sc_minus[i])/(pow(R_i_0[i],2)+abs(d)*R_i_m[i]));
        double under_brackets = P_i_0[i]*log((R_i_p[i]+l_i_sc_plus[i])/(R_i_m[i]+l_i_sc_minus[i]))-abs(d)*under_brackets_arctan;
        double under_sum = vec_sc_m(P_i_0_vec[i], vec_times_sc(u_i[i], under_brackets));
        L_DNIB = L_DNIB + under_sum;
        if(!isfinite(under_sum))
            r = vector_sum(r, vec_times_sc(u_i[i], SHIFT_POINT_DISTANCE_IF_NaN));
        
        
        if(!isfinite(under_sum))
        {
            //py::print("\nfn_pm=",fn_pm);
            //py::print("\ncounter = ", counter_recursivity);
            /*py::print(
        "\nfn_pm=",fn_pm,
        "\nfn.isplus=",fn.isplus,
        "\ni =", i,
        "\nr =", r.x, " ", r.y, " ", r.z,
        "\nd =", d,
        "\nR_i_p + l_i_sc_plus =", R_i_p[i] + l_i_sc_plus[i],
            "\nR_i_m + l_i_sc_minus =", R_i_m[i] + l_i_sc_minus[i],
            "\nlog argument =",
            (R_i_p[i] + l_i_sc_plus[i]) /
            (R_i_m[i] + l_i_sc_minus[i]),
        "\nP_i_0[i] =", P_i_0[i],
        "\nl_i_sc_plus[i] =", l_i_sc_plus[i],
        "\nl_i_sc_minus[i] =", l_i_sc_minus[i],
        "\nR_i_0[i] =", R_i_0[i],
        "\nR_i_p[i] =", R_i_p[i],
        "\nR_i_m[i] =", R_i_m[i],
        "\nunder_brackets_arctan =", under_brackets_arctan,
        "\nunder_brackets =", under_brackets,
        "\nunder_sum =", under_sum,
        "\nL_DNIB =", L_DNIB
        );*/}
        
        }
    }

    L_DNIA = vec_times_sc(L_DNIA, 0.5);
    //L_DNIB = 0.5*L_DNIB;

    point p_minus_p_n = vector_sum(p, vec_times_sc(p_n,-1.0));
    point L_DNI = vector_sum(L_DNIA, vec_times_sc(p_minus_p_n, L_DNIB));
    if(fn.isplus == 1)
    {
        L_DNI = vec_times_sc(L_DNI, -1.0);
    }
    else
    {   
    }
    double L_DNII = L_DNIB;

    //std::cout<<"L_DNII:\n";
    //std::cout<<L_DNII;

    if(eqType == equationType::L)
    {
        if(isfinite(L_DNII) && isfinite(L_DNI.x) && isfinite(L_DNI.y) && isfinite(L_DNI.z))
        {
            return  NearD_Type({L_DNI, L_DNII, {0,0,0}, {0,0,0}});
        }
        else
        {
            //counter_recursivity++;
            return generate_KN_N(fn_pm, _verticies, _faces, _face_curr, r, equationType::L);
        }
    }

    point K_DNIA ={0,0,0};
    double K_DNIB = 0;

    for(int i = 0; i<3; i++)
    {
        {
        double under_brackets = log((R_i_p[i]+l_i_sc_plus[i])/(R_i_m[i]+l_i_sc_minus[i]));
        point under_sum = vec_times_sc(u_i[i], under_brackets);
        K_DNIA = vector_sum(K_DNIA, under_sum);
        }
        ///////////////////////
        {
        double under_brackets_arctan = atan((abs(d)*l_i_sc_plus[i])/(R_i_p[i]*P_i_0[i]));
        under_brackets_arctan -= atan((abs(d)*l_i_sc_minus[i])/(R_i_m[i]*P_i_0[i]));
        under_brackets_arctan += atan(l_i_sc_minus[i]/P_i_0[i]);
        under_brackets_arctan -= atan(l_i_sc_plus[i]/P_i_0[i]);
        double under_sum = vec_sc_m(P_i_0_vec[i], vec_times_sc(u_i[i], under_brackets_arctan/(abs(d)+SHIFT_POINT_DISTANCE_IF_NaN)));
        K_DNIB = K_DNIB + under_sum;
        }
    }
    K_DNIA = vec_times_sc(K_DNIA, -1.0);
    K_DNIB = -1.0*K_DNIB;

    point K_DNI = vector_sum(K_DNIA, vec_times_sc(p_minus_p_n, K_DNIB));
    point K_DNII = L_DNI;

    if(fn.isplus == 1)
    {
        K_DNI = vec_times_sc(K_DNI, -1.0);
    }

    if(isfinite(K_DNI.x) && isfinite(K_DNI.y) && isfinite(K_DNI.z) && isfinite(K_DNII.x) && isfinite(K_DNII.y) && isfinite(K_DNII.z))
    {
        return  NearD_Type({{0,0,0}, 0, K_DNI, K_DNII});
    }
    else
    {
        return generate_KN_N(fn_pm, _verticies, _faces, _face_curr, r, equationType::nK);
    }

}


__device__ __host__ complex L_D_N(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result = {0,0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        sc_point resultA = {0.0,0.0,0.0};
        complex resultB = {0,0};
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        //counter_recursivity = 0;
        NearD_Type LDI_LDII = generate_KN_N(fn_pm, _verticies, _faces,  _face_curr, rp, equationType::L);
        point L_DNI = LDI_LDII.LI;
        double L_DNII = LDI_LDII.LII;
        
        for(uint q = 0; q<N_points_quad; q++)
        {
            sc_point rho_n_arr;
            Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);

            double w_q = seven_points_Q[q][3];

            //////////////////////////////////////////////
            point rq = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};

            double R = distance(rp, rq);
            resultA = vector_sum(resultA, vec_times_sc(rho_n_arr, S_I(R, freq)*w_q));
            ////////////////////////
        }
        resultA = vector_sum(vec_times_sc((sc_point)L_DNI, complex(1/fn.A, 0)), resultA);

        for(uint q = 0; q<N_points_quad; q++)
        {
            //sc_point rho_n_arr;
            //Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);

            double w_q = seven_points_Q[q][3];

            //////////////////////////////////////////////
            point rq = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};

            double R = distance(rp, rq);
            resultB = resultB + w_q*S_I(R, freq);
            ////////////////////////
        }
        resultB = complex(L_DNII/fn.A + resultB);

        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);

        double wp = seven_points_Q[p][3];
        if(fn.isplus+fm.isplus==1)
            result = result +  complex(0,wp*omega(freq)*mu/4)*vec_sc_m((sc_point)rho_m_arr, resultA) + wp*complex(0,1/(omega(freq)*epsilon))*resultB;
        else
            result = result + complex(0,wp*omega(freq)*mu/4)*vec_sc_m((sc_point)rho_m_arr, resultA) - wp*complex(0,1/(omega(freq)*epsilon))*resultB;
    }
    return (fn.L*fm.L)/(4*PI)*result;
}

__device__ __host__ complex K_D_S(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex result = {0.0, 0.0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);

        point rho_n_arr;
        Rho(p, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);

        double wp = seven_points_Q[p][3];
        result = result + wp*vec_sc_m(rho_m_arr, rho_n_arr);
    }
    return result *(fn.L*fm.L/(8*fm.A));
}

__device__ __host__ complex K_D_N(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);

    complex result = {0,0};
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;

    for(uint p = 0; p<N_points_quad; p++)
    {
        point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
        NearD_Type LDI_LDII = generate_KN_N(fn_pm, _verticies, _faces,  _face_curr, rp, equationType::nK);
        point K_DNI = LDI_LDII.KI;
        point K_DNII = LDI_LDII.KII;

        double wp = seven_points_Q[p][3];
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);

        sc_point under_sq_bracket = {0.0, 0.0, 0.0};
        sc_point under_sq_bracket_deep =  {0.0, 0.0, 0.0};
        
        for(uint q = 0; q<N_points_quad; q++)
        {
            sc_point rho_n_arr;
            Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);
            point rq = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};
            double w_q = seven_points_Q[q][3];

            double R = distance(rp, rq);
            under_sq_bracket_deep = vector_sum(under_sq_bracket_deep, vec_times_sc(rho_n_arr, S_II(R, freq)*w_q));
            ////////////////////////
        }
        ////////////////
        under_sq_bracket_deep = vector_sum((sc_point)vec_times_sc(K_DNI, 1/fn.A), under_sq_bracket_deep);
        under_sq_bracket_deep = vector_sum((sc_point)vec_times_sc(K_DNII, sq(k_f(freq))/(fn.A*2)), under_sq_bracket_deep);

        point v_n = {index(_verticies, fn.v_n, 0, 3), index(_verticies, fn.v_n, 1, 3), index(_verticies, fn.v_n, 2, 3)};
        point r_m_vn = vector_sum(rp, vec_times_sc(v_n, -1.0));

        sc_point under_bracket = vec_vec_m((sc_point)r_m_vn, under_sq_bracket_deep);
        under_sq_bracket = vec_vec_m((sc_point)fm.normal, under_bracket);

        result = result + wp*vec_sc_m((sc_point)rho_m_arr, under_sq_bracket);

    }
    return (fn.L*fm.L)/(16*PI)*result;
}

__device__ __host__ SelfD_Type LD_S_generator(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr)
{
    double a[4][4];
    point vm[4];
    point vn[4];
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    vn[1] = {index(_verticies, fn.v1, 0, 3), index(_verticies, fn.v1, 1, 3), index(_verticies, fn.v1, 2, 3)};
    vn[2] = {index(_verticies, fn.v2, 0, 3), index(_verticies, fn.v2, 1, 3), index(_verticies, fn.v2, 2, 3)};
    vn[3] = {index(_verticies, fn.v3, 0, 3), index(_verticies, fn.v3, 1, 3), index(_verticies, fn.v3, 2, 3)};
    vm[1] = {index(_verticies, fm.v1, 0, 3), index(_verticies, fm.v1, 1, 3), index(_verticies, fm.v1, 2, 3)};
    vm[2] = {index(_verticies, fm.v2, 0, 3), index(_verticies, fm.v2, 1, 3), index(_verticies, fm.v2, 2, 3)};
    vm[3] = {index(_verticies, fm.v3, 0, 3), index(_verticies, fm.v3, 1, 3), index(_verticies, fm.v3, 2, 3)};

    vm[0] = {index(_verticies, fm.v_n, 0, 3), index(_verticies, fm.v_n, 1, 3), index(_verticies, fm.v_n, 2, 3)};
    vn[0] = {index(_verticies, fn.v_n, 0, 3), index(_verticies, fn.v_n, 1, 3), index(_verticies, fn.v_n, 2, 3)};

    for(int i =0; i<4; i++)
    {
        for(int j =0; j<4; j++)
        {
            a[i][j] = vec_sc_m(vm[i], vn[j]);
        }
    }
    double l[4];
    double ln[4];
    l[1] = vec_len(vector_sum(vm[2], vec_times_sc(vm[3],-1.0)));
    l[2] = vec_len(vector_sum(vm[1], vec_times_sc(vm[3],-1.0)));
    l[3] = vec_len(vector_sum(vm[2], vec_times_sc(vm[1],-1.0)));

    ln[1] = log((sq(l[1]+l[2])-sq(l[3]))/(sq(l[2])-sq(l[3]-l[1])));
    ln[2] = log((sq(l[2]+l[3])-sq(l[1]))/(sq(l[3])-sq(l[1]-l[2])));
    ln[3] = log((sq(l[3]+l[1])-sq(l[2]))/(sq(l[1])-sq(l[2]-l[3])));

    auto lambdas_ii = [&](uint i, uint j, uint k) {
        double output =  ln[i]/(20*l[i]);
        output += (sq(l[i])+5*sq(l[j])-sq(l[k]))/(120*cube(l[j]))*ln[j];
        output += (sq(l[i])-sq(l[j])+5*sq(l[k]))/(120*cube(l[k]))*ln[k];
        output += (l[k]-l[i])/(60*sq(l[j]));
        output += (l[j]-l[i])/(60*sq(l[k]));
        return output;
    };

    auto lambdas_ij = [&](uint i, uint j, uint k)  {
        double output = (3*sq(l[i])+sq(l[j])-sq(l[k]))/(80*cube(l[i]))*ln[i];
        output += (sq(l[i])+3*sq(l[j])-sq(l[k]))/(80*cube(l[j]))*ln[j];
        output += ln[k]/(40*l[k]);
        output += (l[k]-l[j])/(40*sq(l[i]));
        output += (l[k]-l[i])/(40*sq(l[j]));
        return output;
    };

    auto lambdas_i = [&](uint i, uint j, uint k)  {
        double output = ln[i]/(8*l[i]);
        output += (sq(l[i])+5*sq(l[j])-sq(l[k]))/(48*cube(l[j]))*ln[j];
        output += (sq(l[i])-sq(l[j])+5*sq(l[k]))/(48*cube(l[k]))*ln[k];
        output += (l[k]-l[i])/(24*sq(l[j]));
        output += (l[j]-l[i])/(24*sq(l[k]));
        return output;
    };

    double L_DSII = ln[1]/(3*l[1]) +  ln[2]/(3*l[2]) +  ln[3]/(3*l[3]);

    double L_DSI = lambdas_ii(1,2,3)*(a[1][1]-2*a[1][2]+a[2][2])+lambdas_ij(1,2,3)*(a[1][1]-a[1][3]-a[1][2]+a[2][3]);
    L_DSI += lambdas_ii(2,1,3)*(a[1][1]-2*a[1][3]+a[3][3]) + lambdas_ij(1,2,3)*(a[1][1]-a[1][2]-a[1][3]+a[2][3]);
    L_DSI += lambdas_i(1,2,3)*(-a[1][1]+a[1][0]+a[1][2]-a[2][0]) + lambdas_i(2,1,3)*(-a[1][1]+a[1][0]+a[1][3]-a[3][0]);
    L_DSI += lambdas_i(1,2,3)*(-a[1][1]+a[0][1]+a[1][2]-a[0][2]) + lambdas_i(2,1,3)*(-a[1][1]+a[0][1]+a[1][3]-a[0][3]);
    L_DSI += (a[1][1] - a[1][0] - a[0][1] + a[0][0])*L_DSII;

    return SelfD_Type({L_DSI, L_DSII});
}

__device__ __host__ complex L_D_S(uint fn_pm, uint fm_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq)
{
    fn_pm_params fn = getParamsFromFn_pm(fn_pm, _verticies, _faces, _face_curr);
    fn_pm_params fm = getParamsFromFn_pm(fm_pm, _verticies, _faces, _face_curr);
    double seven_points_Q[7][4] = SEVEN_POINTS_Q_INIT;
    complex from_sum = {0.0,0.0};
    for(uint p = 0; p<N_points_quad; p++)
    {
        point rho_m_arr;
        Rho(p, fm.isplus, fm.v_n, rho_m_arr, _verticies, fm.points_x, fm.points_y, fm.points_z);
        for(uint q = 0; q<N_points_quad; q++)
        {
            point rho_n_arr;
            Rho(q, fn.isplus, fn.v_n, rho_n_arr, _verticies, fn.points_x, fn.points_y, fn.points_z);
            double wp_wq = seven_points_Q[p][3]*seven_points_Q[q][3];
            complex in_brakcet;
            point rp = {fm.points_x[p], fm.points_y[p], fm.points_z[p]};
            point rq = {fn.points_x[q], fn.points_y[q], fn.points_z[q]};
            double R = distance(rp, rq);

            if(fn.isplus+fm.isplus==1){
                in_brakcet = complex(0,omega(freq)*mu/4)*vec_sc_m(rho_m_arr, rho_n_arr) + complex(0,1/(omega(freq)*epsilon));
            }else
            {
                in_brakcet = complex(0,omega(freq)*mu/4)*vec_sc_m(rho_m_arr, rho_n_arr) - complex(0,1/(omega(freq)*epsilon));
            }
            from_sum=from_sum + wp_wq*in_brakcet*S_I(R, freq);
        }
    }
    
    SelfD_Type L_D_S = LD_S_generator(fn_pm, fm_pm, _verticies,  _faces, _face_curr);

    complex resultA = (fn.L*fm.L)/(4*PI)*from_sum;
    complex resultB = (complex(0,fn.L*fm.L*omega(freq)*mu))/(4*PI)*L_D_S.LI;
    complex resultC = complex(0, fn.L*fm.L)/(PI*omega(freq)*epsilon)*L_D_S.LII;
    if(fn.isplus+fm.isplus==1){
        return resultA-resultB+resultC;
    }else
    {
        return resultA+resultB-resultC;
    }
}

__global__ void fun_cuda(const double* _verticies, const uint* _edges, const double* _faces, const uint* _face_curr, double freq, int N, complex* Z_EJ, complex* Z_nHJ)
{
     //std::cout << " n== "<<n<<", m=="<<m<<"\n";
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=(N+1)*(N+1))
    {
        return;
    }
    uint m = uint(idx / (N + 1));
    uint n = uint(idx % (N + 1));

    if(m ==0 || n==0)
    {
        return;
    }
    //py::print("n = ", n, "m= ", m);
    uint fn_m = index(_edges, n, 1, 4);
    uint fn_p = index(_edges, n, 0, 4);
    uint fm_m = index(_edges, m, 1, 4);
    uint fm_p = index(_edges, m, 0, 4);
    uint f_pairs[4][2] = {{fn_m, fm_m}, {fn_m, fm_p}, {fn_p, fm_m}, {fn_p, fm_p}};
    index(Z_EJ, m, n, N+1) = {0.0, 0.0};
    index(Z_nHJ, m, n, N+1) = {0.0, 0.0};
    for(uint o = 0; o<4; o++)
    {
        uint fn_pm = f_pairs[o][0];
        uint fm_pm = f_pairs[o][1];
        Proximity prox = checkProximity(fm_pm, fn_pm, _verticies,  _faces, _face_curr, freq);
                //WritePriximity(prox);
        if(prox == Proximity::Far)
        {
            index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1) + L_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1)  + K_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

        if(prox == Proximity::Near)
        {
            index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1)  + L_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1)  + K_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

        if(prox == Proximity::Self)
        {
            index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1) + L_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1) + K_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

    }
}

__global__ void fun_cuda_batches(size_t batch_nr, size_t batch_size, const double* _verticies, const uint* _edges, const double* _faces, const uint* _face_curr, double freq, int N, complex* Z_EJ, complex* Z_nHJ)
{    
    size_t local = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t idx   = batch_nr * batch_size + local;
    size_t total = (size_t)(N + 1) * (N + 1);
    if (idx >= total) return;

    uint m = uint(idx / (N + 1));
    uint n = uint(idx % (N + 1));

    complex& zej  = Z_EJ[local];     // lokalny indeks w buforze batcha
    complex& znhj = Z_nHJ[local];

    zej = {0.0, 0.0};
    znhj = {0.0, 0.0};

    if(m ==0 || n==0)
    {
        return;
    }
    //py::print("n = ", n, "m= ", m);
    uint fn_m = index(_edges, n, 1, 4);
    uint fn_p = index(_edges, n, 0, 4);
    uint fm_m = index(_edges, m, 1, 4);
    uint fm_p = index(_edges, m, 0, 4);
    uint f_pairs[4][2] = {{fn_m, fm_m}, {fn_m, fm_p}, {fn_p, fm_m}, {fn_p, fm_p}};
    
    for(uint o = 0; o<4; o++)
    {
        uint fn_pm = f_pairs[o][0];
        uint fm_pm = f_pairs[o][1];
        Proximity prox = checkProximity(fm_pm, fn_pm, _verticies,  _faces, _face_curr, freq);
                //WritePriximity(prox);
        if(prox == Proximity::Far)
        {
            zej = zej + L_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            znhj = znhj  + K_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

        if(prox == Proximity::Near)
        {
            zej = zej  + L_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            znhj = znhj + K_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

        if(prox == Proximity::Self)
        {
            zej = zej + L_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
            znhj = znhj + K_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
        }

    }
}

void print_gpu_shared_mem_limits() {
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    py::print("--- LIMS PAMIĘCI SHARED ", prop.name, py::arg("flush") = true);
    py::print("Max shared memory per block:", prop.sharedMemPerBlock, (double)prop.sharedMemPerBlock / 1024.0, py::arg("flush") = true);
    py::print("Max shared memory per SM: ", prop.sharedMemPerMultiprocessor, (double)prop.sharedMemPerMultiprocessor / 1024.0, py::arg("flush") = true);
}

py::tuple L_mn(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    py::print("Start!", py::arg("flush") = true);
    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();

    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);

    uint N = (uint)buf_edges.shape[0]-1;
    complex* Z_EJ = (complex*)calloc((N+1)*(N+1), sizeof(complex));
    complex* Z_nHJ = (complex*)calloc((N+1)*(N+1), sizeof(complex));

    uint threads = 128;
    int blocks = int((N+1)*(N+1)/threads)+1;


    py::print("blocks = ",blocks,py::arg("flush") = true);
    py::print("threads = ",threads,py::arg("flush") = true);
    py::print("N = ",N,py::arg("flush") = true);
    py::print("buf_verticies.size = ",buf_verticies.size,py::arg("flush") = true);
    py::print("buf_faces.size = ",buf_faces.size,py::arg("flush") = true);
    py::print("buf_face_curr.size = ",buf_face_curr.size,py::arg("flush") = true);
    py::print("buf_edges.size = ",buf_edges.size,py::arg("flush") = true);
    py::print("_face_curr[2] = ",_face_curr[2] ,py::arg("flush") = true);
    for(uint m = 1; m<=N; m++)
    {
        for(uint n = 1; n<=N; n++)
        {
            uint fn_m = index(_edges, n, 1, 4);
            uint fn_p = index(_edges, n, 0, 4);
            uint fm_m = index(_edges, m, 1, 4);
            uint fm_p = index(_edges, m, 0, 4);
            uint f_pairs[4][2] = {{fn_m, fm_m}, {fn_m, fm_p}, {fn_p, fm_m}, {fn_p, fm_p}};
            index(Z_EJ, m, n, N+1) = {0.0, 0.0};
            index(Z_nHJ, m, n, N+1) = {0.0, 0.0};
            for(uint o = 0; o<4; o++)
            {
                uint fn_pm = f_pairs[o][0];
                uint fm_pm = f_pairs[o][1];
                // uint fm_pm, uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq
                Proximity prox = checkProximity(fm_pm, fn_pm, _verticies,  _faces, _face_curr, freq);
                if(prox == Proximity::Far)
                {
                    index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1) + L_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                    index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1)  + K_D_F(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                }

                if(prox == Proximity::Near)
                {
                    index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1)  + L_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                    index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1)  + K_D_N(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                }

                if(prox == Proximity::Self)
                {
                    index(Z_EJ, m, n, N+1) = index(Z_EJ, m, n, N+1) + L_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                    index(Z_nHJ, m, n, N+1) = index(Z_nHJ, m, n, N+1) + K_D_S(fn_pm, fm_pm, _verticies,  _faces, _face_curr, freq);
                }
            }
        }
    }

    numpy_array Z_EJ_numpy({N+1,N+1});
    numpy_array Z_nHJ_numpy({N+1,N+1});

    std::memcpy(
    Z_EJ_numpy.mutable_data(),
    Z_EJ,
    (N+1) * (N+1) * sizeof(complex)
    );

    std::memcpy(
        Z_nHJ_numpy.mutable_data(),
        Z_nHJ,
        (N+1) * (N+1) * sizeof(complex)
    );

    free(Z_EJ);
    free(Z_nHJ);
    return py::make_tuple(Z_EJ_numpy,Z_nHJ_numpy);
}

py::tuple L_mn_cuda(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    py::print("Start!", py::arg("flush") = true);
    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();

    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);

    double* d_verticies;
    double* d_faces;
    uint* d_face_curr;
    uint* d_edges;
    complex* d_Z_EJ;
    complex* d_Z_nHJ;

    uint N = (uint)buf_edges.shape[0]-1;
    std_complex* Z_EJ = (std_complex*)calloc((N+1)*(N+1), sizeof(std_complex));
    std_complex* Z_nHJ = (std_complex*)calloc((N+1)*(N+1), sizeof(std_complex));

    int threads = 128.0;
    int blocks = int((N+1)*(N+1)/threads)+1;

    allocateMemory((void**)&d_verticies, sizeof(double)*(long uint)buf_verticies.size);
    allocateMemory((void**)&d_faces, sizeof(double)*(long uint)buf_faces.size);
    allocateMemory((void**)&d_face_curr, sizeof(uint)*(long uint)buf_face_curr.size);
    allocateMemory((void**)&d_edges, sizeof(uint)*(long uint)buf_edges.size);

    allocateMemory((void**)&d_Z_EJ, sizeof(complex)*(N+1)*(N+1));
    allocateMemory((void**)&d_Z_nHJ, sizeof(complex)*(N+1)*(N+1));

    copyMemoryToGPU(d_verticies, _verticies, sizeof(double)*(long uint)buf_verticies.size);
    copyMemoryToGPU(d_faces, _faces, sizeof(double)*(long uint)buf_faces.size);
    copyMemoryToGPU(d_face_curr, _face_curr, sizeof(uint)*(long uint)buf_face_curr.size);
    copyMemoryToGPU(d_edges, _edges, sizeof(uint)*(long uint)buf_edges.size);

    py::print("blocks = ",blocks,py::arg("flush") = true);
    py::print("threads = ",threads,py::arg("flush") = true);
    py::print("N = ",N,py::arg("flush") = true);
    print_gpu_shared_mem_limits();

    py::print("buf_verticies.size = ",buf_verticies.size,py::arg("flush") = true);
    py::print("buf_faces.size = ",buf_faces.size,py::arg("flush") = true);
    py::print("buf_face_curr.size = ",buf_face_curr.size,py::arg("flush") = true);
    py::print("buf_edges.size = ",buf_edges.size,py::arg("flush") = true);
    fun_cuda<<<blocks, threads>>>(d_verticies, d_edges, d_faces, d_face_curr, freq, N, d_Z_EJ, d_Z_nHJ);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        py::print("Kernel launch error:", cudaGetErrorString(err));
    }

    Synchronize();
    numpy_array Z_EJ_numpy({N+1,N+1});
    numpy_array Z_nHJ_numpy({N+1,N+1});

    copyMemoryFromGPU(Z_EJ_numpy.mutable_data(), reinterpret_cast<std_complex*>(d_Z_EJ), sizeof(complex)*(N+1)*(N+1));
    copyMemoryFromGPU(Z_nHJ_numpy.mutable_data(), reinterpret_cast<std_complex*>(d_Z_nHJ), sizeof(complex)*(N+1)*(N+1));

    //py::print("45,15 = ",Z_EJ_numpy.at(45, 15),py::arg("flush") = true);
    //py::print("45,15 = ",Z_EJ_numpy.at(15, 45),py::arg("flush") = true);

    free(Z_EJ);
    free(Z_nHJ);
    cudaFree(d_verticies);
    cudaFree(d_faces);
    cudaFree(d_face_curr);
    cudaFree(d_edges);
    cudaFree(d_Z_EJ);
    cudaFree(d_Z_nHJ);
    return py::make_tuple(Z_EJ_numpy,Z_nHJ_numpy);
}

py::tuple L_mn_cuda_16_batches(numpy_array_real verticies, numpy_array_real faces, numpy_array_int edges, numpy_array_int face_curr, double freq)
{
    py::print("Start!", py::arg("flush") = true);
    auto buf_verticies = verticies.request();
    auto buf_faces = faces.request();
    auto buf_edges = edges.request();
    auto buf_face_curr = face_curr.request();

    const double* _verticies = static_cast<double*>(buf_verticies.ptr);
    const double* _faces = static_cast<double*>(buf_faces.ptr);
    const uint* _edges = static_cast<uint*>(buf_edges.ptr);
    const uint* _face_curr = static_cast<uint*>(buf_face_curr.ptr);

    double* d_verticies;
    double* d_faces;
    uint* d_face_curr;
    uint* d_edges;
    complex* d_Z_EJ;
    complex* d_Z_nHJ;

    uint N = (uint)buf_edges.shape[0]-1;

    size_t threads = 128.0;
    size_t blocks = size_t(size_t(N+1)*size_t(N+1)/(16*threads))+1;

    allocateMemory((void**)&d_verticies, sizeof(double)*(long uint)buf_verticies.size);
    allocateMemory((void**)&d_faces, sizeof(double)*(long uint)buf_faces.size);
    allocateMemory((void**)&d_face_curr, sizeof(uint)*(long uint)buf_face_curr.size);
    allocateMemory((void**)&d_edges, sizeof(uint)*(long uint)buf_edges.size);

    allocateMemory((void**)&d_Z_EJ, sizeof(complex)*(blocks*threads));
    allocateMemory((void**)&d_Z_nHJ, sizeof(complex)*(blocks*threads));

    copyMemoryToGPU(d_verticies, _verticies, sizeof(double)*(long uint)buf_verticies.size);
    copyMemoryToGPU(d_faces, _faces, sizeof(double)*(long uint)buf_faces.size);
    copyMemoryToGPU(d_face_curr, _face_curr, sizeof(uint)*(long uint)buf_face_curr.size);
    copyMemoryToGPU(d_edges, _edges, sizeof(uint)*(long uint)buf_edges.size);

    py::print("blocks = ",blocks,py::arg("flush") = true);
    py::print("threads = ",threads,py::arg("flush") = true);
    py::print("N = ",N,py::arg("flush") = true);
    print_gpu_shared_mem_limits();

    py::print("buf_verticies.size = ",buf_verticies.size,py::arg("flush") = true);
    py::print("buf_faces.size = ",buf_faces.size,py::arg("flush") = true);
    py::print("buf_face_curr.size = ",buf_face_curr.size,py::arg("flush") = true);
    py::print("buf_edges.size = ",buf_edges.size,py::arg("flush") = true);
    numpy_array Z_EJ_numpy({N+1,N+1});
    numpy_array Z_nHJ_numpy({N+1,N+1});

    for(size_t i =0; i<16; i++)
    {
        py::print("iteracja nr ", i);
        size_t count = std::min(threads*blocks, (size_t)(N+1)*(N+1) - (size_t)i*threads*blocks);
        if ((size_t)i*threads*blocks >= (size_t)(N+1)*(N+1)) break;
        fun_cuda_batches<<<(uint)blocks, (uint)threads>>>(i, threads*blocks, d_verticies, d_edges, d_faces, d_face_curr, freq, N, d_Z_EJ, d_Z_nHJ);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            py::print("Kernel launch error:", cudaGetErrorString(err));
        }

        Synchronize();

        copyMemoryFromGPU(Z_EJ_numpy.mutable_data()+i*(threads*blocks), reinterpret_cast<std_complex*>(d_Z_EJ), sizeof(complex)*(count));
        copyMemoryFromGPU(Z_nHJ_numpy.mutable_data()+i*(threads*blocks), reinterpret_cast<std_complex*>(d_Z_nHJ), sizeof(complex)*(count));

    }

    //py::print("45,15 = ",Z_EJ_numpy.at(45, 15),py::arg("flush") = true);
    //py::print("45,15 = ",Z_EJ_numpy.at(15, 45),py::arg("flush") = true);

    cudaFree(d_verticies);
    cudaFree(d_faces);
    cudaFree(d_face_curr);
    cudaFree(d_edges);
    cudaFree(d_Z_EJ);
    cudaFree(d_Z_nHJ);
    return py::make_tuple(Z_EJ_numpy,Z_nHJ_numpy);
}


__device__ point normalna(double* v1, double* v2, double* v3)
{
    point p1 = {v1[0], v1[1], v1[2]};
    point p2 = {v2[0], v2[1], v2[2]};
    point p3 = {v3[0], v3[1], v3[2]};

    point e1 = vector_sum(p2, vec_times_sc(p1, -1.0));   // v2 - v1
    point e2 = vector_sum(p3, vec_times_sc(p1, -1.0));   // v3 - v1

    point N = vec_vec_m(e1, e2);                          // e1 x e2
    double len = vec_len(N);

    return vec_times_sc(N, 1.0/len);                      // n_hat
}


__device__ double area(double* v1, double* v2, double* v3)
{
    point p1 = {v1[0], v1[1], v1[2]};
    point p2 = {v2[0], v2[1], v2[2]};
    point p3 = {v3[0], v3[1], v3[2]};

    point e1 = vector_sum(p2, vec_times_sc(p1, -1.0));   // v2 - v1
    point e2 = vector_sum(p3, vec_times_sc(p1, -1.0));   // v3 - v1

    point N = vec_vec_m(e1, e2);

    return 0.5 * vec_len(N);
}

__global__ void Test_LDNI_kernel(double* v1, double* v2, double* v3, double* r, NearD_Type* result)
{
    double _verticies[9];
    double _faces[7];
    uint _face_curr[4];

    index(_verticies, 0, 0,3) = v1[0];
    index(_verticies, 1, 0,3) = v2[0];
    index(_verticies, 2, 0,3) = v3[0];

    index(_verticies, 0, 1,3) = v1[1];
    index(_verticies, 1, 1,3) = v2[1];
    index(_verticies, 2, 1,3) = v3[1];

    index(_verticies, 0, 2,3) = v1[2];
    index(_verticies, 1, 2,3) = v2[2];
    index(_verticies, 2, 2,3) = v3[2];

    index(_faces, 0, 0,7) = 0;
    index(_faces, 0, 1,7) = 1;
    index(_faces, 0, 2,7) = 2;

    index(_faces, 0, 3,7) = area(v1, v2, v3);
    index(_faces, 0, 4,7) = normalna(v1, v2, v3).x;
    index(_faces, 0, 5,7) = normalna(v1, v2, v3).y;
    index(_faces, 0, 6,7) = normalna(v1, v2, v3).z;

    index(_face_curr, 0, 0,3) = 0;
    index(_face_curr, 0, 1,3) = 0; //nie ma znaczenie
    index(_face_curr, 0, 2,3) = 0;

    point _r = {r[0], r[1], r[2]};

    *result = generate_KN_N(0, _verticies, _faces, _face_curr, _r, equationType::L);
//__device__ NearD_Type generate_KN_N(uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr, point r, equationType eqType)

}

numpy_array_real Test_LDNI(numpy_array_real v1, numpy_array_real v2, numpy_array_real v3, numpy_array_real r )
{
    auto buf_v1 = v1.request();
    auto buf_v2 = v2.request();
    auto buf_v3 = v3.request();
    auto buf_r = r.request();

    py::print("--- Przed rzutowaniem ", py::arg("flush") = true);

    const double* _v1 = static_cast<double*>(buf_v1.ptr);
    const double* _v2 = static_cast<double*>(buf_v2.ptr);
    const double* _v3 = static_cast<double*>(buf_v3.ptr);
    const double* _r = static_cast<double*>(buf_r.ptr);

    double* d_v1;
    double* d_v2;
    double* d_v3;
    double* d_r;

    NearD_Type* result;

    py::print("--- Przed alokacja ", py::arg("flush") = true);

    allocateMemory((void**)&d_v1, sizeof(double)*(long uint)buf_v1.size);
    allocateMemory((void**)&d_v2, sizeof(double)*(long uint)buf_v2.size);
    allocateMemory((void**)&d_v3, sizeof(double)*(long uint)buf_v3.size);
    allocateMemory((void**)&d_r, sizeof(double)*(long uint)buf_r.size);
    allocateMemory((void**)&result, sizeof(NearD_Type));

    copyMemoryToGPU(d_v1, _v1, sizeof(double)*(long uint)buf_v1.size);
    copyMemoryToGPU(d_v2, _v2, sizeof(double)*(long uint)buf_v2.size);
    copyMemoryToGPU(d_v3, _v3, sizeof(double)*(long uint)buf_v3.size);
    copyMemoryToGPU(d_r, _r, sizeof(double)*(long uint)buf_r.size);

    py::print("--- Przed kernelem ", py::arg("flush") = true);
    Test_LDNI_kernel<<<1,1>>>(d_v1, d_v2, d_v3 , d_r, result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        py::print("Kernel launch error:", cudaGetErrorString(err));
    }

    numpy_array_real result_numpy({4});
    copyMemoryFromGPU(result_numpy.mutable_data(), reinterpret_cast<double*>(result), sizeof(double)*4);

    Synchronize();

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_r);
    cudaFree(result);

    return result_numpy;
}