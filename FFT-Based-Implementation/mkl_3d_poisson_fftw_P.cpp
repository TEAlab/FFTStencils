/**
 For compiling --> 
        export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
        export OMP_NUM_THREADS=16
        set MKL_NUM_THREADS = 16
        icc -std=gnu++98 -O3 -qopenmp -xhost -ansi-alias -ipo -AVX512 mkl_3d_poisson_fftw_P.cpp -o mkl_3d_poisson_fftw_P -lm -mkl

 For running -->
 *      ./mkl_3d_poisson_fftw_P N T numThreads
 *      Example: ./mkl_3d_poisson_fftw_P 100 100000 1      
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <complex.h>
#include "mkl_service.h"
#include "mkl_dfti.h"
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>
 #include <cstdio>

 #include <omp.h>

// #include <cilk/cilk.h>
// #include <cilk/cilk_api.h>    
// #include "cilktime.h"
#ifdef USE_PAPI
#include <papi.h>    
#include "papilib.h"
#endif

#ifdef POLYBENCH
    #include <polybench.h>
#endif

using namespace std;

typedef vector<double>            vd;
typedef vector<vector<double> >   vvd;
typedef vector<vector<vector<double> > >   vvvd;

#define PB          push_back
#define SZ(x)       (int)x.size()
#define MAXN        810
#define NUM_STENCIL_PTS     19

int T, N, N_THREADS;
const int BASE = 1024;
// double a1[MAXN][MAXN][MAXN], a2[MAXN][MAXN][MAXN];
double ***a1, ***a2;

// fftw_plan plan_forward, plan_backward;  
// double *forward_input_buffer, *backward_output_buffer;  
// double complex *forward_output_buffer, *backward_input_buffer; 

double complex *a_complex, *odd_mults, *input_complex;

double *mkl_forward_input_buffer, *mkl_backward_output_buffer;
double complex *mkl_forward_output_buffer, *mkl_backward_input_buffer;

template<class T> void out(const vector<T> &a) { cout<<"array: "; for (int i=0;i<SZ(a);i++) cout<<a[i]<<" "; cout<<endl; cout.flush(); }

long getTime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    return ms;
}

void pad_matrix(vvd &v, int rows, int cols){
    int pad_rows = rows - SZ(v);
    int pad_cols = cols - SZ(v[0]);

    for (int i = 0; i < pad_rows; i++)
        v.PB(vd(SZ(v[0]), 0.0));

    for (int i = 0; i < SZ(v); i++){
        for (int j = 0; j < pad_cols; j++)
            v[i].PB(0.0);
    }
}

//  Resize a 2D matrix and pads extra entries with 0
void pad_matrices(vvd &input, vvd &formula)
{
    int n = SZ(input);

    // vvd tmp(3*n, vd(3*n, 0.0));

    // for (int i = 0; i < SZ(tmp); i++){
    //     for (int j = 0; j < SZ(tmp[0]); j++)
    //         tmp[i][j] = input[i % n][j % n];
    // }
    // input = tmp;

    if (SZ(input) < SZ(formula))
        pad_matrix(input, SZ(formula), SZ(formula[0]));
    else
        pad_matrix(formula, SZ(input), SZ(input[0]));

}

// Resizing matrices to [r1+r2, c1+c2] for circular convolution
void  pad_vectors(vd &input, vd &formula)
{
    int n = SZ(input);
    vd tmp = vd(n*3, 0);
    for (int i = 0; i < n; i++)
        tmp[i] = tmp[n + i] = tmp[n + n + i] = input[i];
    input = tmp;

    int diff = abs(SZ(input) - SZ(formula));
    for (int i = 0; i < diff; i++)
        if (SZ(input) < SZ(formula))
            input.PB(0.0);
        else
            formula.PB(0.0);
}

void print_cube(vvvd v, string msg){
    cout << msg << ": " << endl;
    for (int i = 0; i < SZ(v); i++){
        for (int j = 0; j < SZ(v[i]); j++){
            for (int k = 0; k < SZ(v[i][j]); k++)
                cout << v[i][j][k] << "\t";
                // cout << (v[i][j][k] < 1e-8 ? 0 : v[i][j][k]) << " ";

            cout << endl;
        }
        cout << "------------------"<< endl;
    }
    cout <<"================="<< endl;
}

void print_matrix(vvd v, string msg){
    cout << msg << ": " << endl;
    for (int i = 0; i < SZ(v); i++){
        for (int j = 0; j < SZ(v[i]); j++)
            cout << v[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

void print_vector(vd v, string msg){
    cout << msg << ": ";
    for (int i = 0; i < SZ(v); i++)
        cout << v[i] << " ";
    cout << endl;
}

// fftw_plan plan_forward, plan_backward;
DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL, my_desc2_handle = NULL;
// double mkl_forward_input_buffer[MAXN * MAXN], mkl_backward_output_buffer[MAXN * MAXN];
// double complex mkl_forward_output_buffer[MAXN * MAXN], mkl_backward_input_buffer[MAXN * MAXN];

// DFT of real valued matrix. CAUTION: initialize the input array after creating the plan
void mkl_fft_forward(vvvd &v, double complex *output_buffer, int n)
{
    int sz_i = SZ(v), sz_j = SZ(v[0]), sz_k = SZ(v[0][0]);  
    #pragma omp parallel for    
    for (int i = 0; i < sz_i; i++)  
        for (int j = 0; j < sz_j; j++) 
            for (int k = 0; k < sz_k; k++) 
                 mkl_forward_input_buffer[i*n*n + j*n + k] = v[i][j][k];
    // initialize the values of forward_input_buffer outside the sz_i -> n, sz_j -> n, sz_k -> n

    #pragma omp parallel for    
    for (int i = 0; i < n * n * n; i++) 
        mkl_forward_output_buffer[i] = 0.0;

    DftiComputeForward(my_desc1_handle, mkl_forward_input_buffer, mkl_forward_output_buffer);

    #pragma omp parallel for
    for (int i = 0; i < n * n * n; i++) 
        output_buffer[i] = mkl_forward_output_buffer[i];
}

// Inverse DFT of complex input array
void mkl_fft_backward(double complex* input_buffer, vvvd &output, int n)
{
    #pragma omp parallel for    
    for (int i = 0; i < n * n * n; i++){    
        mkl_backward_input_buffer[i] = input_buffer[i]; 
        mkl_backward_output_buffer[i] = 0.0; 
    }

    DftiComputeBackward(my_desc2_handle, mkl_backward_input_buffer, mkl_backward_output_buffer);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                output[i][j][k] = (mkl_backward_output_buffer[i*n*n + j*n + k])/(n * n * n * 1.0);
}


// Takes two array(a_real and b_real) as input and writes the output to "res"
void convolution_fftw_3d(vvvd &a_real, vvvd &input, vvvd &result)
{
    if (T == 0)
        return ;

    int n_formula = N;

    // double complex* a_complex = fftw_alloc_complex(n_formula * n_formula * n_formula); // ideally [rows/2 + 1, cols] will suffice 
    mkl_fft_forward(a_real, a_complex, n_formula);

    // double complex* odd_mults = fftw_alloc_complex(n_formula * n_formula * n_formula); // Do not need to allocate new space, we can use the space of a_complex or b_complex
    bool is_initialized = false;
    // if odd_mults array is initialized

    int t = T;

    // ############# Repeated squaring - start ############
    while (t > 1){
        // cout << "t: " << t << endl;
        if (t & 1){
            if (is_initialized == false){
                #pragma omp parallel for
                for(int i = 0; i < n_formula * n_formula * n_formula; i++)
                    odd_mults[i] = a_complex[i];
                is_initialized = true;
            } else {
                #pragma omp parallel for
                for(int i = 0; i < n_formula * n_formula * n_formula; i++)
                    odd_mults[i] = odd_mults[i] * a_complex[i];
            }
        }
        #pragma omp parallel for
        for(int i = 0; i < n_formula * n_formula * n_formula; i++)
            a_complex[i] = a_complex[i] * a_complex[i];         
        t /= 2;
    }
    if (is_initialized){
        #pragma omp parallel for
        for(int i = 0; i < n_formula * n_formula * n_formula; i++)
            a_complex[i] = a_complex[i] * odd_mults[i];
    }
    // ############# Repeated squaring - end ############

    // while (--t > 0){
    //     cout << "t: " << t << endl;
    //     for(int i = 0; i < n_formula * n_formula; i++)
    //         pointwise_mult[i] = pointwise_mult[i] * a_complex[i];  
    // }
 
    // fft_backward(a_complex, formula, n_formula);

    // // Scale the output array according to number of samples
    // #pragma omp parallel for
    // for (int i = 0; i < SZ(formula); i++)
    //     for (int j = 0; j < SZ(formula[i]); j++)
    //         for (int k = 0; k < SZ(formula[i][j]); k++)
    //         {
    //             double r = formula[i][j][k] / (n_formula * n_formula * n_formula);
    //             formula[i][j][k] = r;
    //         // formula[i][j] = (abs(r) < 1e-8? 0:r);
    //         }
    // print_cube(formula, "Formula");


    // print_cube(input, "Input");

    // double complex* formula_complex = fftw_alloc_complex(N * N * N); 
    // fft_forward(formula, formula_complex, N);

    // double complex* input_complex = fftw_alloc_complex(N * N * N); 
    mkl_fft_forward(input, input_complex, N);


    // double complex* result_complex = fftw_alloc_complex(n * n); // Do not need to allocate new space, we can use the space of a_complex or b_complex
    #pragma omp parallel for
    for (int i = 0; i < N * N * N; i++){
        a_complex[i] = input_complex[i] * a_complex[i];
    }

    mkl_fft_backward(a_complex, result, N);

    // #pragma omp parallel for
    // for (int i = 0; i < N; i++)
    //     for (int j = 0; j < N; j++)
    //         for (int k = 0; k < N; k++)
    //             result[i][j][k] = result[i][j][k] / (N * N * N);

    // print_cube(result, "Result: it needs to be rotated");

    return ;
}

void mkl_init(int n)
{
    MKL_LONG status;
    MKL_LONG len[3] = {n, n, n};
    len[0] = n; len[1] = n; len[2] = n;
    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_REAL, 3, len);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc1_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue( my_desc1_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );  
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiCreateDescriptor(&my_desc2_handle, DFTI_DOUBLE, DFTI_REAL, 3, len);
    status = DftiSetValue(my_desc2_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc2_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue( my_desc2_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );  
    status = DftiCommitDescriptor(my_desc2_handle);
}

void initialize(){
    mkl_init(N);
    
    a_complex = (double complex *)malloc(sizeof(double complex) * N * N * N); //fftw_alloc_complex(N);
    odd_mults = (double complex *)malloc(sizeof(double complex) * N * N * N); //fftw_alloc_complex(N);
    input_complex = (double complex *)malloc(sizeof(double complex) * N * N * N); //fftw_alloc_complex(N);
    mkl_forward_input_buffer = (double *)malloc(sizeof(double) * N * N * N);
    mkl_backward_output_buffer = (double *)malloc(sizeof(double) * N * N * N);
    mkl_forward_output_buffer = (double complex *)malloc(sizeof(double complex) * N * N * N);
    mkl_backward_input_buffer = (double complex *)malloc(sizeof(double complex) * N * N * N);
    
    a1 = (double ***) malloc((N+2) * sizeof(double **));
    a2 = (double ***) malloc((N+2) * sizeof(double **));
    for (int i = 0; i < N+2; i++){
        a1[i] = (double **) malloc((N+2) * sizeof(double *));
        a2[i] = (double **) malloc((N+2) * sizeof(double *));
        for (int j = 0; j < N+2; j++){
            a1[i][j] = (double *) malloc((N+2) * sizeof (double));
            a2[i][j] = (double *) malloc((N+2) * sizeof (double));
        }   
    }

    for (int i = 0; i < N+2; ++i) 
        for (int j = 0; j < N+2; j++)
            for (int k = 0; k < N+2; k++){
                a1[i][j][k] = a2[i][j][k] = 1.0 * (rand() % BASE); 
            }
}

void mkl_destroy(){
    MKL_LONG status;
    status = DftiFreeDescriptor(&my_desc1_handle);
    status = DftiFreeDescriptor(&my_desc2_handle);

    free(a_complex);
    free(odd_mults);
    free(input_complex);

    free(mkl_forward_input_buffer);
    free(mkl_backward_output_buffer);
    free(mkl_forward_output_buffer);
    free(mkl_backward_input_buffer);

    int i, j;
    for (i = 0; i < N+2; i++){
        for (j = 0; j < N+2; j++){
            free(a1[i][j]); // = (double *) malloc((N+2) * sizeof (double));
            free(a2[i][j]); // = (double *) malloc((N+2) * sizeof (double));
        }
        free(a1[i]); //  = (double **) malloc((N+2) * sizeof(double *));
        free(a2[i]); // = (double **) malloc((N+2) * sizeof(double *));
    }
    free(a1);
    free(a2);
}

    // vector<vector<int> > v1;
    // v1.push_back({0, 0, 0});

    // v1.push_back({-1, 0, 0});
    // v1.push_back({1, 0, 0});
    // v1.push_back({0, -1, 0});
    // v1.push_back({0, 1, 0});
    // v1.push_back({0, 0, -1});
    // v1.push_back({0, 0, 1});

    // v1.push_back({-1, -1, 0});
    // v1.push_back({1, -1, 0});
    // v1.push_back({-1, 1, 0});
    // v1.push_back({1, 1, 0});

    // v1.push_back({-1, 0, -1});
    // v1.push_back({1, 0, -1});
    // v1.push_back({0, -1, -1});
    // v1.push_back({0, 1, -1});

    // v1.push_back({-1, 0, 1});
    // v1.push_back({1, 0, 1});
    // v1.push_back({0, -1, 1});
    // v1.push_back({0, 1, 1});

#define getIdx(i, N)   ((i + N) % N)

bool verify(vvvd result){
    int dx[NUM_STENCIL_PTS] = {0, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, 0, 0};
    int dy[NUM_STENCIL_PTS] = {0, 0, 0, -1, 1, 0, 0, -1, -1, 1, 1, 0, 0, -1, 1, 0, 0, -1, 1};
    int dz[NUM_STENCIL_PTS] = {0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1};
    double dv[NUM_STENCIL_PTS] = {2.666, -0.166, -0.166, -0.166, -0.166, -0.166, -0.166, -0.0833, -0.0833, 
                    -0.0833, -0.0833, -0.0833, -0.0833, -0.0833, -0.0833, -0.0833, -0.0833, -0.0833, -0.0833};

    for (int t = 0; t < T; ++t) {
        // cout << "t: " << t << endl;
        for (int i = 0; i < N; ++i) 
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++) {
                    a2[i][j][k] = 0.0;
                    for (int p = 0; p < NUM_STENCIL_PTS; p++){
                        a2[i][j][k] += dv[p] * a1[getIdx(i+dx[p], N)][getIdx(j+dy[p], N)][getIdx(k+dz[p], N)];
                    }

                    // a2[i][j][k] = 0.125 * (a1[getIdx(i+1,N)][j][k] - 2.0 * a1[i][j][k] + a1[getIdx(i-1,N)][j][k])
                    //             + 0.125 * (a1[i][getIdx(j+1,N)][k] - 2.0 * a1[i][j][k] + a1[i][getIdx(j-1,N)][k])
                    //             + 0.125 * (a1[i][j][getIdx(k+1,N)] - 2.0 * a1[i][j][k] + a1[i][j][getIdx(k-1,N)])
                    //             + a1[i][j][k];


                } 

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    a1[i][j][k] = a2[i][j][k];
    }
/*
    cout << "Final Answer (iter): \n";
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k++)
                cout << a1[i][j][k] << "\t";
            cout << endl;
        }
        cout << "------------" << endl;
    }
    cout << "==============" << endl;
*/
    double MSE = 0.0;
    int cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++){
                MSE += (a1[i][j][k] - result[i][j][k]) * (a1[i][j][k] - result[i][j][k]);
                if (fabs (a1[i][j][k] - result[i][j][k]) > 1e-8)
                    cnt++;
            }

    cout << "Number of wrong data: " << cnt << " MSE: " << MSE / ((double)N) << endl;


    return 0;
}


int main(int argc, char *argv[])
{
    double x;
    int t, n, numThreads;

    if (argc < 4){
        cout << "Enter: N T numThreads" << endl;
        return 1;
    }

    if (argc > 1){
        n = atoi(argv[1]);
    }

    if (argc > 2)
        t = atoi(argv[2]);

    numThreads = 1;
    if (argc > 3){
        numThreads = atoi(argv[3]);
        omp_set_num_threads(numThreads);
    }

    N = n; T = t; N_THREADS = numThreads;


    initialize();

    #ifdef USE_PAPI
        papi_init();
    #endif

    int sz_formula = 3;
    // double formula[3][3] = {{0, 1, 0},
    //                         {1, 0, 1},
    //                         {0, 1, 0}}; 

    // double formula[3][3][3] = {{0, 0.125, 0},
    //                         {0.125, (-2.0*(0.125*2.0) + 1.0), 0.125},
    //                         {0, 0.125, 0}}; 

    // a2[i][j][k] = 0.125 * (a1[i+1][j][k] - 2.0 * a1[i][j][k] + a1[i-1][j][k])
    //                             + 0.125 * (a1[i][j+1][k] - 2.0 * a1[i][j][k] + a1[i][j-1][k])
    //                             + 0.125 * (a1[i][j][k+1] - 2.0 * a1[i][j][k] + a1[i][j][k-1])
    //                             + a1[i][j][k];

    // A[1][k][j][i] = 2.666*A[0][k][j][i] -
    //     (0.166*A[0][k-1][j][i] + 0.166*A[0][k+1][j][i] + 0.166*A[0][k][j-1][i] + 0.166*A[0][k][j+1][i] + 0.166*A[0][k][j][i+1] + 0.166*A[0][k][j][i-1])-
    //     (0.0833*A[0][k-1][j-1][i] + 0.0833*A[0][k+1][j-1][i] + 0.0833*A[0][k-1][j+1][i] + 0.0833*A[0][k+1][j+1][i] +
    //      0.0833*A[0][k-1][j][i-1] + 0.0833*A[0][k+1][j][i-1] + 0.0833*A[0][k][j-1][i-1] + 0.0833*A[0][k][j+1][i-1] +
    //      0.0833*A[0][k-1][j][i+1] + 0.0833*A[0][k+1][j][i+1] + 0.0833*A[0][k][j-1][i+1] + 0.0833*A[0][k][j+1][i+1]);

    vvvd a(n, vvd(n, vd(n, 0.0)));

    a[0][0][0] = 2.666;

    a[n-1][0][0] = -0.166;
    a[1][0][0] = -0.166;
    a[0][n-1][0] = -0.166;
    a[0][1][0] = -0.166;
    a[0][0][n-1] = -0.166;
    a[0][0][1] = -0.166;


    a[n-1][n-1][0] = -0.0833;
    a[1][n-1][0] = -0.0833;
    a[n-1][1][0] = -0.0833;
    a[1][1][0] = -0.0833;

    a[n-1][0][n-1] = -0.0833;
    a[1][0][n-1] = -0.0833;
    a[0][n-1][n-1] = -0.0833;
    a[0][1][n-1] = -0.0833;

    a[n-1][0][1] = -0.0833;
    a[1][0][1] = -0.0833;
    a[0][n-1][1] = -0.0833;
    a[0][1][1] = -0.0833;

    vvvd input(n, vvd(n, vd(n))), result(n, vvd(n, vd(n,0.0)));  
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            for (int k = 0; k < n; k++)
                input[i][j][k] = a1[i][j][k]; 

    // fftw_init_threads();
    // fftw_plan_with_nthreads(numThreads);

    long start = getTime();
#ifdef POLYBENCH
    /* Start timer. */
    polybench_start_instruments;
#endif

    convolution_fftw_3d(a, input, result);
    // Result must be rotated (T mod N) indices
    
#ifdef POLYBENCH
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
#endif
    long end = getTime();
/*
    vvvd rotated_result(n, vvd(n, vd(n, 0.0)));
    int k = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                rotated_result[i][j][k] = result[(i+(t%n)) % n][(j+(t%n)) % n][(k+(t%n)) % n];
*/    
    // print_cube(result, "result");

    cout << N << "," << T << "," << numThreads << "," << (end - start) / 1000.0 << endl;

#ifdef USE_PAPI        
    countTotalMiss(p);
    PAPI_shutdown();
    delete threadcounter;
    for (int i = 0; i < p; i++) delete l2miss[i];
    delete l2miss;
    delete errstring;
    delete EventSet;
    delete eventCode;
#endif

    long start_iter = getTime();
    // verifications will not work as the number becomes very large
    // verify(result);
    long end_iter = getTime();

    // cout << "Time (Iter): " << end_iter - start_iter << endl;
    mkl_destroy();

    return 0;
}
