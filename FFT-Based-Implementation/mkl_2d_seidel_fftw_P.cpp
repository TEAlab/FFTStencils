/**
 For compiling --> 
        export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
        export OMP_NUM_THREADS=16
        set MKL_NUM_THREADS = 16
        icc -std=gnu++98 -O3 -qopenmp -xhost -ansi-alias -ipo -AVX512 mkl_2d_seidel_fftw_P.cpp -o mkl_2d_seidel_fftw_P -lm -mkl

 For running -->
 *      ./mkl_2d_seidel_fftw_P N T numThreads
 *      Example: ./mkl_2d_seidel_fftw_P 1000 100000 1      
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

#define PB          push_back
#define SZ(x)       (int)x.size()
#define MAXN        8010

int T, N, N_THREADS;
const int BASE = 1024;
double a1[MAXN][MAXN], a2[MAXN][MAXN];
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

void print_matrix(vvd v, string msg){
    cout << msg << ": " << endl;
    for (int i = 0; i < SZ(v); i++){
        for (int j = 0; j < SZ(v[i]); j++)
            cout << v[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

void print_matrix_arr(double *v, int n, string msg){
    cout << msg << ": " << endl;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++)
            cout << v[i*n + j] << " ";
        cout << endl;
    }
    cout << endl;
}

void print_complex_matrix(double complex* input_buffer1, double complex* input_buffer2, int n, string msg){
    cout << msg << ": " << endl;
    for (int i = 0; i < n * n; i++){
        if (i % n == 0)
            cout << endl;
        printf("ratio:%f\t%f%+fi\t \t%f%+fi\n", crealf(input_buffer1[i])/crealf(input_buffer2[i]), crealf(input_buffer1[i]), cimagf(input_buffer1[i]), crealf(input_buffer2[i]), cimagf(input_buffer2[i]));
        // cout << (*input_buffer[i]).real() << " " << (*input_buffer[i]).imag() << ",\t";
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
void mkl_fft_forward(vvd &v,  double complex *output_buffer, int n)
{
    int sz_i = SZ(v), sz_j = SZ(v[0]);

    #pragma omp parallel for
    for (int i = 0; i < sz_i; i++) 
        for (int j = 0; j < sz_j; j++){
            mkl_forward_input_buffer[i*n + j] = v[i][j];
        }

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mkl_forward_output_buffer[i*n + j] = 0.0;

    // print_matrix_arr(mkl_forward_input_buffer, n, "input bufer");

    DftiComputeForward(my_desc1_handle, mkl_forward_input_buffer, mkl_forward_output_buffer);


    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            output_buffer[i*n + j] = mkl_forward_output_buffer[i*n + j];
            // printf("%f+%f\n", crealf(output_buffer[i*n + j]), cimagf(output_buffer[i*n+j]));
        }
    }
}

// Inverse DFT of complex input array
void mkl_fft_backward(double complex* input_buffer, vvd &output, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            mkl_backward_input_buffer[i*n + j] = input_buffer[i*n + j];
            mkl_backward_output_buffer[i*n + j] = 0.0;
        }

    DftiComputeBackward(my_desc2_handle, mkl_backward_input_buffer, mkl_backward_output_buffer);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            output[i][j] = mkl_backward_output_buffer[i*n + j]/(n * n * 1.0);
}

// Takes two array(a_real and b_real) as input and writes the output to "res"
void convolution_fftw_2d(vvd &a_real, vvd &input, vvd &result)
{
    if (T == 0)
        return ;

    int n_formula = N;
    mkl_fft_forward(a_real, a_complex, n_formula);
    // double complex* odd_mults = fftw_alloc_complex(n_formula * n_formula); // Do not need to allocate new space, we can use the space of a_complex or b_complex
    bool is_initialized = false; // if odd_mult array is initialized

    int t = T;
    // ############# Repeated squaring - start ############
    while (t > 1){
        if (t & 1){
            if (is_initialized == false){
                #pragma omp parallel for
                for(int i = 0; i < n_formula * n_formula; i++)
                    odd_mults[i] = a_complex[i];
                is_initialized = true;
            } else {
                #pragma omp parallel for
                for(int i = 0; i < n_formula * n_formula; i++)
                    odd_mults[i] = odd_mults[i] * a_complex[i];
            }
        }
        #pragma omp parallel for
        for(int i = 0; i < n_formula * n_formula; i++)
            a_complex[i] = a_complex[i] * a_complex[i];         
        t /= 2;
    }
    if (is_initialized){
        #pragma omp parallel for
        for(int i = 0; i < n_formula * n_formula; i++)
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
    //     for (int j = 0; j < SZ(formula[0]); j++){
    //         double r = formula[i][j] / (n_formula * n_formula);
    //         formula[i][j] = r;
    //         // formula[i][j] = (abs(r) < 1e-8? 0:r);
    //     }
    // print_matrix(formula, "Formula");

    // vvd input(N, vd(N, 0.0));
    // #pragma omp parallel for
    // for (int i = 0; i < N; i++)
    //     for (int j = 0; j < N; j++){
    //         input[i][j] = a1[i][j];
    // }
    // print_matrix(input, "Input");

    // reverse(input.begin(), input.end());

    // double complex* formula_complex = fftw_alloc_complex(N * N); 
    // fft_forward(formula, formula_complex, N);

    // double complex* input_complex = fftw_alloc_complex(N * N); 
    
    mkl_fft_forward(input, input_complex, N);
    // fft_forward(input, input_complex, N);


    // double complex* result_complex = fftw_alloc_complex(n * n); // Do not need to allocate new space, we can use the space of a_complex or b_complex
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++){
        a_complex[i] = input_complex[i] * a_complex[i];
    }

    mkl_fft_backward(a_complex, result, N);
    // fft_backward(a_complex, result, N);
    // print_matrix(result, "Result (needs to be rotated)");

    return ;
}

void mkl_init(int n)
{
    MKL_LONG status;
    MKL_LONG len[2] = {n, n};
    len[0] = n; len[1] = n;

    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_REAL, 2, len);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc1_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue( my_desc1_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );  
    status = DftiCommitDescriptor(my_desc1_handle);

    status = DftiCreateDescriptor(&my_desc2_handle, DFTI_DOUBLE, DFTI_REAL, 2, len);
    status = DftiSetValue(my_desc2_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc2_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue( my_desc2_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );  
    status = DftiCommitDescriptor(my_desc2_handle);
}

void initialize(){
    mkl_init(N);

    // forward_input_buffer = fftw_alloc_real(N * N);
    // backward_output_buffer = fftw_alloc_real(N * N);
    // forward_output_buffer = fftw_alloc_complex(N * N);
    // backward_input_buffer = fftw_alloc_complex(N * N);

    a_complex = (double complex *)malloc(sizeof(double complex) * N * N); //fftw_alloc_complex(N);
    odd_mults = (double complex *)malloc(sizeof(double complex) * N * N); //fftw_alloc_complex(N);
    input_complex = (double complex *)malloc(sizeof(double complex) * N * N); //fftw_alloc_complex(N);

    mkl_forward_input_buffer = (double *)malloc(sizeof(double) * N * N);
    mkl_backward_output_buffer = (double *)malloc(sizeof(double) * N * N);

    mkl_forward_output_buffer = (double complex *)malloc(sizeof(double complex) * N * N);
    mkl_backward_input_buffer = (double complex *)malloc(sizeof(double complex) * N * N);

    for (int i = 0; i < N+2; ++i) 
        for (int j = 0; j < N+2; j++)
            a1[i][j] = a2[i][j] = 1.0 * (rand() % BASE); 
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
}


#define getIdx(i, N)   ((i + N) % N)


bool verify(vvd result){


    for (int t = 0; t < T; ++t) {
        // cout << "t: " << t << endl;
        for (int i = 0; i < N; ++i) 
            for (int j = 0; j < N; j++){
                // a2[i] = 0.125 * (a1[i+1] - 2.0 * a1[i] + a1[i-1]); 
                // cout << i << " " << j << " : " << getIdx(i -1, N) << " " << getIdx(i + 1, N) << " " << getIdx(j - 1, N) << " " << getIdx(j + 1, N) << endl;
                // a2[i][j] = a1[getIdx(i - 1, N)][getIdx(j, N)] + a1[getIdx(i, N)][getIdx(j + 1, N)] 
                //             + a1[getIdx(i + 1, N)][getIdx(j, N)] + a1[getIdx(i, N)][getIdx(j - 1, N)];

                a2[i][j] = 0.111 * a1[getIdx(i - 1, N)][getIdx(j - 1, N)]
             + 0.111 * a1[getIdx(i - 1, N)][getIdx(j, N)] 
             + 0.111 * a1[getIdx(i - 1, N)][getIdx(j + 1, N)] 
             + 0.111 * a1[getIdx(i, N)][getIdx(j - 1, N)] 
             + 0.111 * a1[getIdx(i, N)][getIdx(j, N)] 
             + 0.111 * a1[getIdx(i, N)][getIdx(j + 1, N)] 
             + 0.111 * a1[getIdx(i + 1, N)][getIdx(j - 1, N)]
             + 0.111 * a1[getIdx(i + 1, N)][getIdx(j, N)] 
                         + 0.111 * a1[getIdx(i + 1, N)][getIdx(j + 1, N)];
        } 

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; j++)
                a1[i][j] = a2[i][j];
    }

    // cout << "Final Answer (iter): ";
    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < N; j++)
    //         cout << a1[i][j] << " ";
    //     cout << endl;
    // }
    // cout << endl;


    int cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (fabs (a1[i][j] - result[i][j]) > 1e-8)
                cnt++;

    cout << "Number of Mismatched Cell: " << cnt << endl;


    return 0;
}


int main(int argc, char *argv[])
{
    double x;
    int t, n, numThreads;
    // vvd a, b;

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

    double formula[3][3] = {{0.111, 0.111, 0.111},
                            {0.111, 0.111, 0.111},
                            {0.111, 0.111, 0.111}}; 
    // double formula[3][3] = {{1, 0, 1},
    //                         {0, 0, 0},
    //                         {0, 0, 0}}; 
    vvd a(sz_formula, vd(sz_formula));
    for (int i = 0; i < sz_formula; i++)
        for (int j = 0; j < sz_formula; j++)
            a[i][j] = formula[i][j];
    
    vvd input(n, vd(n)), result(n, vd(n,0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            input[i][j] = a1[i][j];

    long start = getTime();
#ifdef POLYBENCH
    /* Start timer. */
    polybench_start_instruments;
#endif

    convolution_fftw_2d(a, input, result);
    // Result must be rotated (T mod N) indices
    
#ifdef POLYBENCH
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
#endif
    long end = getTime();

    vvd rotated_result(n, vd(n, 0.0));
    int k = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rotated_result[i][j] = result[(i+(t%n)) % n][(j+(t%n)) % n];
    // print_matrix(rotated_result, "rotated");

    cout << N << "," << T << "," << numThreads << "," << (end - start) / 1000.0 << endl;

    mkl_destroy();
    
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
    // verify(rotated_result);
    long end_iter = getTime();

    // cout << "Time (Iter): " << end_iter - start_iter << endl;

    return 0;
}

