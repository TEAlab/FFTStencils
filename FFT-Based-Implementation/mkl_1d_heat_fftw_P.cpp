/**
 * For compiling --> 
        export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
        export OMP_NUM_THREADS=68
        set MKL_NUM_THREADS = 64
        icc -std=gnu++98 -O3 -qopenmp -xhost -ansi-alias -ipo -AVX512 mkl_1d_head_fftw_P.cpp -o mkl_1d_heat_fftw_P -lm -mkl

 * For running -->
 *      ./1d_heat_fftw_P N T numThreads
 *      Example: ./1d_heat_fftw_P 1000 100000 1
 *          
 */


#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <complex.h>
#include "mkl_dfti.h"
// #include <fftw3.h>
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
#define MAXN        10001000

int T, N, N_THREADS;
const int BASE = 1024;
double a1[MAXN], a2[MAXN];
double complex *a_complex, *odd_mults, *input_complex;

template<class T> void out(const vector<T> &a) { cout<<"array: "; for (int i=0;i<SZ(a);i++) cout<<a[i]<<" "; cout<<endl; cout.flush(); }


long getTime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    return ms;
}

void print_vector(vd v, string msg){
    cout << msg << ": ";
    for (int i = 0; i < SZ(v); i++)
        cout << v[i] << " ";
    cout << endl;
}

void print_complex_vector(double complex *buffer, int n, string msg){
    cout << msg << ": ";
    for (int i = 0; i < n; i++)
        printf("%f%+fi\n", crealf(buffer[i]), cimagf(buffer[i]));
}

// fftw_plan plan_forward, plan_backward;
DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL, my_desc2_handle = NULL;
double mkl_forward_input_buffer[MAXN], mkl_backward_output_buffer[MAXN];
double complex mkl_forward_output_buffer[MAXN], mkl_backward_input_buffer[MAXN];



// DFT of real valued matrix. CAUTION: initialize the input array after creating the plan
void mkl_fft_forward(vd &v,  double complex *output_buffer, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < SZ(v); i++) // comment ASB: replace SZ() and use profiler like vtune/tau
            mkl_forward_input_buffer[i] = v[i];

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
            mkl_forward_output_buffer[i] = 0.0;

    DftiComputeForward(my_desc1_handle, mkl_forward_input_buffer, mkl_forward_output_buffer);


    #pragma omp parallel for
    for (int i = 0; i < n; i++)
            output_buffer[i] = mkl_forward_output_buffer[i];
}

// Inverse DFT of complex input array
void mkl_fft_backward(double complex* input_buffer, vd &output, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        mkl_backward_input_buffer[i] = input_buffer[i];
        mkl_backward_output_buffer[i] = 0.0;
    }

    DftiComputeBackward(my_desc2_handle, mkl_backward_input_buffer, mkl_backward_output_buffer);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        output[i] = mkl_backward_output_buffer[i];
}

void convolution_fftw_1d(vd &a_real, vd &input, vd &result)
{
    if (T == 0)
        return ;

    int n_formula = N;
    mkl_fft_forward(a_real, a_complex, n_formula);

    bool is_initialized = false; // if odd_mult array is initialized

    int t = T;
    // ############# Repeated squaring - start ############
    while (t > 1){
        if (t & 1){
            if (is_initialized == false){
                #pragma omp parallel for
                for(int i = 0; i < n_formula; i++)
                    odd_mults[i] = a_complex[i];
                is_initialized = true;
            } else {
                #pragma omp parallel for
                for(int i = 0; i < n_formula; i++)
                    odd_mults[i] = odd_mults[i] * a_complex[i];
            }
        }
        #pragma omp parallel for
        for(int i = 0; i < n_formula; i++)
            a_complex[i] = a_complex[i] * a_complex[i];         
        t /= 2;
    }
    if (is_initialized){
        #pragma omp parallel for
        for(int i = 0; i < n_formula; i++)
            a_complex[i] = a_complex[i] * odd_mults[i];
    }
    // ############# Repeated squaring - end ############
/*
    // while (--t > 0){
    //     #pragma omp parallel for
    //     for(int i = 0; i < n_formula; i++)
    //         pointwise_mult[i] = pointwise_mult[i] * a_complex[i];  
    // }
*/
    // fft_backward(a_complex, formula, n_formula);

    // // Scale the output array according to number of samples
    // #pragma omp parallel for
    // for (int i = 0; i < SZ(formula); i++)
    // {
    //     double r = formula[i] / (n_formula);
    //     formula[i] = r;
    // }

    // // print_vector(formula, "Formula");

    // double complex* formula_complex = fftw_alloc_complex(n); 
    // fft_forward(formula, formula_complex, n);

    mkl_fft_forward(input, input_complex, N);
    // print_complex_vector(input_complex, N, "input_complex:");

    #pragma omp parallel for
    for (int i = 0; i < N; i ++){
        a_complex[i] = input_complex[i] * a_complex[i];
    }
    // print_complex_vector(a_complex, N, "Result_complex:");

    mkl_fft_backward(a_complex, result, N);

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        result[i] = result[i] / (N);
    // print_vector(result, "Result (needs to be rotated)");
    return ;
}

void mkl_init(int n)
{
    MKL_LONG status;
    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_REAL, 1, n);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    // status = DftiSetValue(my_desc1_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(my_desc1_handle);

    status = DftiCreateDescriptor(&my_desc2_handle, DFTI_DOUBLE, DFTI_REAL, 1, n);
    status = DftiSetValue(my_desc2_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc2_handle);
}

void initialize(){
    mkl_init(N);
    a_complex = (double complex *)malloc(sizeof(double complex) * N); //fftw_alloc_complex(N);
    odd_mults = (double complex *)malloc(sizeof(double complex) * N); //fftw_alloc_complex(N);
    input_complex = (double complex *)malloc(sizeof(double complex) * N); //fftw_alloc_complex(N);

    for (int i = 0; i < N+2; ++i) {
            a1[i] = a2[i] = 1.0 * (rand() % BASE); 
    }
}

void mkl_destroy(){
    MKL_LONG status;
    status = DftiFreeDescriptor(&my_desc1_handle);
    status = DftiFreeDescriptor(&my_desc2_handle);
    free(a_complex);
    free(odd_mults);
    free(input_complex);
}

#define getIdx(i, N)   ((i + N) % N)


bool verify(vd &result){

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            a2[i] = 0.125 * (a1[getIdx(i+1, N)] - 2.0 * a1[i] + a1[getIdx(i-1, N)]) + a1[i]; 
            // a2[i] = a1[getIdx(i - 1, N)] + a1[getIdx(i + 1, N)];
            // a2[i] = a1[i];
        } 

        for (int i = 0; i < N; ++i) 
            a1[i] = a2[i];
    }

    // print_vector(result, "Result: ");
    // cout << "Final Answer (iter): ";
    // for (int i = 0; i < N; i++)
    //     cout << a1[i] << " ";
    // cout << endl;

    double MSE = 0.0;
    int cnt = 0; 
    for (int i = 0; i < N; i++){
        MSE += (a1[i] - result[i]) * (a1[i] - result[i]);
        if (fabs(a1[i] - result[i]) > 1e-8)
            cnt++;
    }


    cout << "Number of wrong data: " << cnt << " MSE: " << MSE / ((double)N) << endl;

    return 0;
}


int main(int argc, char *argv[])
{
    int t, n, numThreads;
    vd a, b;

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

    a.PB(0.125);
    a.PB((-1) * 0.125 * 2.0 + 1);
    a.PB(0.125);
    // a.PB(0);
    // a.PB(0);
    
    // a.PB(1);
    // a.PB(0);
    // a.PB(1);
    
    // a.PB(1);

    vd input(n), result(n, 0.0);
    for (int i = 0; i < n; i++)
        input[i] = a1[i];

    mkl_init(N);

    long start = getTime();
#ifdef POLYBENCH
    /* Start timer. */
    polybench_start_instruments;
#endif

    convolution_fftw_1d(a, input, result);
    // Result must be rotated (T mod N) indices
    
#ifdef POLYBENCH
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
#endif
    long end = getTime();

    vd rotated_result(SZ(result));
    for (int i = 0; i < n; i++)
        rotated_result[i] = result[(i+(t%n)) % n];

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


