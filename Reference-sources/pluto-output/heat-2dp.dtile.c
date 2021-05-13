#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*
 * Discretized 2D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/*
 * N is the number of points
 * T is the number of timesteps
 */
#ifdef HAS_DECLS
#include "decls.h"
#else
#define N 8000L
#define T 100000L
#endif

#define NUM_FP_OPS 10

/* Define our arrays */
double A[2][N][N];
double total=0; double sum_err_sqr=0;
int chtotal=0;
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;

        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }

    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;

        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    return x->tv_sec < y->tv_sec;
}

int main(int argc, char * argv[]) {
    long int t, i, j, k;
    const int BASE = 1024;

    // for timekeeping
    int ts_return = -1;
    struct timeval start, end, result;
    double tdiff = 0.0;

    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N*N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[0][i][j] = 1.0 * (rand() % BASE);
        }
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif

#undef N
#define N 4000L






    int t1, t2, t3, t4, t5, t6;

    int lb, ub, lbp, ubp, lb2, ub2;
    register int lbv, ubv;

    /* Start of CLooG code */
    if ((N >= 1) && (T >= 1)) {
        for (t1=-1;t1<=floord(T-1,16);t1++) {
            lbp=ceild(t1,2);
            ubp=min(floord(T+N-2,32),floord(16*t1+N+14,32));
#pragma omp parallel for private(lbv,ubv,t2,t3,t4,t5,t6)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=max(0,ceild(t1-7,8));t3<=min(floord(T+N-2,128),floord(16*t1+N+30,128));t3++) {
                    if(t2==lbp ||  t3<= max(0,ceild(t1-3,4))+1){
                        for (t4=max(max(max(0,16*t1),32*t2-N+1),128*t3-N+1);t4<=min(min(min(T-1,16*t1+31),128*t3+127),32*t1-32*t2+N+30);t4++) {
                            for (t5=max(max(32*t2,t4),-32*t1+32*t2+2*t4-31);t5<=min(min(32*t2+31,-32*t1+32*t2+2*t4),t4+N-1);t5++) {
                                lbv=max(128*t3,t4);
                                ubv=min(128*t3+127,t4+N-1);
#pragma ivdep
#pragma vector always
                                for (t6=lbv;t6<=ubv;t6++) {
                                    A[(t4+1)%2][-t4+t5][-t4+t6]                =0.125*(A[t4%2][-t4+t5+1][-t4+t6]-2.0*A[t4%2][-t4+t5][-t4+t6]+A[t4%2][(t4==t5)?(2*N-1):(-t4+t5-1)][-t4+t6])                                           +0.125*(A[t4%2][-t4+t5][-t4+t6+1]-2.0*A[t4%2][-t4+t5][-t4+t6]+A[t4%2][-t4+t5][(t4==t6)?(2*N-1):(-t4+t6-1)])+A[t4%2][-t4+t5][-t4+t6];;
                                    A[(t4+1)%2][2*N-1-(-t4+t5)][-t4+t6]        =0.125*(A[t4%2][(t4==t5)?0:(2*N-1-(-t4+t5)+1)][-t4+t6]-2.0*A[t4%2][2*N-1-(-t4+t5)][-t4+t6]+A[t4%2][2*N-1-(-t4+t5)-1][-t4+t6])                         +0.125*(A[t4%2][2*N-1-(-t4+t5)][-t4+t6+1]-2.0*A[t4%2][2*N-1-(-t4+t5)][-t4+t6]+A[t4%2][2*N-1-(-t4+t5)][(t4==t6)?(2*N-1):(-t4+t6-1)])+A[t4%2][2*N-1-(-t4+t5)][-t4+t6];;
                                    A[(t4+1)%2][-t4+t5][2*N-1-(-t4+t6)]        =0.125*(A[t4%2][-t4+t5+1][2*N-1-(-t4+t6)]-2.0*A[t4%2][-t4+t5][2*N-1-(-t4+t6)]+A[t4%2][(t4==t5)?(2*N-1):(-t4+t5-1)][2*N-1-(-t4+t6)])                   +0.125*(A[t4%2][-t4+t5][(t4==t6)?0:(2*N-1-(-t4+t6)+1)]-2.0*A[t4%2][-t4+t5][2*N-1-(-t4+t6)]+A[t4%2][-t4+t5][2*N-1-(-t4+t6)-1])+A[t4%2][-t4+t5][2*N-1-(-t4+t6)];;
                                    A[(t4+1)%2][2*N-1-(-t4+t5)][2*N-1-(-t4+t6)]=0.125*(A[t4%2][(t4==t5)?0:(2*N-1-(-t4+t5)+1)][2*N-1-(-t4+t6)]-2.0*A[t4%2][2*N-1-(-t4+t5)][2*N-1-(-t4+t6)]+A[t4%2][2*N-1-(-t4+t5)-1][2*N-1-(-t4+t6)]) +0.125*(A[t4%2][2*N-1-(-t4+t5)][(t4==t6)?0:(2*N-1-(-t4+t6)+1)]-2.0*A[t4%2][2*N-1-(-t4+t5)][2*N-1-(-t4+t6)]+A[t4%2][2*N-1-(-t4+t5)][2*N-1-(-t4+t6)-1])+A[t4%2][2*N-1-(-t4+t5)][2*N-1-(-t4+t6)];;
                                }
                            }
                        }
                    }
                    else {
                        for (t4=max(max(max(0,16*t1),32*t2-N+1),128*t3-N+1);t4<=min(min(min(T-1,16*t1+31),128*t3+127),32*t1-32*t2+N+30);t4++) {
                            if(t4%2==0){
                                for (t5=max(max(32*t2,t4),-32*t1+32*t2+2*t4-31);t5<=min(min(32*t2+31,-32*t1+32*t2+2*t4),t4+N-1);t5++) {
                                    lbv=max(128*t3,t4);
                                    ubv=min(128*t3+127,t4+N-1);
#pragma ivdep
#pragma vector always
                                    for (t6=lbv;t6<=ubv;t6++) {
                                        A[1][-t4+t5][-t4+t6]=0.125*(A[0][-t4+t5+1][-t4+t6]-2.0*A[0][-t4+t5][-t4+t6]+A[0][-t4+t5-1][-t4+t6])+0.125*(A[0][-t4+t5][-t4+t6+1]-2.0*A[0][-t4+t5][-t4+t6]+A[0][-t4+t5][-t4+t6-1])+A[0][-t4+t5][-t4+t6];;
                                    }
#pragma ivdep
#pragma vector always
                                    for (t6=lbv;t6<=ubv;t6++) {
                                        A[1][2*N-1-(-t4+t5)][-t4+t6]=0.125*(A[0][2*N-1-(-t4+t5)+1][-t4+t6]-2.0*A[0][2*N-1-(-t4+t5)][-t4+t6]+A[0][2*N-1-(-t4+t5)-1][-t4+t6])+0.125*(A[0][2*N-1-(-t4+t5)][-t4+t6+1]-2.0*A[0][2*N-1-(-t4+t5)][-t4+t6]+A[0][2*N-1-(-t4+t5)][-t4+t6-1])+A[0][2*N-1-(-t4+t5)][-t4+t6];;
                                    }
                                    lbv=2*N-1-max(128*t3,t4);
                                    ubv=2*N-1-min(128*t3+127,t4+N-1);
#pragma ivdep
#pragma vector always
                                    for (t6=ubv;t6<=lbv;t6++) {
                                        A[1][-t4+t5][t4+t6]=0.125*(A[0][-t4+t5+1][t4+t6]-2.0*A[0][-t4+t5][t4+t6]+A[0][-t4+t5-1][t4+t6])+0.125*(A[0][-t4+t5][t4+t6+1]-2.0*A[0][-t4+t5][t4+t6]+A[0][-t4+t5][t4+t6-1])+A[0][-t4+t5][t4+t6];;
                                    }
#pragma ivdep
#pragma vector always
                                    for (t6=ubv;t6<=lbv;t6++) {
                                        A[1][2*N-1-(-t4+t5)][t4+t6]=0.125*(A[0][2*N-1-(-t4+t5)+1][t4+t6]-2.0*A[0][2*N-1-(-t4+t5)][t4+t6]+A[0][2*N-1-(-t4+t5)-1][t4+t6])+0.125*(A[0][2*N-1-(-t4+t5)][t4+t6+1]-2.0*A[0][2*N-1-(-t4+t5)][t4+t6]+A[0][2*N-1-(-t4+t5)][t4+t6-1])+A[0][2*N-1-(-t4+t5)][t4+t6];;
                                    }
                                }
                            }else{

                                for (t5=max(max(32*t2,t4),-32*t1+32*t2+2*t4-31);t5<=min(min(32*t2+31,-32*t1+32*t2+2*t4),t4+N-1);t5++) {
                                    lbv=max(128*t3,t4);
                                    ubv=min(128*t3+127,t4+N-1);
#pragma ivdep
#pragma vector always
                                    for (t6=lbv;t6<=ubv;t6++) {
                                        A[0][-t4+t5][-t4+t6]=0.125*(A[1][-t4+t5+1][-t4+t6]-2.0*A[1][-t4+t5][-t4+t6]+A[1][-t4+t5-1][-t4+t6])+0.125*(A[1][-t4+t5][-t4+t6+1]-2.0*A[1][-t4+t5][-t4+t6]+A[1][-t4+t5][-t4+t6-1])+A[1][-t4+t5][-t4+t6];;
                                    }
#pragma ivdep
#pragma vector always
                                    for (t6=lbv;t6<=ubv;t6++) {
                                        A[0][2*N-1-(-t4+t5)][-t4+t6]=0.125*(A[1][2*N-1-(-t4+t5)+1][-t4+t6]-2.0*A[1][2*N-1-(-t4+t5)][-t4+t6]+A[1][2*N-1-(-t4+t5)-1][-t4+t6])+0.125*(A[1][2*N-1-(-t4+t5)][-t4+t6+1]-2.0*A[1][2*N-1-(-t4+t5)][-t4+t6]+A[1][2*N-1-(-t4+t5)][-t4+t6-1])+A[1][2*N-1-(-t4+t5)][-t4+t6];;
                                    }
                                    lbv=2*N-1-max(128*t3,t4);
                                    ubv=2*N-1-min(128*t3+127,t4+N-1);
#pragma ivdep
#pragma vector always
                                    for (t6=ubv;t6<=lbv;t6++) {
                                        A[0][-t4+t5][t4+t6]=0.125*(A[1][-t4+t5+1][t4+t6]-2.0*A[1][-t4+t5][t4+t6]+A[1][-t4+t5-1][t4+t6])+0.125*(A[1][-t4+t5][t4+t6+1]-2.0*A[1][-t4+t5][t4+t6]+A[1][-t4+t5][t4+t6-1])+A[1][-t4+t5][t4+t6];;
                                    }
#pragma ivdep
#pragma vector always
                                    for (t6=ubv;t6<=lbv;t6++) {
                                        A[0][2*N-1-(-t4+t5)][t4+t6]=0.125*(A[1][2*N-1-(-t4+t5)+1][t4+t6]-2.0*A[1][2*N-1-(-t4+t5)][t4+t6]+A[1][2*N-1-(-t4+t5)-1][t4+t6])+0.125*(A[1][2*N-1-(-t4+t5)][t4+t6+1]-2.0*A[1][2*N-1-(-t4+t5)][t4+t6]+A[1][2*N-1-(-t4+t5)][t4+t6-1])+A[1][2*N-1-(-t4+t5)][t4+t6];;
                                    }
                                }


                            }
                        }

                    }
                }
            }
        }
    }
    /* End of CLooG code */

#ifdef TIME
    gettimeofday(&end, 0);
#undef N
#define N 8000L
    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfms\t", tdiff * 1.0e3);
    printf("|MFLOPS =  %f\n", ((((double)NUM_FP_OPS * N *N *  T) / tdiff) / 1000000L));
#endif

#ifdef VERIFY
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            total+= A[T%2][i][j] ;
        }
    }
    printf("|sum: %e\t", total);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum_err_sqr += (A[T%2][i][j] - (total/N))*(A[T%2][i][j] - (total/N));
        }
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            chtotal += ((char *)A[T%2][i])[j];
        }
    }
    printf("|sum(rep(A)) = %d\n", chtotal);
#endif
    return 0;
}

// icc -O3 -fp-model precise heat_1d_np.c -o op-heat-1d-np -lm
// /* @ begin PrimeTile (num_tiling_levels=1; first_depth=1; last_depth=-1; boundary_tiling_level=-1;) @*/
// /* @ begin PrimeRegTile (scalar_replacement=0; T1t3=8; T1t4=8; ) @*/
// /* @ end @*/
