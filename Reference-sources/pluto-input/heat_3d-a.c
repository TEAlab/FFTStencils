/*
 * Discretized 3D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 *
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
#define N 800L
#define T 800L
#endif

#define NUM_FP_OPS 15

/* Define our arrays */

double total=0; double sum_err_sqr=0;
int chtotal=0;
/* Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 *
 * Return 1 if the difference is negative, otherwise 0.
 */
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    /* Perform the carry for the later subtraction by updating y. */
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

    /* Compute the time remaining to wait.
     * tv_usec is certainly positive.
     */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}


int main(int argc, char * argv[]) {
    long int t, i, j, k;
    const int BASE = 1024;
    long count=0;
    // for timekeeping
    int ts_return = -1;
    struct timeval start, end, result;
    double tdiff = 0.0;
    
    // double A[2][N][N][N];
    double ****A = (double ****)malloc(2 * sizeof (double ***));
    int l;
    for (l = 0; l < 2; l++){
        A[l] = (double ***) malloc(N * sizeof(double **));
        for (i = 0; i < N; i++){
            A[l][i] = (double **) malloc(N * sizeof(double *));
            for (j = 0; j < N; j++)
                A[l][i][j] = (double *) malloc(N * sizeof (double));
        }
    }

    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                A[0][i][j][k] = 1.0 * (rand() % BASE);
            }
        }
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif

// #undef N
// #define N 150L
#undef T
#define T 400L
#pragma scop
    for (t = 0; t < T; t++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    A[1][i][j][k] = 0.125 * (((i+1) >= N ? 0 : A[0][i+1][j][k]) - 2.0 * A[0][i][j][k] + ((i-1) < 0 ? 0 : A[0][i-1][j][k]))
                                + 0.125 * (((j+1) >= N ? 0 : A[0][i][j+1][k]) - 2.0 * A[0][i][j][k] + ((j-1) < 0 ? 0 : A[0][i][j-1][k]))
                                + 0.125 * (((k+1) >= N ? 0 : A[0][i][j][k+1]) - 2.0 * A[0][i][j][k] + ((k-1) < 0 ? 0 : A[0][i][j][k-1]))
                                + A[0][i][j][k];
                    // A[(t+1)%2][i][j][k] =   0.125 * (A[t%2][i+1][j][k] - 2.0 * A[t%2][i][j][k] + A[t%2][(i==0)?(2*N-1):(i-1)][j][k]) + 0.125 * (A[t%2][i][j+1][k] - 2.0 * A[t%2][i][j][k] + A[t%2][i][(j==0)?(2*N-1):(j-1)][k]) + 0.125 * (A[t%2][i][j][(k==0)?(2*N-1):(k-1)] - 2.0 * A[t%2][i][j][k] + A[t%2][i][j][k+1]) + A[t%2][i][j][k];
                    // A[(t+1)%2][2*N-1-(i)][j][k] =   0.125 * (A[t%2][(i==0)?(0):(2*N-1-(i)+1)][j][k] - 2.0 * A[t%2][2*N-1-(i)][j][k] + A[t%2][2*N-1-(i)-1][j][k]) + 0.125 * (A[t%2][2*N-1-(i)][j+1][k] - 2.0 * A[t%2][2*N-1-(i)][j][k] + A[t%2][2*N-1-(i)][(j==0)?(2*N-1):(j-1)][k]) + 0.125 * (A[t%2][2*N-1-(i)][j][(k==0)?(2*N-1):(k-1)] - 2.0 * A[t%2][2*N-1-(i)][j][k] + A[t%2][2*N-1-(i)][j][k+1]) + A[t%2][2*N-1-(i)][j][k];
                    // A[(t+1)%2][i][2*N-1-(j)][k] =   0.125 * (A[t%2][i+1][2*N-1-(j)][k] - 2.0 * A[t%2][i][2*N-1-(j)][k] + A[t%2][(i==0)?(2*N-1):(i-1)][2*N-1-(j)][k]) + 0.125 * (A[t%2][i][(j==0)?(0):(2*N-1-(j)+1)][k] - 2.0 * A[t%2][i][2*N-1-(j)][k] + A[t%2][i][2*N-1-(j)-1][k]) + 0.125 * (A[t%2][i][2*N-1-(j)][(k==0)?(2*N-1):(k-1)] - 2.0 * A[t%2][i][2*N-1-(j)][k] + A[t%2][i][2*N-1-(j)][k+1]) + A[t%2][i][2*N-1-(j)][k];
                    // A[(t+1)%2][2*N-1-(i)][2*N-1-(j)][k] =   0.125 * (A[t%2][(i==0)?(0):(2*N-1-(i)+1)][2*N-1-(j)][k] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][k] + A[t%2][2*N-1-(i)-1][2*N-1-(j)][k]) + 0.125 * (A[t%2][2*N-1-(i)][(j==0)?(0):(2*N-1-(j)+1)][k] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][k] + A[t%2][2*N-1-(i)][2*N-1-(j)-1][k]) + 0.125 * (A[t%2][2*N-1-(i)][2*N-1-(j)][(k==0)?(2*N-1):(k-1)] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][k] + A[t%2][2*N-1-(i)][2*N-1-(j)][k+1]) + A[t%2][2*N-1-(i)][2*N-1-(j)][k];
                    // A[(t+1)%2][i][j][2*N-1-(k)] =   0.125 * (A[t%2][i+1][j][2*N-1-(k)] - 2.0 * A[t%2][i][j][2*N-1-(k)] + A[t%2][(i==0)?(2*N-1):(i-1)][j][2*N-1-(k)]) + 0.125 * (A[t%2][i][j+1][2*N-1-(k)] - 2.0 * A[t%2][i][j][2*N-1-(k)] + A[t%2][i][(j==0)?(2*N-1):(j-1)][2*N-1-(k)]) + 0.125 * (A[t%2][i][j][2*N-1-(k)-1] - 2.0 * A[t%2][i][j][2*N-1-(k)] + A[t%2][i][j][(k==0)?(0):(2*N-1-(k)+1)]) + A[t%2][i][j][2*N-1-(k)];
                    // A[(t+1)%2][2*N-1-(i)][j][2*N-1-(k)] =   0.125 * (A[t%2][(i==0)?(0):(2*N-1-(i)+1)][j][2*N-1-(k)] - 2.0 * A[t%2][2*N-1-(i)][j][2*N-1-(k)] + A[t%2][2*N-1-(i)-1][j][2*N-1-(k)]) + 0.125 * (A[t%2][2*N-1-(i)][j+1][2*N-1-(k)] - 2.0 * A[t%2][2*N-1-(i)][j][2*N-1-(k)] + A[t%2][2*N-1-(i)][(j==0)?(2*N-1):(j-1)][2*N-1-(k)]) + 0.125 * (A[t%2][2*N-1-(i)][j][2*N-1-(k)-1] - 2.0 * A[t%2][2*N-1-(i)][j][2*N-1-(k)] + A[t%2][2*N-1-(i)][j][(k==0)?(0):(2*N-1-(k)+1)]) + A[t%2][2*N-1-(i)][j][2*N-1-(k)];
                    // A[(t+1)%2][i][2*N-1-(j)][2*N-1-(k)] =   0.125 * (A[t%2][i+1][2*N-1-(j)][2*N-1-(k)] - 2.0 * A[t%2][i][2*N-1-(j)][2*N-1-(k)] + A[t%2][(i==0)?(2*N-1):(i-1)][2*N-1-(j)][2*N-1-(k)]) + 0.125 * (A[t%2][i][(j==0)?(0):(2*N-1-(j)+1)][2*N-1-(k)] - 2.0 * A[t%2][i][2*N-1-(j)][2*N-1-(k)] + A[t%2][i][2*N-1-(j)-1][2*N-1-(k)]) + 0.125 * (A[t%2][i][2*N-1-(j)][2*N-1-(k)-1] - 2.0 * A[t%2][i][2*N-1-(j)][2*N-1-(k)] + A[t%2][i][2*N-1-(j)][(k==0)?(0):(2*N-1-(k)+1)]) + A[t%2][i][2*N-1-(j)][2*N-1-(k)];
                    // A[(t+1)%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)] =   0.125 * (A[t%2][(i==0)?(0):(2*N-1-(i)+1)][2*N-1-(j)][2*N-1-(k)] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)] + A[t%2][2*N-1-(i)-1][2*N-1-(j)][2*N-1-(k)]) + 0.125 * (A[t%2][2*N-1-(i)][(j==0)?(0):(2*N-1-(j)+1)][2*N-1-(k)] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)] + A[t%2][2*N-1-(i)][2*N-1-(j)-1][2*N-1-(k)]) + 0.125 * (A[t%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)-1] - 2.0 * A[t%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)] + A[t%2][2*N-1-(i)][2*N-1-(j)][(k==0)?(0):(2*N-1-(k)+1)]) + A[t%2][2*N-1-(i)][2*N-1-(j)][2*N-1-(k)];
                }
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    A[0][i][j][k] = 0.125 * (((i+1) >= N ? 0 : A[1][i+1][j][k]) - 2.0 * A[1][i][j][k] + ((i-1) < 0 ? 0 : A[1][i-1][j][k]))
                                + 0.125 * (((j+1) >= N ? 0 : A[1][i][j+1][k]) - 2.0 * A[1][i][j][k] + ((j-1) < 0 ? 0 : A[1][i][j-1][k]))
                                + 0.125 * (((k+1) >= N ? 0 : A[1][i][j][k+1]) - 2.0 * A[1][i][j][k] + ((k-1) < 0 ? 0 : A[1][i][j][k-1]))
                                + A[1][i][j][k];
                }
            }
        }

    }
#pragma endscop

// #undef N
// #define N 300L
#undef T
#define T 800L
#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken: %7.5lfs\t", tdiff);
    printf("|MFLOPS: %f\n", ((((double)NUM_FP_OPS * N *N * N * (T-1)) / tdiff) / 1000000L));
#endif

#ifdef VERIFY
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                total+= A[T%2][i][j][k] ;
            }
        }
    }
    printf("|sum: %e\t", total);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                sum_err_sqr += (A[T%2][i][j][k] - (total/N))*(A[T%2][i][j][k] - (total/N));
            }
        }
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                chtotal += ((char *)A[T%2][i][j])[k];
            }
        }
    }
    printf("|sum(rep(A)) = %d\n", chtotal);
#endif

    for (l = 0; l < 2; l++){
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++)
                free(A[l][i][j]); //  = (double *) malloc(N * sizeof (double));
            free(A[l][i]); // = (double **) malloc(N * sizeof(double *));
        }   
        free(A[l]); // = (double ***) malloc(N * sizeof(double **));
    }

    return 0;
}

// icc -O3 -fp-model precise heat_1d_np.c -o op-heat-1d-np -lm
// /* @ begin PrimeTile (num_tiling_levels=1; first_depth=1; last_depth=-1; boundary_tiling_level=-1;) @*/
// /* @ begin PrimeRegTile (scalar_replacement=0; T1t5=4; T1t6=4; T1t7=4; T1t8=4; ) @*/
// /* @ end @*/
