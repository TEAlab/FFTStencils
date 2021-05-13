#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

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
#define T 1L
#endif

#define NUM_FP_OPS 15

/* Define our arrays */

// double A[2][N][N][N];
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
                A[1][i][j][k] = 0.0;
            }
        }
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif

#undef N
#define N 400L

// (i==0)?(N-1):(i-1)
// (i==N-1)?(0):(i+1)

// (j==0)?(N-1):(j-1)
// (j==N-1)?(0):(j+1)

// (k==0)?(N-1):(k-1)
// (k==N-1)?(0):(k+1)

#undef T
#define T 1L
/* Copyright (C) 1991-2012 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* We do support the IEC 559 math functionality, real and complex.  */
/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7, t8;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((N >= 1) && (T >= 1)) {
  for (t1=0;t1<=T-1;t1++) {
    lbp=0;
    ubp=N-1;
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7,t8)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=N-1;t4++) {
        for (t5=0;t5<=floord(N-1,512);t5++) {
          lbv=512*t5;
          ubv=min(N-1,512*t5+511);
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            A[1][t3][t4][t8] = 2.666*A[0][t3][t4][t8] - ( 0.166* (t3==0?A[0][N-1][t4][t8]:A[0][t3-1][t4][t8]) + 0.166*(t3==N-1?A[0][0][t4][t8]:A[0][t3+1][t4][t8]) + 0.166*(t4==0?A[0][t3][N-1][t8]:A[0][t3][t4-1][t8]) + 0.166*(t4==N-1?A[0][t3][0][t8]:A[0][t3][t4+1][t8]) + 0.166*(t8==N-1?A[0][t3][t4][0]:A[0][t3][t4][t8+1]) + 0.166*(t8==0?A[0][t3][t4][N-1]:A[0][t3][t4][t8-1])) - ( 0.0833*(t3==0?( t4==0?A[0][N-1][N-1][t8]:A[0][N-1][t4-1][t8]):( t4==0?A[0][t3-1][N-1][t8]:A[0][t3-1][t4-1][t8])) + 0.0833*(t3==N-1?( t4==0?A[0][0][N-1][t8]:A[0][0][t4-1][t8]):( t4==0?A[0][t3+1][N-1][t8]:A[0][t3+1][t4-1][t8])) + 0.0833*(t3==0?( t4==N-1?A[0][N-1][0][t8]:A[0][N-1][t4+1][t8]):( t4==N-1?A[0][t3-1][0][t8]:A[0][t3-1][t4+1][t8])) + 0.0833*(t3==N-1?( t4==N-1?A[0][0][0][t8]:A[0][0][t4+1][t8]):( t4==N-1?A[0][t3+1][0][t8]:A[0][t3+1][t4+1][t8])) + 0.0833*(t3==0?( t8==0?A[0][N-1][t4][N-1]:A[0][N-1][t4][t8-1]):( t8==0?A[0][t3-1][t4][N-1]:A[0][t3-1][t4][t8-1])) + 0.0833*(t3==N-1?( t8==0?A[0][0][t4][N-1]:A[0][0][t4][t8-1]):( t8==0?A[0][t3+1][t4][N-1]:A[0][t3+1][t4][t8-1])) + 0.0833*(t4==0?( t8==0?A[0][t3][N-1][N-1]:A[0][t3][N-1][t8-1]):( t8==0?A[0][t3][t4-1][N-1]:A[0][t3][t4-1][t8-1])) + 0.0833*(t4==N-1?( t8==0?A[0][t3][0][N-1]:A[0][t3][0][t8-1]):( t8==0?A[0][t3][t4+1][N-1]:A[0][t3][t4+1][t8-1])) + 0.0833*(t3==0?( t8==N-1?A[0][N-1][t4][0]:A[0][N-1][t4][t8+1]):( t8==N-1?A[0][t3-1][t4][0]:A[0][t3-1][t4][t8+1])) + 0.0833*(t3==N-1?( t8==N-1?A[0][0][t4][0]:A[0][0][t4][t8+1]):( t8==N-1?A[0][t3+1][t4][0]:A[0][t3+1][t4][t8+1])) + 0.0833*(t4==0?( t8==N-1?A[0][t3][N-1][0]:A[0][t3][N-1][t8+1]):( t8==N-1?A[0][t3][t4-1][0]:A[0][t3][t4-1][t8+1])) + 0.0833*(t4==N-1?( t8==N-1?A[0][t3][0][0]:A[0][t3][0][t8+1]):( t8==N-1?A[0][t3][t4+1][0]:A[0][t3][t4+1][t8+1])));;
          }
        }
      }
    }
    lbp=0;
    ubp=N-1;
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7,t8)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=N-1;t4++) {
        for (t5=0;t5<=floord(N-1,512);t5++) {
          lbv=512*t5;
          ubv=min(N-1,512*t5+511);
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            A[0][t3][t4][t8] = 2.666*A[1][t3][t4][t8] - ( 0.166* (t3==0?A[1][N-1][t4][t8]:A[1][t3-1][t4][t8]) + 0.166*(t3==N-1?A[1][0][t4][t8]:A[1][t3+1][t4][t8]) + 0.166*(t4==0?A[1][t3][N-1][t8]:A[1][t3][t4-1][t8]) + 0.166*(t4==N-1?A[1][t3][0][t8]:A[1][t3][t4+1][t8]) + 0.166*(t8==N-1?A[1][t3][t4][0]:A[1][t3][t4][t8+1]) + 0.166*(t8==0?A[1][t3][t4][N-1]:A[1][t3][t4][t8-1])) - ( 0.0833*(t3==0?( t4==0?A[1][N-1][N-1][t8]:A[1][N-1][t4-1][t8]):( t4==0?A[1][t3-1][N-1][t8]:A[1][t3-1][t4-1][t8])) + 0.0833*(t3==N-1?( t4==0?A[1][0][N-1][t8]:A[1][0][t4-1][t8]):( t4==0?A[1][t3+1][N-1][t8]:A[1][t3+1][t4-1][t8])) + 0.0833*(t3==0?( t4==N-1?A[1][N-1][0][t8]:A[1][N-1][t4+1][t8]):( t4==N-1?A[1][t3-1][0][t8]:A[1][t3-1][t4+1][t8])) + 0.0833*(t3==N-1?( t4==N-1?A[1][0][0][t8]:A[1][0][t4+1][t8]):( t4==N-1?A[1][t3+1][0][t8]:A[1][t3+1][t4+1][t8])) + 0.0833*(t3==0?( t8==0?A[1][N-1][t4][N-1]:A[1][N-1][t4][t8-1]):( t8==0?A[1][t3-1][t4][N-1]:A[1][t3-1][t4][t8-1])) + 0.0833*(t3==N-1?( t8==0?A[1][0][t4][N-1]:A[1][0][t4][t8-1]):( t8==0?A[1][t3+1][t4][N-1]:A[1][t3+1][t4][t8-1])) + 0.0833*(t4==0?( t8==0?A[1][t3][N-1][N-1]:A[1][t3][N-1][t8-1]):( t8==0?A[1][t3][t4-1][N-1]:A[1][t3][t4-1][t8-1])) + 0.0833*(t4==N-1?( t8==0?A[1][t3][0][N-1]:A[1][t3][0][t8-1]):( t8==0?A[1][t3][t4+1][N-1]:A[1][t3][t4+1][t8-1])) + 0.0833*(t3==0?( t8==N-1?A[1][N-1][t4][0]:A[1][N-1][t4][t8+1]):( t8==N-1?A[1][t3-1][t4][0]:A[1][t3-1][t4][t8+1])) + 0.0833*(t3==N-1?( t8==N-1?A[1][0][t4][0]:A[1][0][t4][t8+1]):( t8==N-1?A[1][t3+1][t4][0]:A[1][t3+1][t4][t8+1])) + 0.0833*(t4==0?( t8==N-1?A[1][t3][N-1][0]:A[1][t3][N-1][t8+1]):( t8==N-1?A[1][t3][t4-1][0]:A[1][t3][t4-1][t8+1])) + 0.0833*(t4==N-1?( t8==N-1?A[1][t3][0][0]:A[1][t3][0][t8+1]):( t8==N-1?A[1][t3][t4+1][0]:A[1][t3][t4+1][t8+1])));;
          }
        }
      }
    }
  }
}
/* End of CLooG code */

#undef N
#define N 800L
#undef T
#define T 1L
#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken: %7.5lfms\t", tdiff * 1.0e3);
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
