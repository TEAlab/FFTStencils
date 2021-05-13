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

// (i-2)<(0)?(N+(i-2)):(i-2)
// (i==0)?(N-1):(i-1)
// (i==N-1)?(0):(i+1)
// (i+2)>(N-1)?(i+3-N)?(i+2)

// (j-2)<(0)?(N+(j-2)):(j-2)
// (j==0)?(N-1):(j-1)
// (j==N-1)?(0):(j+1)
// (j+2)>(N-1)?(j+3-N)?(j+2)

//  #undef N
//  #define N 8000L
#undef T
#define T 50000
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
  int t1, t2, t3, t4, t5, t6;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((N >= 1) && (T >= 1)) {
  for (t1=0;t1<=T-1;t1++) {
    lbp=0;
    ubp=floord(N-1,2);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(N-1,256);t4++) {
        for (t5=2*t3;t5<=min(N-1,2*t3+1);t5++) {
          lbv=256*t4;
          ubv=min(N-1,256*t4+255);
#pragma ivdep
#pragma vector always
          for (t6=lbv;t6<=ubv;t6++) {
            A[1][t5][t6] = 0.04*( ((t5-2)<(0)?( (t6-2)<(0)?A[0][N+(t5-2)][N+(t6-2)]:A[0][N+(t5-2)][t6-2] +(t6==0)?A[0][N+(t5-2)][N-1]:A[0][N+(t5-2)][t6-1] +A[0][N+(t5-2)][t6] +(t6==N-1)?A[0][N+(t5-2)][0]:A[0][N+(t5-2)][t6+1] +(t6+2)>(N-1)?A[0][N+(t5-2)][t6+3-N]:A[0][N+(t5-2)][t6+2] ):( (t6-2)<(0)?A[0][t5-2][N+(t6-2)]:A[0][t5-2][t6-2] +(t6==0)?A[0][t5-2][N-1]:A[0][t5-2][t6-1] +A[0][t5-2][t6] +(t6==N-1)?A[0][t5-2][0]:A[0][t5-2][t6+1] +(t6+2)>(N-1)?A[0][t5-2][t6+3-N]:A[0][t5-2][t6+2])) +((t5==0)?( (t6-2)<(0)?A[0][(N-1)][(N+(t6-2))]:A[0][(N-1)][(t6-2)] +(t6==0)?A[0][(N-1)][(N-1)]:A[0][(N-1)][(t6-1)] +A[0][(N-1)][t6] +(t6==N-1)?A[0][(N-1)][(0)]:A[0][(N-1)][(t6+1)] +(t6+2)>(N-1)?A[0][(N-1)][(t6+3-N)]:A[0][(N-1)][(t6+2)] ):( (t6-2)<(0)?A[0][(t5-1)][(N+(t6-2))]:A[0][(t5-1)][(t6-2)] +(t6==0)?A[0][(t5-1)][(N-1)]:A[0][(t5-1)][(t6-1)] +A[0][(t5-1)][t6] +(t6==N-1)?A[0][(t5-1)][(0)]:A[0][(t5-1)][(t6+1)] +(t6+2)>(N-1)?A[0][(t5-1)][(t6+3-N)]:A[0][(t5-1)][(t6+2)])) +( (t6-2)<(0)?A[0][t5][(N+(t6-2))]:A[0][t5][(t6-2)] +(t6==0)?A[0][t5][(N-1)]:A[0][t5][(t6-1)] +A[0][t5][t6] +(t6==N-1)?A[0][t5][(0)]:A[0][t5][(t6+1)] +(t6+2)>(N-1)?A[0][t5][(t6+3-N)]:A[0][t5][(t6+2)]) +((t5==N-1)?( (t6-2)<(0)?A[0][(0)][(N+(t6-2))]:A[0][(0)][(t6-2)] +(t6==0)?A[0][(0)][(N-1)]:A[0][(0)][(t6-1)] +A[0][(0)][t6] +(t6==N-1)?A[0][(0)][(0)]:A[0][(0)][(t6+1)] +(t6+2)>(N-1)?A[0][(0)][(t6+3-N)]:A[0][(0)][(t6+2)] ):( (t6-2)<(0)?A[0][(t5+1)][(N+(t6-2))]:A[0][(t5+1)][(t6-2)] +(t6==0)?A[0][(t5+1)][(N-1)]:A[0][(t5+1)][(t6-1)] +A[0][(t5+1)][t6] +(t6==N-1)?A[0][(t5+1)][(0)]:A[0][(t5+1)][(t6+1)] +(t6+2)>(N-1)?A[0][(t5+1)][(t6+3-N)]:A[0][(t5+1)][(t6+2)])) +((t5+2)>(N-1)?( (t6-2)<(0)?A[0][(t5+3-N)][(N+(t6-2))]:A[0][(t5+3-N)][(t6-2)] +(t6==0)?A[0][(t5+3-N)][(N-1)]:A[0][(t5+3-N)][(N-1)] +A[0][(t5+3-N)][t6] +(t6==N-1)?A[0][(t5+3-N)][(0)]:A[0][(t5+3-N)][(0)] +(t6+2)>(N-1)?A[0][(t5+3-N)][(t6+3-N)]:A[0][(t5+3-N)][(t6+3-N)] ):( +(t6-2)<(0)?A[0][(t5+2)][(N+(t6-2))]:A[0][(t5+2)][(t6-2)] +(t6==0)?A[0][(t5+2)][(N-1)]:A[0][(t5+2)][(t6-1)] +A[0][(t5+2)][t6] +(t6==N-1)?A[0][(t5+2)][(0)]:A[0][(t5+2)][(t6+1)] +(t6+2)>(N-1)?A[0][(t5+2)][(t6+3-N)]:A[0][(t5+2)][(t6+2)])) );;
          }
        }
      }
    }
    lbp=1;
    ubp=floord(N-3,2);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(N-3,256);t4++) {
        for (t5=2*t3;t5<=min(N-3,2*t3+1);t5++) {
          lbv=max(2,256*t4);
          ubv=min(N-3,256*t4+255);
#pragma ivdep
#pragma vector always
          for (t6=lbv;t6<=ubv;t6++) {
            A[0][t5][t6] = 0.04*( ((t5-2)<(0)?( (t6-2)<(0)?A[1][N+(t5-2)][N+(t6-2)]:A[1][N+(t5-2)][t6-2] +(t6==0)?A[1][N+(t5-2)][N-1]:A[1][N+(t5-2)][t6-1] +A[1][N+(t5-2)][t6] +(t6==N-1)?A[1][N+(t5-2)][0]:A[1][N+(t5-2)][t6+1] +(t6+2)>(N-1)?A[1][N+(t5-2)][t6+3-N]:A[1][N+(t5-2)][t6+2] ):( (t6-2)<(0)?A[1][t5-2][N+(t6-2)]:A[1][t5-2][t6-2] +(t6==0)?A[1][t5-2][N-1]:A[1][t5-2][t6-1] +A[1][t5-2][t6] +(t6==N-1)?A[1][t5-2][0]:A[1][t5-2][t6+1] +(t6+2)>(N-1)?A[1][t5-2][t6+3-N]:A[1][t5-2][t6+2])) +((t5==0)?( (t6-2)<(0)?A[1][(N-1)][(N+(t6-2))]:A[1][(N-1)][(t6-2)] +(t6==0)?A[1][(N-1)][(N-1)]:A[1][(N-1)][(t6-1)] +A[1][(N-1)][t6] +(t6==N-1)?A[1][(N-1)][(0)]:A[1][(N-1)][(t6+1)] +(t6+2)>(N-1)?A[1][(N-1)][(t6+3-N)]:A[1][(N-1)][(t6+2)] ):( (t6-2)<(0)?A[1][(t5-1)][(N+(t6-2))]:A[1][(t5-1)][(t6-2)] +(t6==0)?A[1][(t5-1)][(N-1)]:A[1][(t5-1)][(t6-1)] +A[1][(t5-1)][t6] +(t6==N-1)?A[1][(t5-1)][(0)]:A[1][(t5-1)][(t6+1)] +(t6+2)>(N-1)?A[1][(t5-1)][(t6+3-N)]:A[1][(t5-1)][(t6+2)])) +( (t6-2)<(0)?A[1][t5][(N+(t6-2))]:A[1][t5][(t6-2)] +(t6==0)?A[1][t5][(N-1)]:A[1][t5][(t6-1)] +A[1][t5][t6] +(t6==N-1)?A[1][t5][(0)]:A[1][t5][(t6+1)] +(t6+2)>(N-1)?A[1][t5][(t6+3-N)]:A[1][t5][(t6+2)]) +((t5==N-1)?( (t6-2)<(0)?A[1][(0)][(N+(t6-2))]:A[1][(0)][(t6-2)] +(t6==0)?A[1][(0)][(N-1)]:A[1][(0)][(t6-1)] +A[1][(0)][t6] +(t6==N-1)?A[1][(0)][(0)]:A[1][(0)][(t6+1)] +(t6+2)>(N-1)?A[1][(0)][(t6+3-N)]:A[1][(0)][(t6+2)] ):( (t6-2)<(0)?A[1][(t5+1)][(N+(t6-2))]:A[1][(t5+1)][(t6-2)] +(t6==0)?A[1][(t5+1)][(N-1)]:A[1][(t5+1)][(t6-1)] +A[1][(t5+1)][t6] +(t6==N-1)?A[1][(t5+1)][(0)]:A[1][(t5+1)][(t6+1)] +(t6+2)>(N-1)?A[1][(t5+1)][(t6+3-N)]:A[1][(t5+1)][(t6+2)])) +((t5+2)>(N-1)?( (t6-2)<(0)?A[1][(t5+3-N)][(N+(t6-2))]:A[1][(t5+3-N)][(t6-2)] +(t6==0)?A[1][(t5+3-N)][(N-1)]:A[1][(t5+3-N)][(N-1)] +A[1][(t5+3-N)][t6] +(t6==N-1)?A[1][(t5+3-N)][(0)]:A[1][(t5+3-N)][(0)] +(t6+2)>(N-1)?A[1][(t5+3-N)][(t6+3-N)]:A[1][(t5+3-N)][(t6+3-N)] ):( +(t6-2)<(0)?A[1][(t5+2)][(N+(t6-2))]:A[1][(t5+2)][(t6-2)] +(t6==0)?A[1][(t5+2)][(N-1)]:A[1][(t5+2)][(t6-1)] +A[1][(t5+2)][t6] +(t6==N-1)?A[1][(t5+2)][(0)]:A[1][(t5+2)][(t6+1)] +(t6+2)>(N-1)?A[1][(t5+2)][(t6+3-N)]:A[1][(t5+2)][(t6+2)])) );;
          }
        }
      }
    }
  }
}
/* End of CLooG code */
#undef T
#define T 100000
// #undef N
// #define N 16000L
#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfs\n", tdiff );
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
