#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*
 * Discretized 1D heat equation stencil with non periodic boundary conditions
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
#define N 10000L
#define T 10000L
#endif

#define NUM_FP_OPS 4

/* Define our arrays */
double A[2][N];
double total=0; double sum_err_sqr=0;
long int chtotal=0;
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
long count=0;
    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results

    for (i = 0; i < N; i++) {
        A[0][i] = 1.0 * (rand() % BASE);
    }

#ifdef TIME
    gettimeofday(&start, 0);
#endif
#undef T
#define T 5000
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
  int t1, t2, t3, t4, t5;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((N >= 1) && (T >= 1)) {
  for (t1=0;t1<=floord(258*T+N-258,256);t1++) {
    lbp=max(ceild(t1-127,129),t1-T+1);
    ubp=min(floord(2*t1+N,258),t1);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5)
    for (t2=lbp;t2<=ubp;t2++) {
      if ((N >= 2) && (t1 <= floord(258*t2-N+255,2)) && (t1 >= 129*t2)) {
        A[1][0] = 0.125 * ((0 == N-1)?0:A[0][0 +1] - 2.0*A[0][0] + (0 == 0)?0:A[0][0 -1]);;
        for (t4=2*t1-2*t2+1;t4<=2*t1-2*t2+N-1;t4++) {
          A[1][(-2*t1+2*t2+t4)] = 0.125 * (((-2*t1+2*t2+t4) == N-1)?0:A[0][(-2*t1+2*t2+t4)+1] - 2.0*A[0][(-2*t1+2*t2+t4)] + ((-2*t1+2*t2+t4) == 0)?0:A[0][(-2*t1+2*t2+t4)-1]);;
          A[0][(-2*t1+2*t2+t4-1)] = 0.125 * (((-2*t1+2*t2+t4-1) == N-1)?0:A[1][(-2*t1+2*t2+t4-1)+1] - 2.0*A[1][(-2*t1+2*t2+t4-1)] + ((-2*t1+2*t2+t4-1) == 0)?0:A[1][(-2*t1+2*t2+t4-1)-1]);;
        }
        A[0][(N-1)] = 0.125 * (((N-1) == N-1)?0:A[1][(N-1)+1] - 2.0*A[1][(N-1)] + ((N-1) == 0)?0:A[1][(N-1)-1]);;
      }
      if (t1 >= max(ceild(258*t2-N+256,2),129*t2)) {
        A[1][0] = 0.125 * ((0 == N-1)?0:A[0][0 +1] - 2.0*A[0][0] + (0 == 0)?0:A[0][0 -1]);;
        for (t4=2*t1-2*t2+1;t4<=256*t2+255;t4++) {
          A[1][(-2*t1+2*t2+t4)] = 0.125 * (((-2*t1+2*t2+t4) == N-1)?0:A[0][(-2*t1+2*t2+t4)+1] - 2.0*A[0][(-2*t1+2*t2+t4)] + ((-2*t1+2*t2+t4) == 0)?0:A[0][(-2*t1+2*t2+t4)-1]);;
          A[0][(-2*t1+2*t2+t4-1)] = 0.125 * (((-2*t1+2*t2+t4-1) == N-1)?0:A[1][(-2*t1+2*t2+t4-1)+1] - 2.0*A[1][(-2*t1+2*t2+t4-1)] + ((-2*t1+2*t2+t4-1) == 0)?0:A[1][(-2*t1+2*t2+t4-1)-1]);;
        }
      }
      if (N == 1) {
        A[1][0] = 0.125 * ((0 == N-1)?0:A[0][0 +1] - 2.0*A[0][0] + (0 == 0)?0:A[0][0 -1]);;
        A[0][0] = 0.125 * ((0 == N-1)?0:A[1][0 +1] - 2.0*A[1][0] + (0 == 0)?0:A[1][0 -1]);;
      }
      if ((t1 <= min(floord(258*t2-N+255,2),129*t2-1)) && (t1 >= ceild(258*t2-N+1,2))) {
        for (t4=256*t2;t4<=2*t1-2*t2+N-1;t4++) {
          A[1][(-2*t1+2*t2+t4)] = 0.125 * (((-2*t1+2*t2+t4) == N-1)?0:A[0][(-2*t1+2*t2+t4)+1] - 2.0*A[0][(-2*t1+2*t2+t4)] + ((-2*t1+2*t2+t4) == 0)?0:A[0][(-2*t1+2*t2+t4)-1]);;
          A[0][(-2*t1+2*t2+t4-1)] = 0.125 * (((-2*t1+2*t2+t4-1) == N-1)?0:A[1][(-2*t1+2*t2+t4-1)+1] - 2.0*A[1][(-2*t1+2*t2+t4-1)] + ((-2*t1+2*t2+t4-1) == 0)?0:A[1][(-2*t1+2*t2+t4-1)-1]);;
        }
        A[0][(N-1)] = 0.125 * (((N-1) == N-1)?0:A[1][(N-1)+1] - 2.0*A[1][(N-1)] + ((N-1) == 0)?0:A[1][(N-1)-1]);;
      }
      if ((t1 >= ceild(258*t2-N+256,2)) && (t1 <= 129*t2-1)) {
        for (t4=256*t2;t4<=256*t2+255;t4++) {
          A[1][(-2*t1+2*t2+t4)] = 0.125 * (((-2*t1+2*t2+t4) == N-1)?0:A[0][(-2*t1+2*t2+t4)+1] - 2.0*A[0][(-2*t1+2*t2+t4)] + ((-2*t1+2*t2+t4) == 0)?0:A[0][(-2*t1+2*t2+t4)-1]);;
          A[0][(-2*t1+2*t2+t4-1)] = 0.125 * (((-2*t1+2*t2+t4-1) == N-1)?0:A[1][(-2*t1+2*t2+t4-1)+1] - 2.0*A[1][(-2*t1+2*t2+t4-1)] + ((-2*t1+2*t2+t4-1) == 0)?0:A[1][(-2*t1+2*t2+t4-1)-1]);;
        }
      }
      if (2*t1 == 258*t2-N) {
        if ((256*t1+257*N)%258 == 0) {
          A[0][(N-1)] = 0.125 * (((N-1) == N-1)?0:A[1][(N-1)+1] - 2.0*A[1][(N-1)] + ((N-1) == 0)?0:A[1][(N-1)-1]);;
        }
      }
    }
  }
}
/* End of CLooG code */
#undef T
#define T 10000

#ifdef TIME
    gettimeofday(&end, 0);

    ts_return = timeval_subtract(&result, &end, &start);
    tdiff = (double)(result.tv_sec + result.tv_usec * 1.0e-6);

    printf("|Time taken =  %7.5lfs\t", tdiff);
    printf("|MFLOPS =  %f\n", ((((double)NUM_FP_OPS * N *  T) / tdiff) / 1000000L));
#endif

#ifdef VERIFY
    total=0;
    for (i = 0; i < N; i++) {
        total+= A[T%2][i] ;
    }
    printf("|sum: %e\t", total);
    for (i = 0; i < N; i++) {
        sum_err_sqr += (A[T%2][i] - (total/N))*(A[T%2][i] - (total/N));
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 0; i < N; i++) {
        chtotal += ((char *)A[T%2])[i];
    }
    printf("|sum(rep(A)) = %ld\n", chtotal);
#endif
    return 0;
}

// icc -O3 -fp-model precise heat_1d_np.c -o op-heat-1d-np -lm
// /* @ begin PrimeTile (num_tiling_levels=1; first_depth=1; last_depth=-1; boundary_tiling_level=-1;) @*/
// /* @ begin PrimeRegTile (scalar_replacement=0; T1t3=8; T1t4=8; ) @*/
// /* @ end @*/
// ,t2,t3,t4,t5,t6)
