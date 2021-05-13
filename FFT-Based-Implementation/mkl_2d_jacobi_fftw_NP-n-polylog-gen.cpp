/**
 * For compiling --> 
        [NOT USING THIS] export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
        export OMP_NUM_THREADS=68
        set MKL_NUM_THREADS = 64
        icc -std=gnu++98 -O3 -qopenmp -xhost -ansi-alias -ipo -qopt-prefetch=8 -AVX512 mkl_2d_jacobi_fftw_NP-n-polylog-gen.cpp -o mkl_2d_jacobi_fftw_NP-n-polylog-gen -lm -mkl

 * For running -->
 * ./mkl_2d_jacobi_fftw_NP-n-polylog-gen 8000 8000 8000 32 32768 68 0
 * Success: memory allocated!
 * N1 = 8000, N2 = 8000, T = 8000, T_BASE = 32, V_BASE = 32768, numThreads = 68, time = 20.712
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
#include <tr1/unordered_map>

#include <omp.h>

#ifdef USE_PAPI
    #include <papi.h>    
    #include "papilib.h"
#endif

#ifdef POLYBENCH
    #include <polybench.h>
#endif

using namespace std;

#ifndef MAX_n
   #define MAX_n    20
#endif   

#define MAX_N       ( ( 1 << MAX_n ) + 100 )

#ifndef MAX_VAL
   #define MAX_VAL  1024
#endif   

#ifndef STENCIL_SPACE_FACTOR
   #define STENCIL_SPACE_FACTOR  10
#endif   

#ifndef MIN_NT_RATIO
   #define MIN_NT_RATIO 16
#endif


int N_THREADS;

int N1_orig, N2_orig, T_orig, T_base, V_base;

double *input_real;
double *input_real_orig;
double *stencil_real;
double *output_real;
double *tmp_real;

double complex *stencil_complex;
double complex *input_complex;

int next_stencil_index = 0, max_stencil_index;

std::tr1::unordered_map< std::string, int > stencil_map;
std::tr1::unordered_map< long int, DFTI_DESCRIPTOR_HANDLE > desc_map;


long getTime( void )
{
    struct timeval tp;

    gettimeofday( &tp, NULL );
    
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    return ms;
}


inline DFTI_DESCRIPTOR_HANDLE find_DFTI_descriptor_handle( int N1, int N2 )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = NULL;
    long int desc_key = ( long int ) N1 * N2_orig + N2;
    
    std::tr1::unordered_map< long int, DFTI_DESCRIPTOR_HANDLE >::const_iterator got = desc_map.find( desc_key );    

    if ( got == desc_map.end( ) )
      {    
	    MKL_LONG status;
	    MKL_LONG len[ 2 ] = { N1, N2 };

        status = DftiCreateDescriptor( &desc_handle, DFTI_DOUBLE, DFTI_REAL, 2, len );
        status = DftiSetValue( desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
        status = DftiSetValue( desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX );        
        status = DftiSetValue( desc_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );        
        status = DftiCommitDescriptor( desc_handle );
        
        desc_map[ desc_key ] = desc_handle;       
	  }    
	else desc_handle = got->second;
	    	  
	return desc_handle;  
}


inline void mkl_fft_forward( double *input_buffer, double complex *output_buffer, int N1, int N2 )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = find_DFTI_descriptor_handle( N1, N2 );

    DftiComputeForward( desc_handle, input_buffer, output_buffer );    
}


inline void mkl_fft_backward( double complex* input_buffer, double *output_buffer, int N1, int N2 )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = find_DFTI_descriptor_handle( N1, N2 );

    DftiComputeBackward( desc_handle, input_buffer, output_buffer );
    
    int N1N2 = N1 * N2;
    
    #pragma omp parallel for
    for ( int i = 0; i < N1N2; i++ )
       output_buffer[ i ] /= N1N2;       
}


inline void copy_array_horizontal( double *src, double *dst, int L )
{
    #pragma omp parallel for
    for ( int i = 0; i < L; i++ )
       dst[ i ] = src[ i ];
}


inline void copy_array_transposed( double *src, double *dst, int N1, int N2 )
{
    if ( N1 > N2 )
       {
	    #pragma omp parallel for
	    for ( int i = 0; i < N1; i++ )
	       for ( int j = 0, d = i, s = i * N2; j < N2; j++, s++, d += N1 )
	          dst[ d ] = src[ s ];
       } 
    else
       {
	    #pragma omp parallel for
	    for ( int j = 0; j < N2; j++ )
	       for ( int i = 0, s = j, d = j * N1; i < N1; i++, d++, s += N2 )
	          dst[ d ] = src[ s ];
       }         
}


inline void copy_array_vertical( double *src, double *dst, int N1, int N2s, int N2d, int L )
{
    #pragma omp parallel for
    for ( int i = 0, s = 0, d = 0; i < N1; i++, s += N2s, d += N2d )
      for ( int j = 0; j < L; j++ )
         dst[ d + j ] = src[ s + j ];
}



void create_jacobi_stencil( double *stencil_real, int N1, int N2 )
{
    int N1N2 = N1 * N2;
        
    #pragma omp parallel for
    for ( int i = 0; i < N1N2; i++ )
        stencil_real[ i ] = 0;        

    int k = 0;
    
    stencil_real[ k + 0 ] = 0.04;
    stencil_real[ k + 1 ] = 0.04;    
    stencil_real[ k + 2 ] = 0.04;    
    stencil_real[ k + N2 - 1 ] = 0.04;    
    stencil_real[ k + N2 - 2 ] = 0.04;    

    k = N2;
    stencil_real[ k + 0 ] = 0.04;
    stencil_real[ k + 1 ] = 0.04;    
    stencil_real[ k + 2 ] = 0.04;    
    stencil_real[ k + N2 - 1 ] = 0.04;    
    stencil_real[ k + N2 - 2 ] = 0.04;    

    k = N2 + N2;
    stencil_real[ k + 0 ] = 0.04;
    stencil_real[ k + 1 ] = 0.04;    
    stencil_real[ k + 2 ] = 0.04;    
    stencil_real[ k + N2 - 1 ] = 0.04;    
    stencil_real[ k + N2 - 2 ] = 0.04;    

    k = N1N2 - N2 - N2;
    stencil_real[ k + 0 ] = 0.04;
    stencil_real[ k + 1 ] = 0.04;    
    stencil_real[ k + 2 ] = 0.04;    
    stencil_real[ k + N2 - 1 ] = 0.04;    
    stencil_real[ k + N2 - 2 ] = 0.04;    

    k = N1N2 - N2;
    stencil_real[ k + 0 ] = 0.04;
    stencil_real[ k + 1 ] = 0.04;    
    stencil_real[ k + 2 ] = 0.04;    
    stencil_real[ k + N2 - 1 ] = 0.04;    
    stencil_real[ k + N2 - 2 ] = 0.04;    
}


int find_stencil_index( int N1, int N2, int T )
{
    int stencil_index = -1;
    char s[ 100 ];
    
    sprintf( s, "%d-%d-%d", N1, N2, T );
    
    std::string stencil_key = s;
    
    std::tr1::unordered_map< std::string, int >::const_iterator got = stencil_map.find( stencil_key );    
    
    if ( got == stencil_map.end( ) )
      {            
        int N1N2 = N1 * N2, stencil_index_1;
        
        int save = 1, next_stencil_index_temp = next_stencil_index;
    
        if ( next_stencil_index + ( N1N2 << 2 ) > max_stencil_index ) save = 0;

        sprintf( s, "%d-%d-%d", N1, N2, 1 );
              
        std::string stencil_key_1 = s;
        
        got = stencil_map.find( stencil_key_1 );
      
        if  ( got == stencil_map.end( ) )
           {        
	         create_jacobi_stencil( stencil_real, N1, N2 );
	        
	         mkl_fft_forward( stencil_real, stencil_complex + next_stencil_index_temp, N1, N2 );
	
	         stencil_index_1 = next_stencil_index_temp;
	         
	         if ( save ) stencil_map[ stencil_key_1 ] = stencil_index_1;       
	
	         next_stencil_index_temp += N1N2;
           }
        else stencil_index_1 = got->second;
        
        if ( T > 1 )
           { 
             int t = 1, TT = ( T >> 1 );
             
             while ( TT )
               {
                 t <<= 1;
                 TT >>= 1;
               }
               
	         #pragma omp parallel for
			 #pragma ivdep
			 #pragma vector always	            	        
	         for ( int i = 0; i < N1N2; i++ )
	            stencil_complex[ next_stencil_index_temp + i ] = stencil_complex[ stencil_index_1 + i ];        
	                        
	         while ( t > 1 )   
	           {
    	         t >>= 1;   
	           
	             if ( T & t )
	               {
			        #pragma omp parallel for
					#pragma ivdep
					#pragma vector always	            	        
			        for ( int i = 0; i < N1N2; i++ )
			            stencil_complex[ next_stencil_index_temp + i ] = stencil_complex[ next_stencil_index_temp + i ] * stencil_complex[ next_stencil_index_temp + i ] 
			            										  * stencil_complex[ stencil_index_1 + i ];        	               
	               }
	             else
	               {
			        #pragma omp parallel for
					#pragma ivdep
					#pragma vector always	            	        
			        for ( int i = 0; i < N1N2; i++ )
			            stencil_complex[ next_stencil_index_temp + i ] *= stencil_complex[ next_stencil_index_temp + i ];        	               
	               }  
	           }
           
             stencil_index = next_stencil_index_temp;
             
             if ( save ) 
                {
                  stencil_map[ stencil_key ] = stencil_index;
                  next_stencil_index = next_stencil_index_temp + N1N2;       
                }  
           }
        else if ( save ) next_stencil_index = next_stencil_index_temp;  
	  }    
	else stencil_index = got->second;
	  
	return stencil_index;    
}



void apply_stencil_fft( double *input__real, double *output__real, int N1, int N2, int T )
{   
    int N1N2 = N1 * N2; 
    int stencil_index = find_stencil_index( N1, N2, T );
    double complex *stencil__complex = stencil_complex + stencil_index;

    mkl_fft_forward( input__real, input_complex, N1, N2 );
    
    #pragma omp parallel for
	#pragma ivdep
	#pragma vector always        
    for ( int i = 0; i < N1N2; i++ )
       input_complex[ i ] *= stencil__complex[ i ];

    mkl_fft_backward( input_complex, output__real, N1, N2 );
}



void apply_stencil_loops( double *input__real, double *output__real, double *tmp__real, int N1, int N2, int T )
{
    int N1N2 = N1 * N2;

    int flipped = ( N1 < N2 ) ? 1 : 0;
    
    if ( flipped )
      {
        copy_array_transposed( input__real, tmp__real, N1, N2 );
      
        int N = N1;        
        N1 = N2;
        N2 = N;        
      }
    else copy_array_horizontal( input__real, tmp__real, N1N2 );   

    for ( int r = 0; r < T; r++ )
       {
        if ( r & 1 )
           {
            int i, j;

	        #pragma omp parallel for
	        for ( i = 2; i < N1 - 2; i++ ) 
	          {
	            double *tmp___real = tmp__real + i * N2;
	            double *output___real = output__real + i * N2;
	            
	            int j;

				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   tmp___real[ j ] = output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ];

 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   tmp___real[ j ] += output___real[ j - N2 - 2 ] + output___real[ j - N2 - 1 ] + output___real[ j - N2 ] + output___real[ j - N2 + 1 ] + output___real[ j - N2 + 2 ]; 
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   tmp___real[ j ] += output___real[ j - 2 ] + output___real[ j - 1 ] + output___real[ j ] + output___real[ j + 1 ] + output___real[ j + 2 ];
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   tmp___real[ j ] += output___real[ j + N2 - 2 ] + output___real[ j + N2 - 1 ] + output___real[ j + N2 ] + output___real[ j + N2 + 1 ] + output___real[ j + N2 + 2 ];
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   tmp___real[ j ] = 0.04 * ( tmp___real[ j ] + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );    	        
                   
                j = 1;                   
                tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                         + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                         + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                         + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                         + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );                   

                j = 0;                   
                tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                         + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                         + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                         + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                         + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );                   

                j = N2 - 2;                   
                tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ]
                                         + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]
                                         + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]
                                         + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]
                                         + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] );    	        

                j = N2 - 1;                   
                tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ]
                                         + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]
                                         + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]
                                         + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]
                                         + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] );    	        
              }       

            double *tmp___real;
            double *output___real;


            i = 1;

            tmp___real = tmp__real + i * N2;
            output___real = output__real + i * N2;
            
			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = output___real[ j - N2 - 2 ] + output___real[ j - N2 - 1 ] + output___real[ j - N2 ] + output___real[ j - N2 + 1 ] + output___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j - 2 ] + output___real[ j - 1 ] + output___real[ j ] + output___real[ j + 1 ] + output___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j + N2 - 2 ] + output___real[ j + N2 - 1 ] + output___real[ j + N2 ] + output___real[ j + N2 + 1 ] + output___real[ j + N2 + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = 0.04 * ( tmp___real[ j ] + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );               


            j = 1;            
            tmp___real[ j ] = 0.04 * ( output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                     + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );    	        
                                     
            j = 0;
            tmp___real[ j ] = 0.04 * ( output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                     + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );    	                                                         

            j = N2 - 2;
            tmp___real[ j ] = 0.04 * ( output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ] 
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]
                                     + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] );    	                                                         

            j = N2 - 1;
            tmp___real[ j ] = 0.04 * ( output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]
                                     + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] );

                        

            i = 0;

            tmp___real = tmp__real + i * N2;
            output___real = output__real + i * N2;
            
			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = output___real[ j - 2 ] + output___real[ j - 1 ] + output___real[ j ] + output___real[ j + 1 ] + output___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j + N2 - 2 ] + output___real[ j + N2 - 1 ] + output___real[ j + N2 ] + output___real[ j + N2 + 1 ] + output___real[ j + N2 + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = 0.04 * ( tmp___real[ j ] + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );               


            j = 1;            
            tmp___real[ j ] = 0.04 * ( output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                     + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );    	        
                                     
            j = 0;
            tmp___real[ j ] = 0.04 * ( output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ]
                                     + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] + output___real[ j + ( N2 << 1 ) + 2 ] );    	                                                         

            j = N2 - 2;
            tmp___real[ j ] = 0.04 * ( output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]
                                     + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] + output___real[ j + ( N2 << 1 ) + 1 ] );

            j = N2 - 1;
            tmp___real[ j ] = 0.04 * ( output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]
                                     + output___real[ j + ( N2 << 1 ) - 2 ] + output___real[ j + ( N2 << 1 ) - 1 ] + output___real[ j + ( N2 << 1 ) ] );



            i = N1 - 2;

            tmp___real = tmp__real + i * N2;
            output___real = output__real + i * N2;

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j - N2 - 2 ] + output___real[ j - N2 - 1 ] + output___real[ j - N2 ] + output___real[ j - N2 + 1 ] + output___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j - 2 ] + output___real[ j - 1 ] + output___real[ j ] + output___real[ j + 1 ] + output___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = 0.04 * ( tmp___real[ j ] + output___real[ j + N2 - 2 ] + output___real[ j + N2 - 1 ] + output___real[ j + N2 ] + output___real[ j + N2 + 1 ] + output___real[ j + N2 + 2 ] );


            j = 1;            
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                     + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ] );                                     
            j = 0;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                     + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ]
                                     + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ]          + output___real[ j + N2 + 2 ] );
            j = N2 - 2;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ]
                                     + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ]          + output___real[ j + N2 + 1 ] );

            j = N2 - 1;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ]
                                     + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]
                                     + output___real[ j + N2 - 2 ]          + output___real[ j + N2 - 1 ]          + output___real[ j + N2 ] );

               

            i = N1 - 1;

            tmp___real = tmp__real + i * N2;
            output___real = output__real + i * N2;

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] += output___real[ j - N2 - 2 ] + output___real[ j - N2 - 1 ] + output___real[ j - N2 ] + output___real[ j - N2 + 1 ] + output___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               tmp___real[ j ] = 0.04 * ( tmp___real[ j ] + output___real[ j - 2 ] + output___real[ j - 1 ] + output___real[ j ] + output___real[ j + 1 ] + output___real[ j + 2 ] );
               
               
            j = 1;            
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                     + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ] );                                     
            j = 0;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ] + output___real[ j - ( N2 << 1 ) + 2 ]
                                     + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]          + output___real[ j - N2 + 2 ] 
                                     + output___real[ j ]               + output___real[ j + 1 ]               + output___real[ j + 2 ] );
            j = N2 - 2;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ] + output___real[ j - ( N2 << 1 ) + 1 ]
                                     + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]          + output___real[ j - N2 + 1 ]
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ]               + output___real[ j + 1 ] );

            j = N2 - 1;
            tmp___real[ j ] = 0.04 * ( output___real[ j - ( N2 << 1 ) - 2 ] + output___real[ j - ( N2 << 1 ) - 1 ] + output___real[ j - ( N2 << 1 ) ]
                                     + output___real[ j - N2 - 2 ]          + output___real[ j - N2 - 1 ]          + output___real[ j - N2 ]
                                     + output___real[ j - 2 ]               + output___real[ j - 1 ]               + output___real[ j ] );               
           }
        else
           {
            int i, j;

	        #pragma omp parallel for
	        for ( i = 2; i < N1 - 2; i++ ) 
	          {
	            double *output___real = output__real + i * N2;
	            double *tmp___real = tmp__real + i * N2;

                int j;

				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   output___real[ j ] = tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ];

 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   output___real[ j ] += tmp___real[ j - N2 - 2 ] + tmp___real[ j - N2 - 1 ] + tmp___real[ j - N2 ] + tmp___real[ j - N2 + 1 ] + tmp___real[ j - N2 + 2 ]; 
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   output___real[ j ] += tmp___real[ j - 2 ] + tmp___real[ j - 1 ] + tmp___real[ j ] + tmp___real[ j + 1 ] + tmp___real[ j + 2 ];
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   output___real[ j ] += tmp___real[ j + N2 - 2 ] + tmp___real[ j + N2 - 1 ] + tmp___real[ j + N2 ] + tmp___real[ j + N2 + 1 ] + tmp___real[ j + N2 + 2 ];
 
 				#pragma ivdep
				#pragma vector always
    	        for ( j = 2; j < N2 - 2; j++ ) 
                   output___real[ j ] = 0.04 * ( output___real[ j ] + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );    	        
                   
                j = 1;                   
                output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                            + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                            + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                            + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                            + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );                   

                j = 0;                   
                output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                            + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                            + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                            + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                            + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );                   

                j = N2 - 2;                   
                output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ]
                                            + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]
                                            + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]
                                            + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]
                                            + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] );    	        

                j = N2 - 1;                   
                output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ]
                                            + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]
                                            + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]
                                            + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]
                                            + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] );    	        
              }       

            double *output___real;
            double *tmp___real;


            i = 1;

            output___real = output__real + i * N2;
            tmp___real = tmp__real + i * N2;
            
			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = tmp___real[ j - N2 - 2 ] + tmp___real[ j - N2 - 1 ] + tmp___real[ j - N2 ] + tmp___real[ j - N2 + 1 ] + tmp___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j - 2 ] + tmp___real[ j - 1 ] + tmp___real[ j ] + tmp___real[ j + 1 ] + tmp___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j + N2 - 2 ] + tmp___real[ j + N2 - 1 ] + tmp___real[ j + N2 ] + tmp___real[ j + N2 + 1 ] + tmp___real[ j + N2 + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = 0.04 * ( output___real[ j ] + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );               


            j = 1;            
            output___real[ j ] = 0.04 * ( tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );    	        
                                     
            j = 0;
            output___real[ j ] = 0.04 * ( tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                        + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );    	                                                         

            j = N2 - 2;
            output___real[ j ] = 0.04 * ( tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ] 
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] );    	                                                         

            j = N2 - 1;
            output___real[ j ] = 0.04 * ( tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] );

                        

            i = 0;

            output___real = output__real + i * N2;
            tmp___real = tmp__real + i * N2;
            
			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = tmp___real[ j - 2 ] + tmp___real[ j - 1 ] + tmp___real[ j ] + tmp___real[ j + 1 ] + tmp___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j + N2 - 2 ] + tmp___real[ j + N2 - 1 ] + tmp___real[ j + N2 ] + tmp___real[ j + N2 + 1 ] + tmp___real[ j + N2 + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = 0.04 * ( output___real[ j ] + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );               


            j = 1;            
            output___real[ j ] = 0.04 * ( tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );    	        
                                     
            j = 0;
            output___real[ j ] = 0.04 * ( tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ]
                                        + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] + tmp___real[ j + ( N2 << 1 ) + 2 ] );    	                                                         

            j = N2 - 2;
            output___real[ j ] = 0.04 * ( tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] + tmp___real[ j + ( N2 << 1 ) + 1 ] );

            j = N2 - 1;
            output___real[ j ] = 0.04 * ( tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]
                                        + tmp___real[ j + ( N2 << 1 ) - 2 ] + tmp___real[ j + ( N2 << 1 ) - 1 ] + tmp___real[ j + ( N2 << 1 ) ] );



            i = N1 - 2;

            output___real = output__real + i * N2;
            tmp___real = tmp__real + i * N2;

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j - N2 - 2 ] + tmp___real[ j - N2 - 1 ] + tmp___real[ j - N2 ] + tmp___real[ j - N2 + 1 ] + tmp___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j - 2 ] + tmp___real[ j - 1 ] + tmp___real[ j ] + tmp___real[ j + 1 ] + tmp___real[ j + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = 0.04 * ( output___real[ j ] + tmp___real[ j + N2 - 2 ] + tmp___real[ j + N2 - 1 ] + tmp___real[ j + N2 ] + tmp___real[ j + N2 + 1 ] + tmp___real[ j + N2 + 2 ] );


            j = 1;            
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                        + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ] );                                     
            j = 0;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                        + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ]
                                        + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ]          + tmp___real[ j + N2 + 2 ] );
            j = N2 - 2;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ]
                                        + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ]          + tmp___real[ j + N2 + 1 ] );

            j = N2 - 1;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ]
                                        + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]
                                        + tmp___real[ j + N2 - 2 ]          + tmp___real[ j + N2 - 1 ]          + tmp___real[ j + N2 ] );

               

            i = N1 - 1;

            output___real = output__real + i * N2;
            tmp___real = tmp__real + i * N2;

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ];

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] += tmp___real[ j - N2 - 2 ] + tmp___real[ j - N2 - 1 ] + tmp___real[ j - N2 ] + tmp___real[ j - N2 + 1 ] + tmp___real[ j - N2 + 2 ]; 

			#pragma ivdep
			#pragma vector always
	        for ( j = 2; j < N2 - 2; j++ ) 
               output___real[ j ] = 0.04 * ( output___real[ j ] + tmp___real[ j - 2 ] + tmp___real[ j - 1 ] + tmp___real[ j ] + tmp___real[ j + 1 ] + tmp___real[ j + 2 ] );
               
               
            j = 1;            
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                        + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ] );                                     
            j = 0;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ] + tmp___real[ j - ( N2 << 1 ) + 2 ]
                                        + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]          + tmp___real[ j - N2 + 2 ] 
                                        + tmp___real[ j ]               + tmp___real[ j + 1 ]               + tmp___real[ j + 2 ] );
            j = N2 - 2;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ] + tmp___real[ j - ( N2 << 1 ) + 1 ]
                                        + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]          + tmp___real[ j - N2 + 1 ]
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ]               + tmp___real[ j + 1 ] );

            j = N2 - 1;
            output___real[ j ] = 0.04 * ( tmp___real[ j - ( N2 << 1 ) - 2 ] + tmp___real[ j - ( N2 << 1 ) - 1 ] + tmp___real[ j - ( N2 << 1 ) ]
                                        + tmp___real[ j - N2 - 2 ]          + tmp___real[ j - N2 - 1 ]          + tmp___real[ j - N2 ]
                                        + tmp___real[ j - 2 ]               + tmp___real[ j - 1 ]               + tmp___real[ j ] );                   
           }           
       }

    
    if ( flipped )
      {
        if ( !( T & 1 ) ) copy_array_transposed( tmp__real, output__real, N1, N2 );      
        else 
            {
              copy_array_transposed( output__real, tmp__real, N1, N2 );
              copy_array_horizontal( tmp__real, output__real, N1N2 );              
            }
      }
    else if ( !( T & 1 ) ) copy_array_horizontal( tmp__real, output__real, N1N2 );
}



void compute_aperiodic_stencil( double *input__real, double *output__real, double *tmp__real, int N1, int N2, int T, int vertical )
{
    if ( ( T <= T_base ) || ( ( long int ) N1 * N2 * T <= V_base ) )
       {
         apply_stencil_loops( input__real, output__real, tmp__real, N1, N2, T );
         return;
       }

    int N1N2 = N1 * N2;    
    
    if ( ( T * max( 16, MIN_NT_RATIO ) ) >= min( N1, N2 ) )
       {
         compute_aperiodic_stencil( input__real, output__real, tmp__real, N1, N2, ( T >> 1 ), vertical );       

         if ( T & 1 ) apply_stencil_loops( output__real, input__real, tmp__real, N1, N2, 1 );
         else copy_array_horizontal( output__real, input__real, N1N2 );   

         compute_aperiodic_stencil( input__real, output__real, tmp__real, N1, N2, ( T >> 1 ), vertical );                
         
         return;
       }

    apply_stencil_fft( input__real, output__real, N1, N2, T );
   
    int N2T = N2 * T;    
    int N22T = ( N2T << 1 );
    int N24T = ( N2T << 2 );
   
    if ( !vertical )
       {
	    copy_array_horizontal( input__real, input__real + N1N2, N24T );           
		copy_array_horizontal( input__real + ( N1N2 - N24T ), input__real + ( N1N2 + N24T ), N24T );   
		
		compute_aperiodic_stencil( input__real + N1N2, output__real + N1N2, tmp__real + N1N2, ( T << 3 ), N2, T, 0 );

		copy_array_horizontal( output__real + N1N2, output__real, N22T );   
		copy_array_horizontal( output__real + ( N1N2 + N24T + N22T ), output__real + ( N1N2 - N22T ), N22T );      		
	   }
	   
    		
	copy_array_vertical( input__real, input__real + N1N2, N1, N2, ( T << 3 ), ( T << 2 ) );           
	copy_array_vertical( input__real + N2 - ( T << 2 ), input__real + N1N2 + ( T << 2 ), N1, N2, ( T << 3 ), ( T << 2 ) );   
		    
	compute_aperiodic_stencil( input__real + N1N2, output__real + N1N2, tmp__real + N1N2, N1, ( T << 3 ), T, 1 );

	copy_array_vertical( output__real + N1N2 + ( ( T * T ) << 4 ), output__real + N22T, N1 - ( T << 2 ), ( T << 3 ), N2, ( T << 1 ) );           
	copy_array_vertical( output__real + N1N2 + ( ( T * T ) << 4 ) + ( T << 2 ) + ( T << 1 ), output__real + N22T + N2 - ( T << 1 ), N1 - ( T << 2 ), ( T << 3 ), N2, ( T << 1 ) );   	
}


void compute_aperiodic_jacobi_stencil( double *input__real, double *output__real, double *tmp__real, int N1, int N2, int T )
{
//    compute_aperiodic_stencil( input__real, output__real, tmp__real, N1, N2, T, 0 );    

    int N1N2 = N1 * N2; 
    
    int t = -1;
    
    while ( T )
      {
        t++;
        
        if ( T & 1 ) 
           {
             compute_aperiodic_stencil( input__real, output__real, tmp__real, N1, N2, ( 1 << t ), 0 );
             if ( T >> 1 ) copy_array_horizontal( output__real, input__real, N1N2 );
           }  
        
        T >>= 1;
      }      
}


int initialize( int N1, int N2, int input_backup )
{
    int N1N2 = N1 * N2;
    
    input_real = ( double * ) malloc( sizeof( double ) * N1N2 * 2 );
    if ( input_backup ) input_real_orig = ( double * ) malloc( sizeof( double ) * N1N2 );
    output_real = ( double * ) malloc( sizeof( double ) * N1N2 * 2 );
    tmp_real = ( double * ) malloc( sizeof( double ) * N1N2 * 2 );
    stencil_real = ( double * ) malloc( sizeof( double ) * N1N2 );

    next_stencil_index = 0;
    max_stencil_index = STENCIL_SPACE_FACTOR * N1N2;
    stencil_complex = ( double complex * ) malloc( sizeof( double complex ) * N1N2 * ( STENCIL_SPACE_FACTOR + 2 ) );
    input_complex = ( double complex * ) malloc( sizeof( double complex ) * N1N2 );
    
    if ( ( input_real == NULL ) || ( input_backup && ( input_real_orig == NULL ) ) || ( output_real == NULL ) || ( tmp_real == NULL ) || ( stencil_real == NULL ) 
       || ( input_complex == NULL ) || ( stencil_complex == NULL ) )
        {
          cout << "Error: memory allocation failed!" << endl;
          return 1;
        }
    else cout << "Success: memory allocated!" << endl;        
    
    for ( int i = 0; i < N1N2; ++i ) 
       input_real[ i ] = 1.0 * ( rand( ) % MAX_VAL ); 
       
    return 0;   
}



void destroy( int n1, int n2, int input_backup )
{
    free( input_real );
    if ( input_backup ) free( input_real_orig );
    free( output_real );    
    free( tmp_real );
    free( stencil_real );
    
    free( stencil_complex );
    free( input_complex );   
}


void verify( double *input__real, double *output__real, int N1, int N2, int T )
{
    int N1N2 = N1 * N2;
    
    copy_array_horizontal( input__real, tmp_real, N1N2 );
    
    double *tmp2_real = tmp_real + N1N2;
    
    for ( int r = 0; r < T; r++ ) 
      {
        int i, j;
         
        #pragma omp parallel for
        for ( i = 0; i < N1; i++ ) 
	       for ( j = 0; j < N2; j++ ) 
	          {
	            int k = i * N2 + j;
                tmp2_real[ k ] = 0.04 * ( ( ( ( i >= 2 ) && ( j >= 2 ) ) ? tmp_real[ k - ( N2 << 1 ) - 2 ] : 0 )      + ( ( ( i >= 2 ) && ( j >= 1 ) ) ? tmp_real[ k - ( N2 << 1 ) - 1 ] : 0 )      + ( ( i >= 2 ) ? tmp_real[ k - ( N2 << 1 ) ] : 0 )      + ( ( ( i >= 2 ) && ( j <= N2 - 2 ) ) ? tmp_real[ k - ( N2 << 1 ) + 1 ] : 0 )      + ( ( ( i >= 2 ) && ( j <= N2 - 3 ) ) ? tmp_real[ k - ( N2 << 1 ) + 2 ] : 0 )
                                        + ( ( ( i >= 1 ) && ( j >= 2 ) ) ? tmp_real[ k - N2 - 2 ] : 0 )               + ( ( ( i >= 1 ) && ( j >= 1 ) ) ? tmp_real[ k - N2 - 1 ] : 0 )               + ( ( i >= 1 ) ? tmp_real[ k - N2 ] : 0 )               + ( ( ( i >= 1 ) && ( j <= N2 - 2 ) ) ? tmp_real[ k - N2 + 1 ] : 0 )               + ( ( ( i >= 1 ) && ( j <= N2 - 3 ) ) ? tmp_real[ k - N2 + 2 ] : 0 )
                                        + ( ( j >= 2 ) ? tmp_real[ k - 2 ] : 0 )                                      + ( ( j >= 1 ) ? tmp_real[ k - 1 ] : 0 )                                      + tmp_real[ k ]                                         + ( ( j <= N2 - 2 ) ? tmp_real[ k + 1 ] : 0 )                                      + ( ( j <= N2 - 3 ) ? tmp_real[ k + 2 ] : 0 )
                                        + ( ( ( i <= N1 - 2 ) && ( j >= 2 ) ) ? tmp_real[ k + N2 - 2 ] : 0 )          + ( ( ( i <= N1 - 2 ) && ( j >= 1 ) ) ? tmp_real[ k + N2 - 1 ] : 0 )          + ( ( i <= N1 - 2 ) ? tmp_real[ k + N2 ] : 0 )          + ( ( ( i <= N1 - 2 ) && ( j <= N2 - 2 ) ) ? tmp_real[ k + N2 + 1 ] : 0 )          + ( ( ( i <= N1 - 2 ) && ( j <= N2 - 3 ) ) ? tmp_real[ k + N2 + 2 ] : 0 )
                                        + ( ( ( i <= N1 - 3 ) && ( j >= 2 ) ) ? tmp_real[ k + ( N2 << 1 ) - 2 ] : 0 ) + ( ( ( i <= N1 - 3 ) && ( j >= 1 ) ) ? tmp_real[ k + ( N2 << 1 ) - 1 ] : 0 ) + ( ( i <= N1 - 3 ) ? tmp_real[ k + ( N2 << 1 ) ] : 0 ) + ( ( ( i <= N1 - 3 ) && ( j <= N2 - 2 ) ) ? tmp_real[ k + ( N2 << 1 ) + 1 ] : 0 ) + ( ( ( i <= N1 - 3 ) && ( j <= N2 - 3 ) ) ? tmp_real[ k + ( N2 << 1 ) + 2 ] : 0 ) );
              }  
              
        copy_array_horizontal( tmp2_real, tmp_real, N1N2 );            
      }


    int cnt = 0; 
      
    for ( int i = 0; i < N1N2; i++ )
      {           
        if ( fabs( tmp_real[ i ] - output__real[ i ] ) > 1e-8 )
            cnt++;
      }      

    cout << "Number of incorrect output entries = " << cnt << ", total entries checked = " << N1N2 << endl;
}


int main( int argc, char *argv[ ] )
{
    int numThreads;

    if ( argc < 3 )
      {
        cout << "Enter: N2 N2 T ( > 0 ) T_base V_base numThreads verify (0/1)" << endl;
        return 1;
      }

    N1_orig = atoi( argv[ 1 ] );
    N2_orig = atoi( argv[ 2 ] );
       
    T_orig = atoi( argv[ 3 ] );

    T_base = 128;
    if ( argc > 4 ) T_base = atoi( argv[ 4 ] );

    V_base = 1024;
    if ( argc > 5 ) V_base = atoi( argv[ 5 ] );

    if ( ( N1_orig < 1 ) || ( N2_orig < 1 ) || ( T_orig < 1 ) || ( T_base < 1 ) || ( V_base < 1 ) )
      {
        cout << "Error: N1, N2, T, T_base, and V_base must be positive integers." << endl;
        return 2; 
      }


    numThreads = 1;
    if ( argc > 6 )
      {
        numThreads = max( 1, atoi( argv[ 6 ] ) );
        omp_set_num_threads( numThreads );
//        omp_set_dynamic( 0 );
//        omp_set_nested( 1 );        
      }

    int verifyRes = 0;
    if ( argc > 7 ) verifyRes = atoi( argv[ 7 ] );

    int N1 = N1_orig;
    int N2 = N2_orig;
    int N1N2 = N1 * N2;

    N_THREADS = numThreads;

    if ( initialize( N1_orig, N2_orig, verifyRes ) )
      {
        destroy( N1_orig, N2_orig, verifyRes );
        return 1; 
      }
    
    #ifdef USE_PAPI
        papi_init( );
    #endif

    if ( verifyRes ) copy_array_horizontal( input_real, input_real_orig, N1N2 );

    long start = getTime( );
    
#ifdef POLYBENCH
    /* Start timer. */
    polybench_start_instruments;
#endif

    compute_aperiodic_jacobi_stencil( input_real, output_real, tmp_real, N1_orig, N2_orig, T_orig );
        
#ifdef POLYBENCH
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
#endif
    long end = getTime( );
    cout << "N1 = " << N1 << ", N2 = " << N2 << ", T = " << T_orig << ", T_BASE = " << T_base << ", V_BASE = " << V_base << ", numThreads = " << numThreads << ", time = " << (end - start) / 1000.0 << endl;

#ifdef USE_PAPI        
    countTotalMiss( p );
    PAPI_shutdown( );
    delete threadcounter;
    for ( int i = 0; i < p; i++ ) delete l2miss[ i ];
    delete l2miss;
    delete errstring;
    delete EventSet;
    delete eventCode;
#endif  

    if ( verifyRes )
      {
        copy_array_horizontal( input_real_orig, input_real, N1N2 );
	
	    long start_iter = getTime( );    
	    verify( input_real, output_real, N1_orig, N2_orig, T_orig );
	    long end_iter = getTime( );
	    // cout << "Time (Iter): " << end_iter - start_iter << endl;
      }

    destroy( N1_orig, N2_orig, verifyRes );

    return 0;
}

