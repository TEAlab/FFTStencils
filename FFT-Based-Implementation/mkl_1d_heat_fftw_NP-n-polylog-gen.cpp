/**
 * For compiling --> 
        [NOT USING THIS] export GOMP_CPU_AFFINITY='0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268'
        export OMP_NUM_THREADS=68
        set MKL_NUM_THREADS = 64
        icc -std=gnu++98 -O3 -qopenmp -xhost -ansi-alias -ipo -qopt-prefetch=3 -AVX512 mkl_1d_heat_fftw_NP-n-polylog-gen.cpp -o mkl_1d_heat_fftw_NP-n-polylog-gen -lm -mkl

 * For running -->
 * ./mkl_1d_heat_fftw_NP-n-polylog-gen 8000 8000 64 32768 68 0
 * Success: memory allocated!
 * N = 8000, T = 8000, T_BASE = 64, V_BASE = 32768, numThreads = 68, time = 11.844
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
   #define MAX_n    25
#endif   

#define MAX_N       ( ( 1 << MAX_n ) + 100 )

#ifndef MAX_VAL
   #define MAX_VAL  1024
#endif   

#ifndef STENCIL_SPACE_FACTOR
   #define STENCIL_SPACE_FACTOR  10
#endif   

#ifndef MIN_NT_RATIO
   #define MIN_NT_RATIO 8
#endif


int N_THREADS;

int N_orig, T_orig, T_base, V_base;

double *input_real;
double *input_real_orig;
double *stencil_real;
double *output_real;
double *tmp_real;

double complex *stencil_complex;
double complex *input_complex;

int next_stencil_index = 0, max_stencil_index;

std::tr1::unordered_map< std::string, int > stencil_map;
std::tr1::unordered_map< int, DFTI_DESCRIPTOR_HANDLE > desc_map;


long getTime( void )
{
    struct timeval tp;

    gettimeofday( &tp, NULL );
    
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    return ms;
}


inline DFTI_DESCRIPTOR_HANDLE find_DFTI_descriptor_handle( int N )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = NULL;
    int desc_key = N;
    
    std::tr1::unordered_map< int, DFTI_DESCRIPTOR_HANDLE >::const_iterator got = desc_map.find( desc_key );    

    if ( got == desc_map.end( ) )
      {    
	    MKL_LONG status;

        status = DftiCreateDescriptor( &desc_handle, DFTI_DOUBLE, DFTI_REAL, 1, N );
        status = DftiSetValue( desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
        status = DftiSetValue( desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX );        
        status = DftiSetValue( desc_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT );        
        status = DftiCommitDescriptor( desc_handle );
        
        desc_map[ desc_key ] = desc_handle;       
	  }    
	else desc_handle = got->second;
	    	  
	return desc_handle;  
}


inline void mkl_fft_forward( double *input_buffer, double complex *output_buffer, int N )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = find_DFTI_descriptor_handle( N );

    DftiComputeForward( desc_handle, input_buffer, output_buffer );    
}


inline void mkl_fft_backward( double complex* input_buffer, double *output_buffer, int N )
{
    DFTI_DESCRIPTOR_HANDLE desc_handle = find_DFTI_descriptor_handle( N );

    DftiComputeBackward( desc_handle, input_buffer, output_buffer );
    
    #pragma omp parallel for
    for ( int i = 0; i < N; i++ )
       output_buffer[ i ] /= N;       
}


inline void copy_array( double *src, double *dst, int L )
{
    #pragma omp parallel for
    for ( int i = 0; i < L; i++ )
       dst[ i ] = src[ i ];
}


void create_heat_stencil( double *stencil_real, int N )
{
    #pragma omp parallel for
    for ( int i = 0; i < N; i++ )
        stencil_real[ i ] = 0;        

    stencil_real[ 0 ] = ( -1 ) * 0.125 * 2.0 + 1;
    stencil_real[ 1 ] = 0.125;    
    stencil_real[ N - 1 ] = 0.125;    
}


int find_stencil_index( int N, int T )
{
    int stencil_index = -1;
    char s[ 100 ];
    
    sprintf( s, "%d-%d", N, T );
    
    std::string stencil_key = s;
    
    std::tr1::unordered_map< std::string, int >::const_iterator got = stencil_map.find( stencil_key );    
    
    if ( got == stencil_map.end( ) )
      {            
        int stencil_index_1;
        
        int save = 1, next_stencil_index_temp = next_stencil_index;
    
        if ( next_stencil_index + ( N << 2 ) > max_stencil_index ) save = 0;

        sprintf( s, "%d-%d", N, 1 );
              
        std::string stencil_key_1 = s;
        
        got = stencil_map.find( stencil_key_1 );
      
        if  ( got == stencil_map.end( ) )
           {        
	         create_heat_stencil( stencil_real, N );
	        
	         mkl_fft_forward( stencil_real, stencil_complex + next_stencil_index_temp, N );
	
	         stencil_index_1 = next_stencil_index_temp;
	         
	         if ( save ) stencil_map[ stencil_key_1 ] = stencil_index_1;       
	
	         next_stencil_index_temp += N;
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
	         for ( int i = 0; i < N; i++ )
	            stencil_complex[ next_stencil_index_temp + i ] = stencil_complex[ stencil_index_1 + i ];        
	                        
	         while ( t > 1 )   
	           {
    	         t >>= 1;   
	           
	             if ( T & t )
	               {
			        #pragma omp parallel for
					#pragma ivdep
					#pragma vector always	            	        
			        for ( int i = 0; i < N; i++ )
			            stencil_complex[ next_stencil_index_temp + i ] = stencil_complex[ next_stencil_index_temp + i ] * stencil_complex[ next_stencil_index_temp + i ] 
			            										       * stencil_complex[ stencil_index_1 + i ];        	               
	               }
	             else
	               {
			        #pragma omp parallel for
					#pragma ivdep
					#pragma vector always	            	        
			        for ( int i = 0; i < N; i++ )
			            stencil_complex[ next_stencil_index_temp + i ] *= stencil_complex[ next_stencil_index_temp + i ];        	               
	               }  
	           }
           
             stencil_index = next_stencil_index_temp;
             
             if ( save ) 
                {
                  stencil_map[ stencil_key ] = stencil_index;
                  next_stencil_index = next_stencil_index_temp + N;       
                }  
           }
        else if ( save ) next_stencil_index = next_stencil_index_temp;  
	  }    
	else stencil_index = got->second;
	  
	return stencil_index;    
}



void apply_stencil_fft( double *input__real, double *output__real, int N, int T )
{   
    int stencil_index = find_stencil_index( N, T );
    double complex *stencil__complex = stencil_complex + stencil_index;

    mkl_fft_forward( input__real, input_complex, N );
    
    #pragma omp parallel for
	#pragma ivdep
	#pragma vector always        
    for ( int i = 0; i < N; i++ )
       input_complex[ i ] *= stencil__complex[ i ];

    mkl_fft_backward( input_complex, output__real, N );
}




void apply_stencil_loops( double *input__real, double *output__real, double *tmp__real, int N, int T )
{
    copy_array( input__real, tmp__real, N );   

    for ( int i = 0; i < T; i++ )
       {
        if ( i & 1 )
           {
	        #pragma omp parallel for
	        for ( int i = 1; i < N - 1; i++ ) 
               tmp__real[ i ] = 0.125 * ( output__real[ i + 1 ] - 2.0 * output__real[ i ] + output__real[ i - 1 ] ) + output__real[ i ];
	
	        int i = 0; 
	        tmp__real[ i ] = 0.125 * ( output__real[ i + 1 ] - 2.0 * output__real[ i ] ) + output__real[ i ];
	        int k = N - 1;
	        tmp__real[ k ] = 0.125 * ( - 2.0 * output__real[ k ] + output__real[ k - 1 ] ) + output__real[ k ];           
           }
        else
           {
	        #pragma omp parallel for
	        for ( int i = 1; i < N - 1; i++ ) 
               output__real[ i ] = 0.125 * ( tmp__real[ i + 1 ] - 2.0 * tmp__real[ i ] + tmp__real[ i - 1 ] ) + tmp__real[ i ];
	
	        int i = 0; 
	        output__real[ i ] = 0.125 * ( tmp__real[ i + 1 ] - 2.0 * tmp__real[ i ] ) + tmp__real[ i ];
	        int k = N - 1;
	        output__real[ k ] = 0.125 * ( - 2.0 * tmp__real[ k ] + tmp__real[ k - 1 ] ) + tmp__real[ k ];           
           }           
       }
    
    if ( !( T & 1 ) ) copy_array( tmp__real, output__real, N );
}



void compute_aperiodic_stencil( double *input__real, double *output__real, double *tmp__real, int N, int T )
{
    if ( ( T <= T_base ) || ( ( long int ) N * T <= V_base ) )
       {
         apply_stencil_loops( input__real, output__real, tmp__real, N, T );
         return;
       }

    if ( ( T * max( 4, MIN_NT_RATIO ) ) >= N )
       {
         compute_aperiodic_stencil( input__real, output__real, tmp__real, N, ( T >> 1 ) );       

         if ( T & 1 ) apply_stencil_loops( output__real, input__real, tmp__real, N, 1 );
         else copy_array( output__real, input__real, N );   

         compute_aperiodic_stencil( input__real, output__real, tmp__real, N, ( T >> 1 ) );                
         
         return;
       }

    apply_stencil_fft( input__real, output__real, N, T );

	copy_array( input__real, input__real + N, ( T << 1 ) );           
	copy_array( input__real + ( N - ( T << 1 ) ), input__real + ( N + ( T << 1 ) ), ( T << 1 ) );   
		    
	compute_aperiodic_stencil( input__real + N, output__real + N, tmp__real + N, ( T << 2 ), T );

	copy_array( output__real + N, output__real, T );           
	copy_array( output__real + ( N + ( T << 1 ) + T ), output__real + ( N - T ), T );   	
}


void compute_aperiodic_heat_stencil( double *input__real, double *output__real, double *tmp__real, int N, int T )
{
//    compute_aperiodic_stencil( input__real, output__real, tmp__real, N, T );    

    int t = -1;
    
    while ( T )
      {
        t++;
        
        if ( T & 1 ) 
           {
             compute_aperiodic_stencil( input__real, output__real, tmp__real, N, ( 1 << t ) );
             if ( T >> 1 ) copy_array( output__real, input__real, N );
           }  
        
        T >>= 1;
      }      
}



int initialize( int N, int input_backup )
{
    input_real = ( double * ) malloc( sizeof( double ) * N * 2 );
    if ( input_backup ) input_real_orig = ( double * ) malloc( sizeof( double ) * N );
    output_real = ( double * ) malloc( sizeof( double ) * N * 2 );
    tmp_real = ( double * ) malloc( sizeof( double ) * N * 2 );
    stencil_real = ( double * ) malloc( sizeof( double ) * N );

    next_stencil_index = 0;
    max_stencil_index = STENCIL_SPACE_FACTOR * N;
    stencil_complex = ( double complex * ) malloc( sizeof( double complex ) * N * ( STENCIL_SPACE_FACTOR + 2 ) );
    input_complex = ( double complex * ) malloc( sizeof( double complex ) * N );
    
    if ( ( input_real == NULL ) || ( input_backup && ( input_real_orig == NULL ) ) || ( output_real == NULL ) || ( tmp_real == NULL ) || ( stencil_real == NULL ) 
       || ( input_complex == NULL ) || ( stencil_complex == NULL ) )
        {
          cout << "Error: memory allocation failed!" << endl;
          return 1;
        }
    else cout << "Success: memory allocated!" << endl;        
    
    for ( int i = 0; i < N; ++i ) 
       input_real[ i ] = 1.0 * ( rand( ) % MAX_VAL ); 
       
    return 0;   
}



void destroy( int input_backup )
{
    free( input_real );
    if ( input_backup ) free( input_real_orig );
    free( output_real );    
    free( tmp_real );
    free( stencil_real );
    
    free( stencil_complex );
    free( input_complex );   
}



void verify( double *input__real, double *output__real, int N, int T )
{
    double *tmp2_real = tmp_real + N;

    copy_array( input__real, tmp_real, N );

    for ( int tt = 0; tt < T; ++tt ) 
      {
        #pragma omp parallel for
        for ( int i = 0; i < N; ++i ) 
           {
             if ( i == 0 )
                tmp2_real[ i ] = 0.125 * ( tmp_real[ i + 1 ] - 2.0 * tmp_real[ i ] ) + tmp_real[ i ];
             else if ( i == N - 1 )
                     tmp2_real[ i ] = 0.125 * ( - 2.0 * tmp_real[ i ] + tmp_real[ i - 1 ] ) + tmp_real[ i ]; 
                  else
                     tmp2_real[ i ] = 0.125 * ( tmp_real[ i + 1 ] - 2.0 * tmp_real[ i ] + tmp_real[ i - 1 ] ) + tmp_real[ i ];
           } 

        copy_array( tmp2_real, tmp_real, N );            
      }

    int ttl = 0, cnt = 0; 
      
    for ( int i = 0; i < N; i++ )
      {           
        ttl++;   
        if ( fabs( tmp_real[ i ] - output__real[ i ] ) > 1e-8 )
            cnt++;
      }      

    cout << "Number of incorrect output entries = " << cnt << ", total entries checked = " << ttl << endl;
}


int main( int argc, char *argv[ ] )
{
    int numThreads;

    if ( argc < 3 )
      {
        cout << "Enter: N T ( > 0 ) T_base V_base numThreads verify (0/1)" << endl;
        return 1;
      }

    N_orig = atoi( argv[ 1 ] );
       
    T_orig = atoi( argv[ 2 ] );

    T_base = 128;
    if ( argc > 3 ) T_base = atoi( argv[ 3 ] );

    V_base = 1024;
    if ( argc > 4 ) V_base = atoi( argv[ 4 ] );

    if ( ( N_orig < 1 ) || ( T_orig < 1 ) || ( T_base < 1 ) || ( V_base < 1 ) )
      {
        cout << "Error: N, T, T_base, and V_base must be positive integers." << endl;
        return 2; 
      }


    numThreads = 1;
    if ( argc > 5 )
      {
        numThreads = max( 1, atoi( argv[ 5 ] ) );
        omp_set_num_threads( numThreads );
//        omp_set_dynamic( 0 );
//        omp_set_nested( 1 );        
      }

    int verifyRes = 0;
    if ( argc > 6 ) verifyRes = atoi( argv[ 6 ] );

    int N = N_orig;

    N_THREADS = numThreads;

    if ( initialize( N_orig, verifyRes ) )
      {
        destroy( verifyRes );
        return 1; 
      }
    
    #ifdef USE_PAPI
        papi_init( );
    #endif

    if ( verifyRes ) copy_array( input_real, input_real_orig, N );

    long start = getTime( );
    
#ifdef POLYBENCH
    /* Start timer. */
    polybench_start_instruments;
#endif

    compute_aperiodic_heat_stencil( input_real, output_real, tmp_real, N_orig, T_orig );
        
#ifdef POLYBENCH
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
#endif
    long end = getTime( );
    cout << "N = " << N << ", T = " << T_orig << ", T_BASE = " << T_base << ", V_BASE = " << V_base << ", numThreads = " << numThreads << ", time = " << (end - start) / 1000.0 << endl;

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
        copy_array( input_real_orig, input_real, N );
	
	    long start_iter = getTime( );    
	    verify( input_real, output_real, N_orig, T_orig );
	    long end_iter = getTime( );
	    // cout << "Time (Iter): " << end_iter - start_iter << endl;
      }

    destroy( verifyRes );

    return 0;
}

