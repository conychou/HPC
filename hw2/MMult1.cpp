// g++ -fopenmp -std=c++11 -O3 -march=native MMult1.cpp && ./a.out
// Run on CIMS crunchy5
//
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include <string>

#define BLOCK_SIZE 16

void printMatrix(std::string s, int m, int n, double *arr) {
    std::cout<<s<<"=["<<std::endl;
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
                std::cout<<arr[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"]"<<std::endl;
}
// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long i = 0; i < m; i++) {
     for (long j = 0; j < n; j++) {
        for (long p = 0; p < k; p++) {
            double A_ip = a[i*k+p];
            double B_pj = b[p*n+j];
            double C_ij = c[i*n+j];
            C_ij = C_ij + A_ip * B_pj;
            c[i*n+j] = C_ij;
      }
    }
  }
  //  printMatrix("c_ref",m,n,c);
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  
    for (long i = 0; i < m; i+=BLOCK_SIZE) {
        for (long j = 0; j < n; j+=BLOCK_SIZE) {
            for (long p = 0; p < k; p+=BLOCK_SIZE) {
                    for(long i0 = i; i0 < std::min(i+BLOCK_SIZE, m); i0++) {
                        for(long j0 = j; j0 < std::min(j+BLOCK_SIZE, n); j0++) {
                            for(long p0 = p; p0 < std::min(p+BLOCK_SIZE, k); p0++) {
                                double A_ip = a[i0*k+p0];
                                double B_pj = b[p0*n+j0];
                                double C_ij = c[i0*n+j0];
                                C_ij = C_ij + A_ip * B_pj;
                                c[i0*n+j0] = C_ij;

                            }   
                        }
                    }
            }
        }   
    }
}

void MMult2(long m, long n, long k, double *a, double *b, double *c) {
    
  #pragma prallel for collapse(2)
    for (long i = 0; i < m; i+=BLOCK_SIZE) {
        for (long j = 0; j < n; j+=BLOCK_SIZE) {
            for (long p = 0; p < k; p+=BLOCK_SIZE) {
                    for(long i0 = i; i0 < std::min(i+BLOCK_SIZE, m); i0++) {
                        for(long j0 = j; j0 < std::min(j+BLOCK_SIZE, n); j0++) {
                            for(long p0 = p; p0 < std::min(p+BLOCK_SIZE, k); p0++) {
                                double A_ip = a[i0*k+p0];
                                double B_pj = b[p0*n+j0];
                                double C_ij = c[i0*n+j0];
                                C_ij = C_ij + A_ip * B_pj;
                                c[i0*n+j0] = C_ij;

                            }   
                        }
                    }
            }
        }
    }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf("                     Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 10;//1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c1 = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c2 = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c1[i] = 0;
    for (long i = 0; i < m*n; i++) c2[i] = 0;

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }
    double time0 = t.toc();

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c1);
    }
    double time1 = t.toc();
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult2(m, n, k, a, b, c2);
    }
    double time2 = t.toc();
   // printMatrix("A",k,m,a);
   // printMatrix("B",n,k,b);
   // printMatrix("C",n,m,c);
   // printMatrix("C_ref",n,m,c_ref);
    double flops0 = NREPEATS*p*p*p*2/time0/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth0 = NREPEATS*p*p*p*4*sizeof(double)/time0/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double flops1 = NREPEATS*p*p*p*2/time1/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth1 = NREPEATS*p*p*p*4*sizeof(double)/time1/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double flops2 = NREPEATS*p*p*p*2/time2/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth2 = NREPEATS*p*p*p*4*sizeof(double)/time2/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double max_err1 = 0;
    double max_err2 = 0;
    for (long i = 0; i < m*n; i++) max_err1 = std::max(max_err1, fabs(c1[i] - c_ref[i]));
    for (long i = 0; i < m*n; i++) max_err2 = std::max(max_err2, fabs(c2[i] - c_ref[i]));
    printf("MMult0             : %10d %10f %10f %10f\n", p, time0, flops0, bandwidth0);
    printf("MMult1 blocking    : %10d %10f %10f %10f %10e\n", p, time1, flops1, bandwidth1, max_err1);
    printf("MMult2 blocking+omp: %10d %10f %10f %10f %10e\n\n", p, time2, flops2, bandwidth2, max_err2);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c_ref);
    aligned_free(c1);
    aligned_free(c2);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
