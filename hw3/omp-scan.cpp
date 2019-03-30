#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define THREAD_NUM 64
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
/*
        printf("\nprefix1=");
        for(int idx=0; idx<n; idx++)
            printf("%d ",prefix_sum[idx]);
        printf("\n");
*/
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
    if(n == 0)  return;
    omp_set_num_threads(THREAD_NUM);
    long p = THREAD_NUM; 
    long block_size = n/THREAD_NUM;
    long* s = (long*) malloc(THREAD_NUM * sizeof(long));
/*    printf("\nA=");
    for(int idx=0; idx<n; idx++)
        printf("%d ",A[idx]);
    printf("\n");
*/
    #pragma omp parallel shared(p,block_size)
    {

  /*      #pragma omp single
        {
            printf("num_thread =%d\n",omp_get_num_threads());
            printf("p=%d, block_size=%d\n",p,block_size);
        }
*/
        #pragma omp for
        for(long i=0; i<p; i++) {
            s[i] = 0;
            for(long size = 0; size<block_size; size++) {
                s[i] += A[i*block_size+size];
            }
        }  
/* 
        #pragma omp single
        {
            printf("s=");
            for(long idx=0; idx<p; idx++)
                printf("[%ld]=%ld, ",idx,s[idx]);
            printf("\n");
        }
*/
        #pragma omp for
        for(long block = 0; block < p; block++) {
            long sum = 0;
            for(int i=0; i<block; i++)
                sum += s[i];   
            prefix_sum[block*block_size] = sum;
        }
        #pragma omp for 
        for(long block = 0; block < p; block++) {
            long pos = block*block_size;
            for(long idx=1; idx<block_size; idx++) {
                prefix_sum[pos+idx] = prefix_sum[pos+idx-1]+A[pos+idx-1];
            }  
        }
 /*
        #pragma omp single
        {
            printf("\nprefix2=");
            for(int idx=0; idx<n; idx++)
                printf("[%d]=%ld ",idx,prefix_sum[idx]);

            printf("\n");
        }
*/
    } 
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = i;//rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
