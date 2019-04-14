#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void mmul(double* C, const double* A, const double* B, long N){
  
  #pragma omp parallel for schedule(static) collapse(2)
  for(int row=0; row<N; row++) {
    for(int col=0; col<N; col++) {
        double sum = 0.0f;
        for (long i = 0; i < N; i++) 
            sum += A[row*N+i] * B[col+i*N];
        C[row*N+col] = sum;
    }
  }
}

void Check_CUDA_Error(void){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: : %s\n", cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 32   //32*32 = 1024

__global__ void mmul_kernel(double* C, double* A, const double* B, long N){
  
  int ROW = (blockIdx.y) * blockDim.y + threadIdx.y;
  int COL = (blockIdx.x) * blockDim.x + threadIdx.x;
  double tmpSum = 0.0f;
  if(ROW < N && COL < N) {
    for(int i=0; i<N; i++) {
        tmpSum += A[ROW*N+i]*B[i*N+COL];
    }
  }
  C[ROW*N + COL] = tmpSum;
}

int main() {
  long N = (1UL<<11);   // 2048 * 2048
  double *A, *B, *C, *C_ref;
  cudaMallocHost((void**)&A, N*N* sizeof(double));
  cudaMallocHost((void**)&B, N*N* sizeof(double));
  cudaMallocHost((void**)&C, N*N* sizeof(double));
  cudaMallocHost((void**)&C_ref, N*N* sizeof(double));
  #pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
        A[i*N+j] = 1.0/(i+1);
        B[i*N+j] = 2.0/(i+1);
    }
  }
  double tt = omp_get_wtime();
  mmul(C_ref, A, B, N);
  printf("CPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, N*N*sizeof(double));
  cudaMalloc(&B_d, N*N*sizeof(double));
  cudaMalloc(&C_d, N*N*sizeof(double));

  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(B_d, B, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  int Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //printf("N=%ld, Nb=%ld\n",N,Nb);
  dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grids(Nb,Nb);
  mmul_kernel<<<Grids,Blocks>>>(C_d, A_d, B_d, N);
  Check_CUDA_Error();

  cudaMemcpyAsync(C, C_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double sumdiff = 0.0f; 
  for(int row=0; row<N; row++) {
    for(int col=0; col<N; col++) {
        double diff = fabs(C[row*N+col]-C_ref[row*N+col]);
        sumdiff += diff;
        //printf("Error[%d][%d] = %f, %f %f\n", row, col, 
          //      fabs(C[row*N+col]-C_ref[row*N+col]),C[row*N+col],C_ref[row*N+col]);
    }
  }
  printf("sumError = %f\n", sumdiff);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  cudaFreeHost(C_ref);
  
  return 0;
}
