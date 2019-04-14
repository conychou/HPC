#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void Check_CUDA_Error(void){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: : %s\n", cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 32   //32*32 = 1024
#define MAX_ITER 64

__global__ void mmul_kernel(double* new_X, double* X, long N){
  
  int ROW = (blockIdx.y) * blockDim.y + threadIdx.y;
  int COL = (blockIdx.x) * blockDim.x + threadIdx.x;
  double hsquare= pow(1.0/N,2);
  double tmpSum = 0.0f;
  if(ROW > 0 && COL> 0 && ROW < N-1 && COL < N-1) 
    tmpSum = 0.25*(hsquare+X[ROW*N+(COL-1)]+X[ROW*N+(COL+1)]+X[(ROW-1)*N+COL]+X[(ROW+1)*N+COL]);
  new_X[ROW*N + COL] = tmpSum;   
}

int main() {
  long N = (1UL<<11)+1;   // 2048 * 2048
  double *X, *new_X;
  cudaMallocHost((void**)&X, N*N* sizeof(double));
  cudaMallocHost((void**)&new_X, N*N* sizeof(double));
  #pragma omp parallel for schedule(static) collapse(2)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
        X[i*N+j] = 0.0;
        new_X[i*N+j] = 0.0;
    }
  }

  double *X_d, *new_X_d;
  cudaMalloc(&X_d, N*N*sizeof(double));
  cudaMalloc(&new_X_d, N*N*sizeof(double));

  cudaMemcpyAsync(X_d, X, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(new_X_d, new_X_d, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double tt = omp_get_wtime();

  int Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //printf("N=%ld, Nb=%ld\n",N,Nb);
  dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 Grids(Nb,Nb);
  for (int iter=0; iter<=MAX_ITER; iter++){
    if(iter%2==0)  
        mmul_kernel<<<Grids,Blocks>>>(new_X_d, X_d, N);
    else
        mmul_kernel<<<Grids,Blocks>>>(X_d, new_X_d, N);
        
    Check_CUDA_Error();
  }

  cudaMemcpyAsync(new_X, new_X_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
/*  for(int row=0; row<N; row++) {
    for(int col=0; col<N; col++) {
        printf("new_X[%d][%d] = %f\n", row, col, new_X[row*N+col]); 
    }
  }
*/
  cudaFree(X_d);
  cudaFree(new_X_d);
  cudaFreeHost(X);
  cudaFreeHost(new_X);

  return 0;
}
