/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total = 0.0;

/*** Spawn parallel region ***/
#pragma omp parallel 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  //total = 0.0;        // initialize at first, or it will overwrite the value by different threads
  #pragma omp for schedule(dynamic,10) reduction(+:total)   //let every thread has its own copy total and combine in the end
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  #pragma omp barrier
    #pragma omp single      // print once 
    printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
