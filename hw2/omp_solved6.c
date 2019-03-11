/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum = 0.0;
//#pragma omp parallel        // parallel in the function and let sum be default shared variable
//tid = omp_get_thread_num();
  #pragma omp parallel for reduction(+:sum)   // parallel for loop
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",omp_get_thread_num(),i);
    }

return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

//#pragma omp parallel shared(sum)          // parallel in function, not here
  sum = dotprod();

printf("Sum = %f\n",sum);

}

