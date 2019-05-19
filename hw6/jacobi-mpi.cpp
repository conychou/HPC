/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
using namespace std;
/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **lu, int lN, double invhsq){
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++){
  	for (j = 1; j <= lN; j++){
    	tmp = ((4.0*lu[i][j] - lu[i-1][j] - lu[i][j-1] - lu[i+1][j] - lu[i][j+1]) * invhsq - 1);
		lres += tmp * tmp;
	}
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}
/*******************************************************************/
/*             Update Bounds of subdomain with me process          */
/*******************************************************************/

void updateBound(double** x, int neighBor[], MPI_Comm comm2d, MPI_Datatype column_type, int rank, int lN) {

   int S = 0, E = 1, Nor = 2, W = 3;
   int flag;
   MPI_Status status;
   MPI_Request request;
   /****************** North/South communication ******************/
#if 1
   flag = 1;
   /* Send my boundary to North and receive from South */
    if(neighBor[Nor] >= 0)	MPI_Send(&(x[lN][1]), lN, MPI_DOUBLE, neighBor[Nor], flag, comm2d);
    if(neighBor[S] >= 0)		MPI_Recv(&(x[0][1]), lN, MPI_DOUBLE, neighBor[S], flag, comm2d, &status);
   /* Send my boundary to South and receive from North */
    if(neighBor[S] >= 0)		MPI_Send(&(x[1][1]), lN, MPI_DOUBLE, neighBor[S], flag, comm2d);
    if(neighBor[Nor] >= 0)	MPI_Recv(&(x[lN+1][1]), lN, MPI_DOUBLE, neighBor[Nor], flag, comm2d, &status);
#endif
   /****************** East/West communication ********************/
#if 1
   flag = 2;
   // Send my boundary to East and receive from West //
   if(neighBor[E]>=0)	MPI_Send(&(x[1][lN]), 1, column_type, neighBor[E], flag, comm2d); 
   if(neighBor[W] >= 0)		MPI_Recv(&(x[1][0]), 1, column_type, neighBor[W], flag, comm2d, &status);

   // Send my boundary to West and receive from East //
   if(neighBor[W]>=0)	MPI_Send(&(x[1][1]), 1, column_type, neighBor[W], flag, comm2d); 
   if(neighBor[E] >= 0)		MPI_Recv(&(x[1][lN+1]), 1, column_type, neighBor[E], flag, comm2d, &status);
#endif
}

int main(int argc, char * argv[]){
  int rank, i, j, num_p, N, lN, domain_p, iter, max_iters;
  MPI_Comm comm, comm2d;
  int dims[2];
  int periods[2];
  MPI_Datatype column_type;
  int S = 0, E = 1, Nor = 2, W = 3;
  int neighBor[4];
  int *xs, *ys, *xe, *ye;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_p);
  domain_p = sqrt(num_p);
  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, num_p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / domain_p;
  if ((N % domain_p != 0) && rank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  double tt = MPI_Wtime();

   /* Allocate coordinates of processes */
   xs = (int*) malloc(num_p*sizeof(int));
   xe = (int*) malloc(num_p*sizeof(int));
   ys = (int*) malloc(num_p*sizeof(int));
   ye = (int*) malloc(num_p*sizeof(int));
 
   /* Create 2D cartesian grid */
   periods[0] = 0;
   periods[1] = 0;
   /* Number of dimensions */
   int ndims = 2;
   /* Invert (Ox,Oy) classic convention */
   dims[0] = domain_p;//y_domains;
   dims[1] = domain_p;//x_domains;
   MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &comm2d);
 
   /* Identify neighBors */
   neighBor[0] = MPI_PROC_NULL;
   neighBor[1] = MPI_PROC_NULL;
   neighBor[2] = MPI_PROC_NULL;
   neighBor[3] = MPI_PROC_NULL;
  
   /* Left/West and Right/East neighBors */
   MPI_Cart_shift(comm2d, 0, 1, &neighBor[W], &neighBor[E]);
   /* Bottom/South and Upper/North neighBors */
   MPI_Cart_shift(comm2d, 1, 1, &neighBor[S], &neighBor[Nor]);
  
	for(int idx=0; idx<4; idx++)
		cout<<rank<<": neighBor["<<idx<<"]="<<neighBor[idx]<<endl; 
   /* Create column data type to communicate with East and West neighBors */
   MPI_Type_vector(lN, 1, lN+2, MPI_DOUBLE, &column_type);	//****cony
   MPI_Type_commit(&column_type);

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double ** lu    = (double **) calloc(sizeof(double*), lN + 2);
  double ** lunew = (double **) calloc(sizeof(double*), lN + 2);
  double ** lutemp;
  for (i = 0; i <= lN+1; i++){
	lu[i] = (double *) calloc(sizeof(double), lN+2);
	lunew[i] = (double *) calloc(sizeof(double), lN+2);
  }
  
  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  // initial residual //
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    // Jacobi step for local points //
  //  for (i = 0; i <= lN+1; i++){
    //	for (j = 0; j <= lN+1; j++){
    for (i = 1; i <= lN; i++){
    	for (j = 1; j <= lN; j++){
      		lunew[i][j]  = 0.25 * (hsq + lu[i-1][j] + lu[i][j-1] + lu[i+1][j] + lu[i][j+1]);
		}
	}

    // communicate ghost values //
    updateBound(lunew, neighBor, comm2d, column_type, rank, lN);
/*
if(rank==3) {
  for (i = 0; i <= lN+1; i++){
  	for (j = 0; j <= lN+1; j++)
  		cout<<lunew[i][j]<<",";
	cout<<endl;
  }
}
*/
    // copy newu to u using pointer flipping //
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == rank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(xs);
  free(ys);
  free(xe);
  free(ye);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
