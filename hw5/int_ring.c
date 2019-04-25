#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    if( argc != 2) {
        printf("Usage: ./int_ring (number of iteration)\n");
        exit(1);
    }
    int rank, num_p, iter = 0, message, N;
    MPI_Status status;
    N = atoi(argv[1]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    int prev = (rank+num_p-1)%num_p;
    int next = (rank+1)%num_p;

    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();
    if(rank==0) {
        printf("N=%d\n",N);
        message = 42;
        MPI_Send(&message,1,MPI_INT,next,0,MPI_COMM_WORLD);
//        printf("process %d send message %d to %d\n", rank, message,next);
        message = 0;
        iter++;
    }

    while(iter<N) {
  //      printf("process %d cur message %d, iter %d\n", rank, message, next, iter);
        MPI_Recv(&message,1,MPI_INT,prev,0,MPI_COMM_WORLD,&status);
    //    printf("process %d receive message %d from %d\n", rank, message, prev);
        MPI_Send(&message,1,MPI_INT,next,0,MPI_COMM_WORLD);
        iter++;
        message = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if(rank==0) {
        MPI_Recv(&message,1,MPI_INT,prev,0,MPI_COMM_WORLD,&status);
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize( );
    return 0;
}
