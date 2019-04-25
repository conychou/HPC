#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    if( argc != 2) {
        printf("Usage: ./int_ring (number of iteration)\n");
        exit(1);
    }
    int rank, num_p, iter = 0, N;
    MPI_Status status;
    N = atoi(argv[1]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    int prev = (rank+num_p-1)%num_p;
    int next = (rank+1)%num_p;
    int number_amount = 0;
    int size = 2*1024*1024/sizeof(int);     //2MB
    int *message = (int*)malloc(2*1024*1024);

    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();
    if(rank==0) {
        printf("N=%d\n",N);
        MPI_Send(message,size,MPI_INT,next,0,MPI_COMM_WORLD);
      //  printf("process %d send message %d to %d\n", rank, message,next);
        iter++;
    }

    while(iter<N) {
       // printf("process %d cur message %d, iter %d\n", rank, message, next, iter);
        MPI_Recv(message,size,MPI_INT,prev,0,MPI_COMM_WORLD,&status);
        //MPI_Get_count(&status, MPI_INT, &number_amount);
       // printf("process %d receive #%d data from %d\n", rank, number_amount, prev);
        MPI_Send(message,size,MPI_INT,next,0,MPI_COMM_WORLD);
        iter++;
    }
    if(rank==0) {
        MPI_Recv(message,size,MPI_INT,prev,0,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if(rank==0) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    free(message);
    MPI_Finalize( );
    return 0;
}
