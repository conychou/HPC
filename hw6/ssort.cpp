// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 1000000;
  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand()%10;
  }
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);
    // timing
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();
  // sort locally
  std::sort(vec, vec+N);

  /*  if(rank==1) {
        for(int idx=0; idx<N; idx++)
            std::cout<<vec[idx]<<",";
    }*/
  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
    int size = N/p;
    int *splitter = (int*)malloc((p-1)*sizeof(int));
    for(int idx=0; idx<p-1; idx++) {
        splitter[idx] = vec[(idx+1)*size];
    }
   /* if(rank==0) {
        for(int idx=0; idx<p-1; idx++)
            cout<<"sp"<<idx<<":"<<splitter[idx]<<",";
    }*/
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
    int *gathers = NULL;
    if(rank==0) {
        gathers = (int*)malloc((p-1)*p*sizeof(int));   
    }
    MPI_Gather(splitter, p-1, MPI_INT, gathers, p-1, MPI_INT, 0,
           MPI_COMM_WORLD);  
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
    if(rank==0) {
        std::sort(gathers,gathers+(p-1)*p);
        /*
        cout<<endl<<"gather after sort"<<endl;
        for(int idx=0; idx<p*(p-1); idx++)
            cout<<gathers[idx]<<",";
        cout<<endl;
        */
        for(int idx=0; idx<p-1; idx++) {
            splitter[idx] = gathers[(idx+1)*(p-1)];  
        } 
    }
  // root process broadcasts splitters to all other processes
    MPI_Bcast(splitter, p-1, MPI_INT, 0, MPI_COMM_WORLD);
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
    int * scounts = (int *) calloc(sizeof(int), p);  
    int * sdispls = (int *) calloc(sizeof(int), p);
    
    for(int idx=0; idx<p-1; idx++) {
        sdispls[idx+1] = std::lower_bound(vec, vec+N, splitter[idx]) - vec;
        scounts[idx] = sdispls[idx+1]-sdispls[idx]; 
    }
    scounts[p-1] = N-sdispls[p-1]; 
  /*  if(rank==1) {
        for(int idx=0; idx<p; idx++) 
            cout<<idx<<":"<<sdispls[idx]<<","<<scounts[idx]<<endl;
    }*/
    int * rcounts = (int *)calloc(sizeof(int), p);
    int * rdispls = (int *)calloc(sizeof(int), p);

    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
    rdispls[0] = 0;
    for(int i=1;i<p;i++){
        if(rcounts[i] != 0){
            rdispls[i] = rdispls[i-1] + rcounts[i-1];
        }   
    }
    int bucketSize = 0;
    for(int i=0;i<p;i++)
    {
        bucketSize += rcounts[i];
    }
    int * localBucket = (int *)calloc(sizeof(int),bucketSize);
    MPI_Alltoallv(vec,scounts,sdispls,MPI_INT,localBucket,rcounts,rdispls,MPI_INT,MPI_COMM_WORLD);
  // do a local sort of the received data
    sort(localBucket,localBucket+bucketSize);
  // every process writes its result to a file

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }

   FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }
    for(int n = 0; n < bucketSize; ++n)
      fprintf(fd, "  %d\n", localBucket[n]);

    fclose(fd);

    free(scounts);
    free(sdispls);
    free(rcounts);
    free(rdispls);
    free(splitter);
    free(localBucket);
    if(rank==0)
        free(gathers);
  free(vec);
  MPI_Finalize();
  return 0;
}
