/*
 *  Run on CIMS crunchy5
 */
#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include <memory>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
extern const bool parallelism_enabled = true;
#else
extern const bool parallelism_enabled = false;
#endif

#define MAX_ITER 1000
#define BLOCK_SIZE  16
#define RESIDUAL 1e3

using namespace std;

double measure_error(vector<double> &new_x, vector<double> &x,int thread_count) {
    int n=x.size();
    double error=0.0;
    #pragma omp parallel for num_threads(thread_count) reduction(+:error) if(parallelism_enabled)
    for(int i=0; i<n; i++) {
        error += abs(new_x[i]-x[i]);
    }
    return error;
}
void printx(string s, vector<double> &x, int n) {
    cout<<s<<"=[";
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) 
            cout<<x[i*n+j]<<",";
        cout<<endl;
    }
    cout<<"]"<<endl;
}
void jacobi() {
    
    Timer t;
    cout<<__func__<<":"<<endl;
    for(int thread_count = 1; thread_count <= 32; thread_count = thread_count<<1) {
        cout<<"# of THREAD = "<<thread_count<<endl; 
        for(int N=BLOCK_SIZE; N<=2048; N=N<<1) {
            int n=N+1;
            double first_err = DBL_MAX;
            vector<double> x(n*n,0.0);
            double hsquare= pow(1.0/n,2);
            int iter=0;
            t.tic();
            for(iter=0; iter<MAX_ITER; iter++) {
                vector<double> new_x(n*n, 0);
                #pragma omp parallel for num_threads(thread_count) if(parallelism_enabled)
                for(int i=1; i<n-1; i++) {
                    for(int j=1; j<n-1; j++) {
                        new_x[i*n+j] = 0.25*(hsquare + x[(i-1)*n+j]+x[(i+1)*n+j]+x[i*n+(j-1)]+x[i*n+(j+1)]);
                    }
                }
                if(first_err==DBL_MAX)  {
                    first_err = measure_error(new_x, x,thread_count);
                    //cout<<"first_err="<<first_err<<endl;
                } else {
                    if(!first_err) break;
                    double new_error = measure_error(new_x, x, thread_count);
                    double residual = new_error/first_err;
                    if(new_error == 0 || residual >= RESIDUAL) break;
                    x = new_x;
                   // cout<<"iter "<<iter<<": "<<residual<<endl;
                }
            }
            double time = t.toc();
            cout<<"dimension:"<<N<<" time:"<<time<<endl;
            //printx("end x",x,n);
        }
    }
}

int main() {
    jacobi();
    return 0;
}
