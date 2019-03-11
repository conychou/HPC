CC       = g++
CFLAGS  = -fopenmp -std=c++11 -O3 -march=native
TARGET	 = MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

all: $(TARGET)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< 
%: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o $(TARGET)
