#include "mpi.h"
#include <iostream>
using namespace std;

void init_data(float* ptr,size_t size) {
    for(int i = 0; i < size; ++i) {
        ptr[i] = i;
    }
}

int main(int argc, char* argv[]) {
	int rank=0,size=0,len=0,message=104;
    int dim = atoi(argv[1]);
	char version[MPI_MAX_LIBRARY_VERSION_STRING];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_library_version(version, &len);

    float* a = new float[dim]; init_data(a,dim);
    float* b = new float[dim]; init_data(b,dim);
    size_t slaves_no = size - 1;
    size_t BLOCK_SIZE = dim/slaves_no;
    size_t OF_BLOCK   = dim%slaves_no;
	if(rank == 0) {
		//std::cout << "I am master\n";
        //std::cout << "INIT -  BLOCK SIZE: " << BLOCK_SIZE << std::endl;
        //std::cout << "INIT -  BLOCK OF: " << OF_BLOCK << std::endl;
        float accum_in = 0.f;
        float glob_accum = 0.0f;
        MPI_Status status; 
		for(int i = 1; i < size; ++i) {
            MPI_Recv(&accum_in,1,MPI_FLOAT,MPI_ANY_SOURCE,201,MPI_COMM_WORLD,&status);
            //std::cout << "MASTER: received local accum from " << status.MPI_SOURCE << " " << accum_in << std::endl;
            glob_accum+= accum_in;
        }
        std::cout << glob_accum << std::endl;
	} else {
        int block_start = (rank-1)*BLOCK_SIZE;
        int block_end = block_start+BLOCK_SIZE-1 + ((rank == slaves_no))*OF_BLOCK;        
		//std::cout << "I am slave " << rank << " range: " << block_start << ":" << block_end << std::endl;
        float accum = 0.f;
        for(int i = block_start; i <= block_end; ++i) {
            accum+=a[i]*b[i];
        }
		MPI_Send(&accum,1,MPI_FLOAT,0,201,MPI_COMM_WORLD);

	}


	MPI_Finalize();

return 0;
}
