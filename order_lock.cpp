#include "mpi.h"
#include <iostream>
using namespace std;
int main(int argc, char* argv[]) {
	int rank=0,size=0,len=0,message=104;
	char version[MPI_MAX_LIBRARY_VERSION_STRING];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_library_version(version, &len);
	if(rank == 0) {
		std::cout << "I am master\n";
		for(int i = 1; i < size; ++i) {
			MPI_Send(&message,1,MPI_INT,i,201,MPI_COMM_WORLD);
			MPI_Recv(&message,1,MPI_INT,i,201,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	} else {
		MPI_Recv(&message,1,MPI_INT,0,201,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		std::cout << "I am slave " << rank << std::endl;
		MPI_Send(&message,1,MPI_INT,0,201,MPI_COMM_WORLD);
	}


	MPI_Finalize();

return 0;
}
