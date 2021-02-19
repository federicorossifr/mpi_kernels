#include "mpi.h"
#include <iostream>
using namespace std;

void init_data(float* ptr,size_t size) {
    for(int i = 0; i < size; ++i) {
        ptr[i] = 1;
    }
}

void print_matrix(float* ptr,int rows, int cols) {
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << ptr[i*cols+j] << " ";
        }
        std::cout << std::endl;
        
    }
    
}
/* NxM * MxK GEMM */
int main(int argc, char* argv[]) {
	int rank=0,size=0,len=0,message=104;
    int N = atoi(argv[1]), M = atoi(argv[2]), K = atoi(argv[3]);
	char version[MPI_MAX_LIBRARY_VERSION_STRING];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_library_version(version, &len);

    float* a = new float[N*M]; init_data(a,N*M);
    float* b = new float[M*K]; init_data(b,M*K);
    float* c = new float[N*K];

    int slaves_no = size - 1;
    int STRIPE_SIZE = N/slaves_no;
    int STRIPE_OF   = N%slaves_no;
	if(rank == 0) {

		//std::cout << "I am master\n";
        //std::cout << "INIT -  BLOCK SIZE: " << BLOCK_SIZE << std::endl;
        //std::cout << "INIT -  BLOCK OF: " << OF_BLOCK << std::endl;
        float accum_in = 0.f;
        float glob_accum = 0.0f;

		for(int i = 1; i < size; ++i) {
            // Find out receiver

            MPI_Status status; 
            MPI_Probe(MPI_ANY_SOURCE,201,MPI_COMM_WORLD,&status);

            //Get stripe idx 
            int idx = status.MPI_SOURCE;
            int recv_stripe_size = 0;
            MPI_Get_count(&status, MPI_INT, &recv_stripe_size);

            //Set dst ptr
            //std::cout << "Receiving: " << recv_stripe_size << std::endl;
            //std::cout << "From: " << idx << std::endl;
            //std::cout << "Pin position: " << (idx-1)*K*STRIPE_SIZE << std::endl;
            float* ptr = &(c[(idx-1)*K*STRIPE_SIZE]);
            //Receive and store computed GEMM stripe

            MPI_Recv(ptr,recv_stripe_size,MPI_FLOAT,idx,201,MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        print_matrix(c,N,K);
	} else {
        int stripe_start = (rank-1)*STRIPE_SIZE*K;
        int STRIPE_SIZE_ADJ = STRIPE_SIZE+((rank == slaves_no))*STRIPE_OF;
        int stripe_end = stripe_start+(STRIPE_SIZE_ADJ)*K-1;        
		//std::cout << "I am slave " << rank << " range: " << stripe_start << ":" << stripe_end << std::endl;
        float* stripe_data = new float[stripe_end-stripe_start+1];
        float* a_pin = &a[(rank-1)*M*STRIPE_SIZE];
        for(int i = 0; i < STRIPE_SIZE_ADJ; ++i ) {
            for(int j = 0; j < K; ++j) {
                float accum = 0.f;
                for(int k = 0; k < M;++k ) {
                    accum+=a_pin[i*M+k]*b[k*K+j];
                }
                stripe_data[i*K+j] = accum;
            }
        }
		MPI_Send(stripe_data,stripe_end-stripe_start+1,MPI_FLOAT,0,201,MPI_COMM_WORLD);

	}


	MPI_Finalize();

return 0;
}
