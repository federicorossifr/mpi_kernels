#include "mpi.h"
#include <iostream>
using namespace std;

void init_data(float* ptr,size_t size) {
    for(int i = 0; i < size; ++i) {
        ptr[i] = 1;
    }
}

void print_matrix(float* ptr,int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::cout << ptr[i*cols+j] << " ";
        std::cout << std::endl; 
    }
}


void local_gemm(float* a,float* b,int N, int M, int K, int stripe_dim) {
        float* stripe_data = new float[stripe_dim];
        for(int i = 0; i < N; ++i ) {
            for(int j = 0; j < K; ++j) {
                float accum = 0.f;
                for(int k = 0; k < M;++k ) {
                    accum+=a[i*M+k]*b[k*K+j];
                }
                stripe_data[i*K+j] = accum;
            }
        }
		MPI_Send(stripe_data,stripe_dim,MPI_FLOAT,0,201,MPI_COMM_WORLD);
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
        float* a_pin = &a[(rank-1)*M*STRIPE_SIZE];
        local_gemm(a_pin,b,STRIPE_SIZE_ADJ,M,K,stripe_end-stripe_start+1);
	}


	MPI_Finalize();

return 0;
}
