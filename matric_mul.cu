#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <cstdlib>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
const int BLOCKSIZE = 32;
const int TM = 8;
const int TN = 8;
const int BK = 8;
const int BM = 64;
const int BN = 64;

void checkCudaError(cudaError_t err, const char* action) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during " << action << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

void generateRandomMarix(float* array, int size, int seed){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i = 0; i < size; i++){
        array[i] = dis(gen);
    }

    // for(int i = 0;i < size;i++){
    //     std::cout << array[i] << std::endl;
    // }
}


__global__ void hello(){
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}


__global__ void matrix_multiply(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){
    // this is the function/kernel that will run on every thread. ssuming A is of size M x K and B is of size K x N
    // NAIVE IMPLEMENTATION:

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if ((x < M) && (y < N)){
        // printf("something is happening %u and %u\n", x, y);
        float temp = 0.0;
        for (int i = 0;i < K; i++){
            temp += G_A[x*K + i] * G_B[i*N + y];
        }

        G_C[x*N + y] = alpha * temp + beta * G_C[x*N + y];
    }


}

__global__ void matrix_multiply_v2(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){
    // this is the function/kernel that will run on every thread. ssuming A is of size M x K and B is of size K x N
    // OPTIMIZATION - 1: Using Global Memory Coalescing

    int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);


    if ((x < M) && (y < N)){
        // printf("something is happening %u and %u\n", x, y);
        float temp = 0.0;
        for (int i = 0;i < K; i++){
            temp += G_A[x*K + i] * G_B[i*N + y];
        }

        G_C[x*N + y] = alpha * temp + beta * G_C[x*N + y];
    }

}


__global__ void matrix_multiply_v3(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){

    __shared__ float As[BLOCKSIZE*BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];

    G_A += blockIdx.x * BLOCKSIZE * K;
    G_B += blockIdx.y * BLOCKSIZE;
    G_C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    float temp = 0.0;

    for (int i = 0; i < K; i += BLOCKSIZE){
        As[threadRow * BLOCKSIZE + threadCol] = G_A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = G_B[threadRow * N + threadCol];

        __syncthreads();

        G_A += BLOCKSIZE;
        G_B += BLOCKSIZE * N;

        for (int j = 0; j < BLOCKSIZE; j++){
            temp += As[threadRow * BLOCKSIZE + j] * Bs[j * BLOCKSIZE + threadCol];
        }

        __syncthreads();
    }

    G_C[threadRow * N + threadCol] = alpha * temp + beta * G_C[threadRow * N + threadCol];


}


__global__ void matrix_multiply_v4(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){

    __shared__ float As[64*8];
    __shared__ float Bs[64*8];

    
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    
    
    G_A += cRow*BM*K;
    G_B += cCol*BN;
    G_C += cRow*BM*N + cCol*BN;
    

    float threadArray[TM] = {0.0};
    
    int colA = threadIdx.x % BK;
    int rowA = threadIdx.x / BK;
    int colB = threadIdx.x % BN;
    int rowB = threadIdx.x / BN;

    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    for (int i = 0; i<K; i+=BK){
        
        As[rowA*BK + colA] = G_A[rowA*K + colA];
        Bs[rowB*BN + colB] = G_B[rowB*N + colB];

        __syncthreads();

        G_A += BK;
        G_B += BK*N;
        // printf("happening here");
        
        
        for (int j = 0; j < BK; ++j){
            float Btemp = Bs[j*BN + threadCol];

            for (int k = 0; k < TM; ++k){

                threadArray[k] += As[threadRow*BK*TM + k*BK + j] * Btemp ;

            }
        }
        __syncthreads();

    }
    
    for (int i = 0; i < TM; ++i){
        G_C[(threadRow*TM + i)*N + threadCol] = alpha*threadArray[i] + beta * G_C[(threadRow*TM + i)*N + threadCol];
    }
}


__global__ void matrix_multiply_v5(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){

    __shared__ float As[64*8];
    __shared__ float Bs[64*8];

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    G_A += cRow*BM*K;
    G_B += cCol*BN;
    G_C += cRow*BM*N + cCol*BN;

    float gridRegister[TM * TN] = {0.0};

    float mRegister[TM] = {0.0};
    float nResgiter[TN] = {0.0};

    int strideA = 8;
    int strideB = 1;

    int innerRowA = threadIdx.x / BK;
    int innnerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    int threadRow = threadIdx.x / TN;
    int threadCol = threadIdx.x % TN;

    for (int i = 0; i < K; i+=BK){

        for (int i_a=0;i_a < BM; i_a += strideA){
            As[(innerRowA + i_a) * BK + innnerColA] = G_A[(innerRowA + i_a) * K + innnerColA];
        }

        for (int i_b=0;i_b < BK; i_b += strideB){
            Bs[(innerRowB + i_b) * BN + innerColB] = G_B[(innerRowB + i_b) * N + innerColB];
        }

        __syncthreads();

        G_A += BK;
        G_B += BK*N;

        for (int j = 0; j < BK; j++){

            for (int j_a = 0; j_a < TM; j_a++){
                mRegister[j_a] = As[(threadRow*TM + j_a)*BK + j];
            }

            for (int j_b = 0;j_b < TN;j_b++){
                nResgiter[j_b] = Bs[j*BN + threadCol*TN + j_b];
            }

            for (int a_i = 0;a_i < TM;a_i++){
                for (int b_i = 0;b_i < TN;b_i++){
                    gridRegister[a_i*TN + b_i] += mRegister[a_i]*nResgiter[b_i];
                }
            }

        }

        __syncthreads();

    }

    for (int i = 0; i < TM; ++i){
        for (int j = 0; j < TN; ++j){
            G_C[(threadRow*TM + i)*N + (threadCol*TN + j)] = alpha*gridRegister[i*TN + j] + beta * G_C[(threadRow*TM + i)*N + (threadCol*TN + j)];
        }
        
    }
  
}


__global__ void matrix_multiply_v6(float* G_A, float* G_B, float* G_C, float alpha, float beta, int M, int K, int N){

    __shared__ float As[128*8];
    __shared__ float Bs[128*8];

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    G_A += cRow*2*BM*K;
    G_B += cCol*2*BN;
    G_C += cRow*2*BM*N + cCol*2*BN;

    float gridRegister[TM * TN] = {0.0};

    float mRegister[TM] = {0.0};
    float nResgiter[TN] = {0.0};


    int innerRowA = threadIdx.x / (BK/4);
    int innerColA = threadIdx.x % (BK/4);
    int innerRowB = threadIdx.x / ((2*BN) / 4);
    int innerColB = threadIdx.x % ((2*BN) / 4);

    int threadRow = threadIdx.x / 16;
    int threadCol = threadIdx.x % 16;

    for (int i = 0; i < K; i+=BK){

        float4 temp = reinterpret_cast<float4* >(&G_A[innerRowA*K + innerColA*4])[0];
        As[(innerColA*4 + 0)*2*BM + innerRowA] = temp.x;
        As[(innerColA*4 + 1)*2*BM + innerRowA] = temp.y;
        As[(innerColA*4 + 2)*2*BM + innerRowA] = temp.z;
        As[(innerColA*4 + 3)*2*BM + innerRowA] = temp.w;

        reinterpret_cast<float4* >(&Bs[innerRowB*2*BN + innerColB * 4])[0] = reinterpret_cast<float4* >(&G_B[innerRowB*N + innerColB*4])[0];


        __syncthreads();

        G_A += BK;
        G_B += BK*N;

        for (int j = 0; j < BK; j++){

            for (int j_a = 0; j_a < TM; j_a++){
                // mRegister[j_a] = As[(threadRow*TM + j_a)*BK + j];
                mRegister[j_a] = As[j*2*BM + threadRow*TM + j_a];
            }

            for (int j_b = 0;j_b < TN;j_b++){
                nResgiter[j_b] = Bs[j*2*BN + threadCol*TN + j_b];
            }

            for (int a_i = 0;a_i < TM;a_i++){
                for (int b_i = 0;b_i < TN;b_i++){
                    gridRegister[a_i*TN + b_i] += mRegister[a_i]*nResgiter[b_i];
                }
            }

        }

        __syncthreads();

    }

    for (int i = 0; i < TM; ++i){
        for (int j = 0; j < TN; j+=4){
            float4 temp = reinterpret_cast<float4* >(&G_C[(threadRow*TM + i)*N + (threadCol*TN + j)])[0];
            temp.x = alpha*gridRegister[i*TN + j + 0] + beta*temp.x;
            temp.y = alpha*gridRegister[i*TN + j + 1] + beta*temp.y;
            temp.z = alpha*gridRegister[i*TN + j + 2] + beta*temp.z;
            temp.w = alpha*gridRegister[i*TN + j + 3] + beta*temp.w;

            reinterpret_cast<float4* >(&G_C[(threadRow*TM + i)*N + (threadCol*TN + j)])[0] = temp;
        }   
    }
    // for (int i = 0; i < TM; ++i){
    //     for (int j = 0; j < TN; ++j){
    //         G_C[(threadRow*TM + i)*N + (threadCol*TN + j)] = alpha*gridRegister[i*TN + j] + beta * G_C[(threadRow*TM + i)*N + (threadCol*TN + j)];
    //     }
        
    // }
  
}


int main(int c, char *arg[]){

    // hello<<<3, 2>>>();
    // cudaDeviceSynchronize();
    // // cudaError_t err = cudaDeviceSynchronize();
    // // checkCudaError(err, "cudaDeviceSynchronize");

    // // // Reset the device to flush the output buffer
    // // err = cudaDeviceReset();
    // // checkCudaError(err, "cudaDeviceReset");

    // INITIALIZE 2 random matrices of size 4096**2
    
    char* imp = arg[1];

    int M = 4096;
    int K = 4096;
    int N = 4096;
    float alpha = 0.2;
    float beta = 0.41;

    int size_a = M * N;
    int size_b = N * K;
    int size_c = M * N;

    float* A = new float[size_a];
    float* B = new float[size_b];
    float* C = new float[size_c];

    for (int i = 0; i < size_c; i++){
        C[i] = 0.0;
    }

    generateRandomMarix(A, size_a, 11);
    generateRandomMarix(B, size_b, 21);

    // Device points (on GPU)

    float* D_A;
    float* D_B;
    float* D_C;
    
    // transfer the data from host to the GPU device

    cudaError_t err = cudaMalloc(&D_A, sizeof(float)*size_a);
    checkCudaError(err, "cudaMalloc A");

    err = cudaMalloc(&D_B, sizeof(float)*size_b);
    checkCudaError(err, "cudaMalloc B");

    err = cudaMalloc(&D_C, sizeof(float) * size_c);
    checkCudaError(err, "cudaMalloc C");


    err = cudaMemcpy(D_A, A, sizeof(float)*size_a, cudaMemcpyHostToDevice);
    checkCudaError(err, "copy host A from GPU");

    err = cudaMemcpy(D_B, B, sizeof(float)*size_b, cudaMemcpyHostToDevice);
    checkCudaError(err, "copy host B from GPU");

    err = cudaMemcpy(D_C, C, sizeof(float)*size_c, cudaMemcpyHostToDevice);
    checkCudaError(err, "Copy host C from GPU");


    // RE-WRITE THE kERNAL for matrix multiplication
    

    
    // NAIVE Matric Multiplication
    if (strcmp(imp, "naive") == 0){
        printf("naive is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 blockDim(32, 32);
        matrix_multiply<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
    }else if (strcmp(imp, "imp_2") == 0){
        printf("part 2 is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 blockDim(32*32);
        matrix_multiply_v2<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
    }else if (strcmp(imp, "imp_3") == 0){
        printf("implementation 3 is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 blockDim(32, 32);
        matrix_multiply_v3<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
    }else if (strcmp(imp, "imp_4") == 0){ 
        printf("implementation 4 is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64), 1);
        dim3 blockDim(64*8, 1);
        matrix_multiply_v4<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
    }else if (strcmp(imp, "imp_5") == 0){
        printf("implementation 5 is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64), 1);
        dim3 blockDim(8*8, 1);
        matrix_multiply_v5<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
        
    } else {
        printf("implementation 6 is being executed %s", imp);
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128), 1);
        dim3 blockDim(16*16, 1);
        matrix_multiply_v6<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);
    }
    // matrix_multiply<<<gridDim, blockDim>>>(D_A, D_B, D_C, alpha, beta, M, K, N);

    err = cudaDeviceSynchronize();
    checkCudaError(err, "cudaDeviceSynchronize");

    err = cudaMemcpy(C, D_C, sizeof(float)*size_c, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Copy host C from GPU");

    // Release the data from the GPU

    err = cudaFree(D_A);
    checkCudaError(err, "copy while cleaning A on GPU");

    err = cudaFree(D_B);
    checkCudaError(err, "copy while cleaning B on GPU");

    err = cudaFree(D_C);
    checkCudaError(err, "copy while cleaning C on GPU");

    float total = 0.0;

    for (int i = 0; i < size_c; i++){
        total += C[i];
    }

    printf("total is calculated as - %.2f", total);

    return 0;

}