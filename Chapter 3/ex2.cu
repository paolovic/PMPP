#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 3

__global__
void matrixVecMult(float* B, float* C, float* A, int width){
    int row = threadIdx.x+blockIdx.x*blockDim.x;
    
    if (row < width){
        float sum = 0;
        for (int i=0; i<width; ++i){
            sum+=B[row*width+i] * C[i];
        }
        A[row] = sum;
    }
}

int main(){
    int width = WIDTH;

    // Allocate memory for matrices on host
    float *h_A = new float[width];
    float *h_B = new float[width * width];
    float *h_C = new float[width];

    // Initialize matrix B with some values
    for (int i = 0; i < width * width; ++i) {
        h_B[i] = 1.0f;  // You can use random values here if you'd like
    }

    // Initialize vector C with some values
    for (int i = 0; i < width; ++i) {
        h_C[i] = i+1;
    }

    // Allocate memory for matrices on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, width * sizeof(float));
    cudaMalloc(&d_B, width * width * sizeof(float));
    cudaMalloc(&d_C, width * sizeof(float));

    cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(128);  // 128 threads per block in 1D
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x);  // Number of blocks (1D)

    // Launch the kernel
    matrixVecMult<<<gridDim, blockDim>>>(d_B, d_C, d_A, width);

    // Check for errors
    cudaDeviceSynchronize();

    // Copy the result matrix C back to host
    cudaMemcpy(h_A, d_A, width * sizeof(float), cudaMemcpyDeviceToHost);

    // (Optional) Print out some values of matrix C to check correctness
    std::cout << "A[0] = " << h_A[0] << std::endl;
    std::cout << "A[1] = " << h_A[1] << std::endl;
    std::cout << "A[2] = " << h_A[width-1] << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}