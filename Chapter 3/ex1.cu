#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 1023  // Size of the matrix (1023x1023)

__global__
void matrixMatrixMultRow(float* A, float* B, float* C, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width) {
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int i = 0; i < width; ++i) {
                sum += A[row * width + i] * B[i * width + j];
            }
            C[row * width + j] = sum;
        }
    }
}

int main() {
    int width = WIDTH;

    // Allocate memory for matrices on host
    float *h_A = new float[width * width];
    float *h_B = new float[width * width];
    float *h_C = new float[width * width];

    // Initialize matrices A and B with some values
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = 1.0f;  // You can use random values here if you'd like
        h_B[i] = 1.0f;
    }

    // Allocate memory for matrices on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, width * width * sizeof(float));
    cudaMalloc(&d_B, width * width * sizeof(float));
    cudaMalloc(&d_C, width * width * sizeof(float));

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(128);  // 128 threads per block in 1D
    dim3 gridDim((width + 128 - 1) / 128);  // Number of blocks (1D)

    // Launch the kernel
    matrixMatrixMultRow<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);

    // Check for errors
    cudaDeviceSynchronize();

    // Copy the result matrix C back to host
    cudaMemcpy(h_C, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // (Optional) Print out some values of matrix C to check correctness
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[0][1] = " << h_C[1] << std::endl;
    std::cout << "C[1][0] = " << h_C[width] << std::endl;

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