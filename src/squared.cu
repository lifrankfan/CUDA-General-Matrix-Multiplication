#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "squared.cuh";
//#include <device_functions.h>

#define TILE_WIDTH 16

__global__ void squaredKernel(float* A, float* B, float* C, int width) {
    // set size of subtiles
    __shared__ float subA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subB[TILE_WIDTH][TILE_WIDTH];

    // convenience
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

    // because squared, only require row
    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;

    // temporary value for each subtile
    float Cvalue = 0.0;

    for (int m = 0; m < (width - 1) / TILE_WIDTH + 1; m++) {
        if (row < width && m * TILE_WIDTH + tx < width) {
            subA[ty][tx] = A[row * width + m * TILE_WIDTH + tx];
        }
        else {
            subA[ty][tx] = 0.0;
        }

        if (m * TILE_WIDTH + ty < width && col < width) {
            subB[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        }
        else {
            subB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += subA[ty][k] * subB[k][tx];
        }

        __syncthreads();
    }

    // completes one cell entry
    if (row < width && col < width) {
        C[row * width + col] = Cvalue;
    }
}

int squared(FILE * file_A, FILE * file_B) {
/*==========================================================*/
// Set row and column
    int rows_A, cols_A, rows_B, cols_B;

    // reads first line of files as dimensions of matrix
    fscanf(file_A, "%d %d", &rows_A, &cols_A);
    fscanf(file_B, "%d %d", &rows_B, &cols_B);
/*===========================================================*/

    // ensure matrices are squared
    if (cols_A != rows_B) {
        fprintf(stderr, "Error: incompatible matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int width = cols_B;

    // allocate memory for host matrices
    float* h_A = (float*)malloc(rows_A * cols_A * sizeof(float));
    float* h_B = (float*)malloc(rows_B * cols_B * sizeof(float));
    float* h_C = (float*)malloc(rows_A * cols_B * sizeof(float));

    // read data into h_A and h_B
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_A; j++) {
            fscanf(file_A, "%f", &h_A[i * cols_A + j]);
        }
    }

    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            fscanf(file_B, "%f", &h_B[i * cols_B + j]);
        }
    }

    // allocate GPU memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, rows_A * cols_A * sizeof(float));
    cudaMalloc((void**)&d_B, rows_B * cols_B * sizeof(float));
    cudaMalloc((void**)&d_C, rows_A * cols_B * sizeof(float));

    // copy memory to GPU
    cudaMemcpy(d_A, h_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows_B * cols_B * sizeof(float), cudaMemcpyHostToDevice);

    // grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (rows_A - 1) / TILE_WIDTH + 1);

    // launch the kernel
    squaredKernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C, width);

    // copy GPU memory back to CPU once the kernel completes
    cudaMemcpy(h_C, d_C, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);

    // prints resulting matrix
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            printf("%.2f ", h_C[i * cols_B + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    fclose(file_A);
    fclose(file_B);

    return 1;
}
