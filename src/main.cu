#include <stdio.h>
#include <stdlib.h>
#include "squared.cuh"

// main file to decide which kernel to run.
int main(int argc, char* argv[]) {
/*===========================================================*/
//READ FILE
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [matrix A] [matrix B]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    FILE* file_A = fopen(argv[1], "r");
    FILE* file_B = fopen(argv[2], "r");

    if (file_A == NULL || file_B == NULL) {
        fprintf(stderr, "Error opening matrix files.\n");
        exit(EXIT_FAILURE);
    }
/*===========================================================*/

    return squared(file_A, file_B);
}