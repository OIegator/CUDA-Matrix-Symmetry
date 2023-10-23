
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdio.h>


void GenerateRandomMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = 0;
        }
        if (rand() % 2 == 0) {
            int random_col = rand() % cols;
            matrix[i * cols + random_col] = 1;
        }
    }
}


void PrintMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Функция на CPU для проверки симметрии строки относительно средней вертикальной линии матрицы
bool IsRowSymmetric(int* row, int length) {
    for (int i = 0; i < length / 2; ++i) {
        if (row[i] != row[length - i - 1]) {
            return false;
        }
    }
    return true;
}

// Функция на GPU для проверки симметрии строки относительно средней вертикальной линии матрицы
__global__ void _IsRowSymmetricGPU(const int* matrix, bool* result, int rows, int cols) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx < rows) {
        const int* row = matrix + rowIdx * cols;
        result[rowIdx] = true;
        for (int i = 0; i < cols / 2; ++i) {
            if (row[i] != row[cols - i - 1]) {
                result[rowIdx] = false;
            }
        }
    }
}
// Оптимизированная функция на GPU для проверки симметрии строки относительно средней вертикальной линии матрицы
__global__ void IsRowSymmetricGPU(int* matrix, int* result, int rows, int cols) {
    __shared__ int sharedCache[256][32]; // Кеш в разделяемой памяти для 256 строк, каждая строка - 16 элементов слева + 16 элементов справа

    const int t = threadIdx.x;
    const int bx = blockIdx.x;
    bool isSymmetric = true;
    for (int part = 0; part < cols / 32; part++) {
        // Загружаем 16 элементов слева и 16 элементов справа в кеш для каждой из 256 строк
        for (int k = 0; k < 16; ++k) {

            sharedCache[t / 16 + k * 16][t % 16] = matrix[(bx / 16 * blockDim.x + t / 16 + k * 16) * cols + part * 16 + t % 16];
            sharedCache[t / 16 + k * 16][32 - 1 - t % 16] = matrix[(bx / 16 * blockDim.x + t / 16 + k * 16) * cols + cols - 1 - part * 16 - t % 16];
        }

        // Барьер синхронизации для ожидания, пока все потоки загрузят кеш
        __syncthreads();

        // Проверяем симметрию для текущей строки
        for (int j = 0; j < 16; ++j) {
            if (sharedCache[t][j] != sharedCache[t][32 - 1 - j]) {
                isSymmetric = false; 
            }
        }
        __syncthreads();
    }
    result[bx * blockDim.x + t / 16] = isSymmetric;
}



int main() {
    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    srand(static_cast<unsigned>(time(nullptr))); // Инициализируем генератор случайных чисел

    const int numRows = 50000;
    const int numCols = 1024;
    /*const int blockDim = 1024;*/
    const int matrixSize = numRows * numCols;
    const int blockDim = 256;
    const int numBlocks = (numRows + 256 - 1) / 256;

    int* h_matrix = new int[matrixSize]; 
    bool* h_symmetryResults = new bool[numRows]; 

    // Генерируем случайную матрицу на хосте
    GenerateRandomMatrix(h_matrix, numRows, numCols);

    startCPU = clock();

    // Проверяем симметрию строк и сохраняем результаты в векторе
    bool symmetryResults[numRows];
    for (int i = 0; i < numRows; ++i) {
        symmetryResults[i] = IsRowSymmetric(h_matrix + i * numCols, numCols);

    }
    elapsedTimeCPU = (float)(clock() - startCPU) / CLOCKS_PER_SEC;

    // Выделяем память на устройстве
    int* d_matrix;
    int* d_symmetryResults;
    cudaMalloc((void**)&d_matrix, sizeof(int) * matrixSize);
    cudaMalloc((void**)&d_symmetryResults, sizeof(bool) * numRows);

    // Копируем матрицу с хоста на устройство
    cudaMemcpy(d_matrix, h_matrix, sizeof(int) * matrixSize, cudaMemcpyHostToDevice);

    cudaEventRecord(startCUDA, 0);
   
    // Вызываем на GPU для проверки симметрии
    IsRowSymmetricGPU <<<numBlocks, blockDim>>> (d_matrix, d_symmetryResults, numRows, numCols);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    // Копируем результаты с устройства на хост
    cudaMemcpy(h_symmetryResults, d_symmetryResults, sizeof(bool) * numRows, cudaMemcpyDeviceToHost);

    // Освобождаем память на устройстве
    cudaFree(d_matrix);
    cudaFree(d_symmetryResults);

    // Выводим результаты на консоль
    for (int i = 0; i < numRows; ++i) {
        std::cout << "Row " << i << ": " << (h_symmetryResults[i] ? "Symmetric" : "Not Symmetric") << "\t"
             << (symmetryResults[i] ? "Symmetric" : "Not Symmetric") << std::endl;
    }

    // Сравниваем результаты CPU и GPU
    bool resultsMatch = true;
    for (int i = 0; i < numRows; ++i) {
        if (symmetryResults[i] != h_symmetryResults[i]) {
            resultsMatch = false;
            break;
        }
    }

    // Выводим результаты сравнения
    if (resultsMatch) {
        std::cout << "Results match between CPU and GPU.\n";
    }
    else {
        std::cout << "Results do not match between CPU and GPU.\n";
    }

    // Выводим процентное ускорение
    float speedup = elapsedTimeCPU * 1000 / elapsedTimeCUDA;

    std::cout << std::endl;
    std::cout << std::setw(20) << std::left << "Measurement" << std::setw(20) << "Time (ms)" << std::endl;
    std::cout << std::setw(20) << std::left << "CPU" << std::setw(20) << elapsedTimeCPU * 1000 << std::endl;
    std::cout << std::setw(20) << std::left << "CUDA" << std::setw(20) << elapsedTimeCUDA << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(20) << std::left << "Speedup" << "x" << speedup << std::endl;


    delete[] h_matrix;
    delete[] h_symmetryResults;


    return 0;
}
