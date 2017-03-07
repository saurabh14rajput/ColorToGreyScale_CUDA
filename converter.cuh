
#include <stdio.h>
#include "helper.h"

/**
 * CUDA kernel definition.
 */
__global__
void colorToGreyCudaKernel(const uchar4* const rgbaImage,
		unsigned char* const greyImage,
		const int numRows, const int numCols)
{
	//Getting theb exact pixel to work on the current thread
	const long pixelIndex = threadIdx.x + blockDim.x*blockIdx.x;

	//In case we end up creating more threads than the number of pixels, we need to
	//check that we dont access array index out of bounds
	if(pixelIndex<numRows*numCols) { 
		uchar4 const currentPixel = rgbaImage[pixelIndex];
		greyImage[pixelIndex] = .299f*currentPixel.x + .587f*currentPixel.y  + .114f*currentPixel.z;
		//greyImage[pixelIndex] = (currentPixel.x + currentPixel.y  + currentPixel.z)/3;
	}
}

/**
 * Stub that calls CUDA kernel.
 */ 
void colorToGreyCuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
		unsigned char* const d_greyImage, const size_t numRows, const size_t numCols)
{
	const int numThreadsPerBlock = 512;
	//const int numThreadsPerBlock = 1024;
	printf("Number of threads per block: %d\n",numThreadsPerBlock);
	//Calculating the number of blocks we will be needing.
	const int numBlocks = (1 + ((numRows*numCols - 1) / numThreadsPerBlock)); 
	//setting the block size in terms of number of threads
	const dim3 blockSize(numThreadsPerBlock, 1, 1);
	//setting the grid size in terms of number of blocks
	const dim3 gridSize(numBlocks, 1, 1);
	//calling the CUDA kernel
	colorToGreyCudaKernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
}

/**
 * CPU RGBA to Greay converter.
 */
void rgbaToGreyscaleCpu(const uchar4* const rgbaImage, unsigned char *const greyImage,
		const size_t numRows, const size_t numCols)
{
	for (size_t r = 0; r < numRows; ++r) {
		for (size_t c = 0; c < numCols; ++c) {
			//getting to the right index in the array of input image
			const uchar4 rgba = rgbaImage[r * numCols + c];
			const float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
			greyImage[r * numCols + c] = channelSum;
		}
	}
}
