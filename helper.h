#ifndef __HELPER_H__
#define __HELPER_H__

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timeHelper.h"

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

//Getting image in the Open CV Mat format
//Converting it to RGBA for the device
//loading the image in both host and device memory
void initialSetup(uchar4 **inputImage, unsigned char **greyImage,
		uchar4 **d_rgbaImage, unsigned char **d_greyImage,
		const std::string &filename) {
	//Initializing the context
	checkCudaErrors(cudaFree(0));
	//creating a instance of CV image in Mat(matrices) format and loading the input image in that object
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}
	//OpenCV stores matrices as BGR (a difrent order than RGB) and also we need it in a 4 channel format for the GPU to process it.
	//So we convert it in RGBA format
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//Lets allocate memory for output on the host
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//Reserving memory on the device for the input color image and grey output image
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	//Let us set the output memory all to zero before calculating the output
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); 

	//Let us transfer the input image to the device memory now
	//We dont need to put synchronization primitives here since this is a blocking call 
	//and will only return once the operation is finished
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void finalTouches(const std::string& output_file, unsigned char* data_ptr) {
	//output the image
	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
	cv::imwrite(output_file.c_str(), output);
}

void releaseCudaMemory()
{
	//free memory
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
	cudaDeviceReset();
}

#endif
