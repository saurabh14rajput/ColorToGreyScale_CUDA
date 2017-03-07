/*
 *	nvcc `pkg-config --cflags opencv` main.cu `pkg-config --libs opencv` -o main.out
 *	./main.out doge.jpg
 */

#include <iostream>
#include <string>
#include <stdio.h>
#include "helper.h"
#include "timeHelper.h"
#include "converter.cuh"

void cudaStub(std::string inputFileName, std::string outputFileName);
void openCvStub(std::string inputFileName, std::string outputFileName);
void cpuStub(std::string inputFileName, std::string outputFileName);


/**
 * The main method is a stub which takes input file name and calls 
 * all three stubs (openCv, cpu, and gpu) for further prosessing
 */
int main(int argc, char **argv) {
	
	std::string inputFileName;
	std::string CudaOutputFileName  = "./greyscale_output_cuda.png";
	std::string openCvOutputFileName = "./greyscale_output_opencv.png";
	std::string cpuOutputFileName = "./greyscale_output_serial.png";

	switch (argc){
		case 2:
			inputFileName = std::string(argv[1]);
			break;
		default:
			std::cerr << "Usage: ./grayscale input_file" << std::endl;
			exit(1);
	}

	cpuStub(inputFileName, cpuOutputFileName);	
	openCvStub(inputFileName, openCvOutputFileName);
	//int p;
	//for(p=0;p<10;p++)
	cudaStub(inputFileName, CudaOutputFileName);	

	//free the memory on the device
	releaseCudaMemory();
	return 0;
}

/**
 * This is the CUDA stub that calls the method to reserve memory on the device and the host and 
 * then transfer the input from host to device.
 * Then it calls another method which sets ups the grid and block size and inturn calls
 * the CUDA kernel
 */
void cudaStub(std::string inputFileName, std::string outputFileName) {
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	//Loading the image into the device memory
	initialSetup(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, inputFileName);
	printf("The number of rows in the image:%d\n",numRows());
	printf("The number of cols in the image:%d\n",numCols());
	printf("The total number of pixls in the image:%d\n",numRows()*numCols());
	//Starting the timer to measure the performance
	GpuTimer timer;
	timer.Start();
	//call to the stub that will call the kernel
	colorToGreyCuda(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	//call to kernel is non blocking, so lets put sync primitive
	cudaDeviceSynchronize(); 
	timer.Stop();
	checkCudaErrors(cudaGetLastError());
	int err = printf("Time taken by CUDA conversion: %f msecs.\n", timer.Elapsed());
	if (err < 0) {
		//Erroe in printing
		std::cerr << "ERROR! STDOUT is CLOSED" << std::endl;
		exit(1);
	}
	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
	//output the image
	finalTouches(outputFileName, h_greyImage);
}

/**
 * This is the openCv stub that converts the input BGBA file to Greyscale
 */
void openCvStub(std::string inputFileName, std::string outputFileName) {
	cv::Mat image;
	image = cv::imread(inputFileName.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "error opening file: " << inputFileName << std::endl;
		exit(1);
	}
	GpuTimer timer;
	timer.Start();
	//Converting to Grey
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
	//allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	timer.Stop();
	int err = printf("Time taken by OPEN CV conversion: %f msecs.\n", timer.Elapsed());
	//output the image
	cv::imwrite(outputFileName.c_str(), imageGrey);
}

/**
 * Does the initial processing for cpu and in turn calls the method which actually converts 
 * RGBA image to Grey pixel by pixel serially
 */
void cpuStub(std::string inputFileName, std::string outputFileName) {
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
	//Load the image for input and output
	initialSetup(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, inputFileName);
	GpuTimer timer;
	timer.Start();
	rgbaToGreyscaleCpu(h_rgbaImage, h_greyImage, numRows(), numCols());
	timer.Stop();
	int err = printf("Time taken by SERIAL conversion: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		//error in printing
		std::cerr << "Error! STDOUT is Closed!" << std::endl;
		exit(1);
	}
	finalTouches(outputFileName, h_greyImage);
}
