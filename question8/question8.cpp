// question8.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"

using namespace cv;

int main(int argc, char **argv)
{
	Mat a = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	imshow("title", a);
	waitKey();
	return 0;
}
