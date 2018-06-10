#pragma once
/** @overload double mycalibrateCamera( InputArrayOfArrays objectPoints,
InputArrayOfArrays imagePoints, Size imageSize,
InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
OutputArray stdDeviations, OutputArray perViewErrors,
int flags = 0, TermCriteria criteria = TermCriteria(
TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) )
*/
#include "opencv2/calib3d.hpp"
using namespace cv;
double mycalibrateCamera(InputArrayOfArrays objectPoints,
	InputArrayOfArrays imagePoints, Size imageSize,
	InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
	OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
	TermCriteria criteria = TermCriteria(
		TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));
