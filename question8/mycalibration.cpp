#include"stdafx.h"
using namespace cv;
void myInitIntrinsicParams2D(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix,
	double aspectRatio)
{

}
double myCalibrateCamera(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
	CvMat* rvecs, CvMat* tvecs, int flags, CvTermCriteria termCrit)
{
	const int NINTRINSIC = 12;
	Ptr<CvMat> matM, _m, _Ji, _Je, _err;
	/*Levenberg-Marquardt Algorithm*/
	CvLevMarq solver;
	double reprojErr = 0;

	double A[9], k[8] = { 0,0,0,0,0,0,0,0 };
	CvMat matA = cvMat(3, 3, CV_64F, A), _k;
	int i, nimages, maxPoints = 0,  total = 0, ni = 0, pos, nparams, npstep, cn;
	double aspectRatio = 0.;
	
	//0. allocate buffers
	nimages = npoints->rows*npoints->cols;
	npstep = npoints->rows;
	for (i = 0; i < nimages; i++)
	{
		ni = npoints->data.i[i*npstep];
		maxPoints = MAX(maxPoints, ni);
		total += ni;
	}
	matM = cvCreateMat(1, total, CV_64FC3);
	_m = cvCreateMat(1, total, CV_64FC2);

	cvConvertPointsHomogeneous(objectPoints, matM);
	cvConvertPointsHomogeneous(imagePoints, _m);

	nparams = NINTRINSIC + nimages * 6;
	_Ji = cvCreateMat(maxPoints * 2, NINTRINSIC, CV_64FC1);
	_Je = cvCreateMat(maxPoints * 2, 6, CV_64FC1);
	_err = cvCreateMat(maxPoints * 2, 1, CV_64FC1);
	cvZero(_Ji);

	_k = cvMat(distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(distCoeffs->type)), k);
	if (distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 8)
	{
		if (distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 5)
			flags |= CV_CALIB_FIX_K3;
		flags |= CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	const double minValidAspectRatio = 0.01;
	const double maxValidAspectRatio = 100.0;

	//1. initialize intrinsic parameters & LM solver
	/*fing initial guess of homography*/
	myInitIntrinsicParams2D(matM, _m, npoints, imageSize, &matA, aspectRatio);


	return 0;
}
double mycalibrateCamera(InputArrayOfArrays _objectPoints,
	InputArrayOfArrays _imagePoints,
	Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
	OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria)
{
	/*initialize the camera matrix and the distortion vector*/
	Mat cameraMatrix0 = _cameraMatrix.getMat();
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix0.convertTo(cameraMatrix, CV_64F);
	Mat distCoeffs0 = _distCoeffs.getMat();
	Mat distCoeffs = Mat(1, 5, CV_64F, Scalar::all(0));
	distCoeffs0.convertTo(distCoeffs, CV_64F);


	size_t num_images = _objectPoints.total();
	CV_Assert(num_images > 0);
	Mat objPt, imgPt, npoints;//the array of the points
	Mat rvecM((int)num_images, 3, CV_64FC1), tvecM((int)num_images, 3, CV_64FC1);

	/*get calibration data*/
	int i, j = 0, ni = 0, total = 0;
	for ( i = 0; i < num_images; i++)
	{
		ni = _objectPoints.getMat(i).checkVector(3, CV_32F);
		total += ni;
	}
	npoints.create(1, (int)num_images, CV_32S);
	objPt.create(1, (int)total, CV_32FC3);
	imgPt.create(1, (int)total, CV_32FC2);
	/*the head of the data array*/
	Point3f* objPtData = objPt.ptr<Point3f>();
	Point2f* imgPtData = imgPt.ptr<Point2f>();
	/*Record pointers at each point in their array*/
	for ( i = 0; i < num_images; i++,j+=ni)
	{
		Mat objMat = _objectPoints.getMat(i);
		Mat imgMat = _imagePoints.getMat(i);
		ni = objMat.checkVector(3, CV_32F);
		/*int ni1 = imgMat.checkVector(2, CV_32F);*/
		npoints.at<int>(i) = ni;
		memcpy(objPtData + j, objMat.data, ni * sizeof(objPtData[0]));
		memcpy(imgPtData + j, imgMat.data, ni * sizeof(imgPtData[0]));

	}
	/*Convert to C format*/
	CvMat c_objPt = objPt, c_imgPt = imgPt, c_npoints = npoints;
	CvMat c_cameraMatrix = cameraMatrix, c_distCoeffs = distCoeffs;
	CvMat c_rvecM = rvecM, c_tvecM = tvecM;
	
	/*start to calculate*/
	double reprojErr = myCalibrateCamera(&c_objPt, &c_imgPt, &c_npoints, imageSize,
		&c_cameraMatrix, &c_distCoeffs, &c_rvecM,
		&c_tvecM, flags, criteria);
	return 0;
}
