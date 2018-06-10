#include"stdafx.h"
using namespace cv;
CV_IMPL void myFindExtrinsicCameraParams2(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* A,
	const CvMat* distCoeffs, CvMat* rvec, CvMat* tvec)
{
	const int max_iter = 20;
	int i, count;
	double h[9], iC[9], h1_norm, h2_norm;
	double param[6];//Extrinsic Camera Parameters
	CvMat invCaMat = cvMat(3, 3, CV_64F, iC);
	CvMat matH = cvMat(3, 3, CV_64F, h);
	CvMat _r = cvMat(3, 1, CV_64F, param);
	CvMat _t = cvMat(3, 1, CV_64F, param + 3);
	CvMat _param = cvMat(6, 1, CV_64F, param);
	CvMat _h1, _h2, _h3;
	CvMat _dpdr, _dpdt;

	count = MAX(objectPoints->cols, objectPoints->rows);
	cvInvert(A, &invCaMat, CV_SVD);


	cvFindHomography(objectPoints, imagePoints, &matH);
	cvGetCol(&matH, &_h1, 0);
	_h2 = _h1; _h2.data.db++;
	_h3 = _h2; _h3.data.db++;
	cvGEMM(&invCaMat, &_h1, 1, 0, 0, &_h1);
	cvGEMM(&invCaMat, &_h2, 1, 0, 0, &_h2);
	cvGEMM(&invCaMat, &_h3, 1, 0, 0, &_h3);

	h1_norm = sqrt(h[0] * h[0] + h[3] * h[3] + h[6] * h[6]);
	h2_norm = sqrt(h[1] * h[1] + h[4] * h[4] + h[7] * h[7]);
	cvScale(&_h1, &_h1, 1. / MAX(h1_norm, DBL_EPSILON));
	cvScale(&_h2, &_h2, 1. / MAX(h2_norm, DBL_EPSILON));
	cvScale(&_h3, &_t, 2. / MAX(h1_norm + h2_norm, DBL_EPSILON));
	cvCrossProduct(&_h1, &_h2, &_h3);
	cvRodrigues2(&matH, &_r);


	// refine extrinsic parameters using iterative algorithm
	CvLevMarq solver(6, count * 2, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, max_iter, FLT_EPSILON), true);
	cvCopy(&_param, solver.param);

	for (;;)
	{
		CvMat *matJ = 0, *_err = 0;
		const CvMat *__param = 0;
		bool proceed = solver.update(__param, matJ, _err);
		cvCopy(__param, &_param);
		if (!proceed || !_err)
			break;
		cvReshape(_err, _err, 2, 1);
		if (matJ)
		{
			cvGetCols(matJ, &_dpdr, 0, 3);
			cvGetCols(matJ, &_dpdt, 3, 6);
			cvProjectPoints2(objectPoints, &_r, &_t, A, distCoeffs,
				_err, &_dpdr, &_dpdt, 0, 0, 0);
		}
		else
		{
			cvProjectPoints2(objectPoints, &_r, &_t, A, distCoeffs,
				_err, 0, 0, 0, 0, 0);
		}
		cvSub(_err, imagePoints, _err);
		cvReshape(_err, _err, 1, 2 * count);
	}
	cvCopy(solver.param, &_param);

	_r = cvMat(rvec->rows, rvec->cols,
		CV_MAKETYPE(CV_64F, CV_MAT_CN(rvec->type)), param);
	_t = cvMat(tvec->rows, tvec->cols,
		CV_MAKETYPE(CV_64F, CV_MAT_CN(tvec->type)), param + 3);

	cvConvert(&_r, rvec);
	cvConvert(&_t, tvec);

}

CV_IMPL void myInitIntrinsicParams2D(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix)
{
	Ptr<CvMat> matA, _b;

	int i, j, pos, num_images, ni = 0;
	double camera_parameters[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
	double H[9], focal_len[2];
	CvMat _cameraMat = cvMat(3, 3, CV_64F, camera_parameters);
	CvMat matH = cvMat(3, 3, CV_64F, H);//homography
	CvMat _focal_length = cvMat(2, 1, CV_64F, focal_len);//focal length
	/*whether the point data is legal*/

	num_images = npoints->rows + npoints->cols - 1;//the number of images

	matA = cvCreateMat(2 * num_images, 2, CV_64F);
	_b = cvCreateMat(2 * num_images, 1, CV_64F);
	/*the principle points*/
	camera_parameters[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
	camera_parameters[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;

	// calculating 
	for (i = 0, pos = 0; i < num_images; i++, pos += ni)
	{
		double* Ap = matA->data.db + i * 4;
		double* bp = _b->data.db + i * 2;
		ni = npoints->data.i[i];
		double h[3], v[3], d1[3], d2[3];
		double n[4] = { 0,0,0,0 };
		CvMat _imageMat, _objectMat;
		cvGetCols(objectPoints, &_objectMat, pos, pos + ni);
		cvGetCols(imagePoints, &_imageMat, pos, pos + ni);


		cvFindHomography(&_objectMat, &_imageMat, &matH);
		/*Translation change*/
		H[0] -= H[6] * camera_parameters[2]; H[1] -= H[7] * camera_parameters[2]; H[2] -= H[8] * camera_parameters[2];
		H[3] -= H[6] * camera_parameters[5]; H[4] -= H[7] * camera_parameters[5]; H[5] -= H[8] * camera_parameters[5];
		/*Then the H is :
						[ f_1 0   0 ]
						[ 0   f_2 0 ] *  [R|t]
						[ 0   0   1 ]
		  According to the paper, Using the knowledge that r1 and r2 are orthonormal, we have:
						h_T_1 A−T A−1h_2 = 0 (3)
						h_T_1 A−T A−1h_1 = h_T_2 A−T A−1h_2 . (4)

		*/
		for (j = 0; j < 3; j++)
		{
			double t0 = H[j * 3], t1 = H[j * 3 + 1];
			h[j] = t0; v[j] = t1;//h=h1,v=h2
			d1[j] = (t0 + t1)*0.5;
			d2[j] = (t0 - t1)*0.5;
			n[0] += t0 * t0; n[1] += t1 * t1;//n0=h31^2,n1=h32^2
			n[2] += d1[j] * d1[j]; n[3] += d2[j] * d2[j];
		}

		for (j = 0; j < 4; j++)
			n[j] = 1. / sqrt(n[j]);

		for (j = 0; j < 3; j++)
		{
			h[j] *= n[0]; v[j] *= n[1];
			d1[j] *= n[2]; d2[j] *= n[3];
		}
		/*  we have:
			[h11h21,      h12h22,      h13h23      ] * [1/f_1^2] = 0
			[h11^2-h22^2  h12^2-h22^2  h13^2-h23^2 ]   [1/f_2^2]
													   [1	   ]
			That is to say:
			[h11h21,      h12h22,	 ] * [1/f_1^2] = [-h13h23    ]
			[h11^2-h22^2  h12^2-h22^2]   [1/f_2^2] = [h13^2-h23^2]
			*/
		Ap[0] = h[0] * v[0]; Ap[1] = h[1] * v[1];
		Ap[2] = d1[0] * d2[0]; Ap[3] = d1[1] * d2[1];
		bp[0] = -h[2] * v[2]; bp[1] = -d1[2] * d2[2];
	}
	/*solve the linear system matA*_focal_length=_b, where _focal_length is the reciprocal of focal length, matA is the inverse of rotation matrix*/
	cvSolve(matA, _b, &_focal_length, CV_NORMAL + CV_SVD);//Least-squares solution
	camera_parameters[0] = sqrt(fabs(1. / focal_len[0]));
	camera_parameters[4] = sqrt(fabs(1. / focal_len[1]));


	cvConvert(&_cameraMat, cameraMatrix);
}
CV_IMPL double myCalibrateCamera2(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
	CvMat* rvecs, CvMat* tvecs, CvTermCriteria termCrit)
{
	const int NINTRINSIC = 12;//number of internal parameters
	Ptr<CvMat> matM, _m, _Ji, _Je, _err;
	/*Levenberg-Marquardt Algorithm*/
	CvLevMarq solver;
	double reprojErr = 0;//return value

	double Camera_Parameters[9], k[8] = { 0,0,0,0,0,0,0,0 };
	CvMat CaMat = cvMat(3, 3, CV_64F, Camera_Parameters), _k;
	int i, nimages/*the number of images*/, maxPoints = 0, total = 0;/*the total number of points*/
	int ni = 0, pos, nparams, npstep, cn;
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
	matM = cvCreateMat(1, total, CV_64FC3);//the matrix of objectpoints
	_m = cvCreateMat(1, total, CV_64FC2);//the matrix of imagepoints

	cvConvertPointsHomogeneous(objectPoints, matM);
	cvConvertPointsHomogeneous(imagePoints, _m);
	/*CV_MAT_TYPE(objectPoints->type)== CV_64FC3;
	CV_MAT_TYPE(matM->type)== CV_64FC3;*/
	nparams = NINTRINSIC + nimages * 6;
	_Ji = cvCreateMat(maxPoints * 2, NINTRINSIC, CV_64FC1);
	_Je = cvCreateMat(maxPoints * 2, 6, CV_64FC1);
	_err = cvCreateMat(maxPoints * 2, 1, CV_64FC1);
	cvZero(_Ji);

	_k = cvMat(distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(distCoeffs->type)), k);


	//1. initialize intrinsic parameters & LM solver
	/*fing initial guess of homography*/
	myInitIntrinsicParams2D(matM, _m, npoints, imageSize, &CaMat);
	/*Levenberg-Marquardt Algorithm */
	solver.init(nparams, 0, termCrit);
	{
		double* param = solver.param->data.db;
		uchar* mask = solver.mask->data.ptr;
		param[0] = Camera_Parameters[0]; param[1] = Camera_Parameters[4];//focal length
		param[2] = Camera_Parameters[2]; param[3] = Camera_Parameters[5];//principle point
		/*disortions*/
		param[4] = k[0]; param[5] = k[1]; param[6] = k[2]; param[7] = k[3];
		param[8] = k[4]; param[9] = k[5]; param[10] = k[6]; param[11] = k[7];

		/*Do not consider k4,k5,k6*/
		mask[9] = 0;
		mask[10] = 0;
		mask[11] = 0;
	}
	//2. initialize extrinsic parameters
	for (i = 0, pos = 0; i < nimages; i++, pos += ni)
	{
		CvMat _Mi, _mi, _ri, _ti;
		ni = npoints->data.i[i*npstep];

		cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);//rotation vector
		cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);//translation vector

		cvGetCols(matM, &_Mi, pos, pos + ni);
		cvGetCols(_m, &_mi, pos, pos + ni);
		myFindExtrinsicCameraParams2(&_Mi, &_mi, &CaMat, &_k, &_ri, &_ti);
	}
	// 3. run the optimization
	for (;;)
	{
		const CvMat* _param = 0;
		CvMat *_JtJ = 0, *_JtErr = 0;
		double* _errNorm = 0;
		bool proceed = solver.updateAlt(_param, _JtJ, _JtErr, _errNorm);
		double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;
		int test;

		/*the first 12 parameters*/
		Camera_Parameters[0] = param[0]; Camera_Parameters[4] = param[1];//focal length
		Camera_Parameters[2] = param[2]; Camera_Parameters[5] = param[3];//principle point
		k[0] = param[4]; k[1] = param[5]; k[2] = param[6]; k[3] = param[7];
		k[4] = param[8]; k[5] = param[9]; k[6] = param[10]; k[7] = param[11];

		if (!proceed)
			break;

		reprojErr = 0;

		for (i = 0, pos = 0; i < nimages; i++, pos += ni)
		{
			CvMat _ObjectP, _ImageP, _ri, _ti, _dpdr, _dpdt, _dpdf, _dpdc, _dpdk, _imageP2, _part;
			ni = npoints->data.i[i*npstep];
			/*get rotation and translation vector*/
			cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);
			cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);
			/*get object points and image points*/
			cvGetCols(matM, &_ObjectP, pos, pos + ni);
			cvGetCols(_m, &_ImageP, pos, pos + ni);

			_Je->rows = _Ji->rows = _err->rows = ni * 2;
			//Optional Nx3 matrix of derivatives of image points with respect to components of the rotation vector
			cvGetCols(_Je, &_dpdr, 0, 3);
			//Optional Nx3 matrix of derivatives of image points w.r.t.components of the translation vector
			cvGetCols(_Je, &_dpdt, 3, 6);
			//Optional Nx2 matrix of derivatives of image points w.r.t. fx and fy
			cvGetCols(_Ji, &_dpdf, 0, 2);
			//Optional Nx2 matrix of derivatives of image points w.r.t. cx and cy
			cvGetCols(_Ji, &_dpdc, 2, 4);
			//Optional Nx4 matrix of derivatives of image points w.r.t.distortion coefficients
			cvGetCols(_Ji, &_dpdk, 4, NINTRINSIC);
			cvReshape(_err, &_imageP2, 2, 1);

			if (test = (_JtJ || _JtErr))
			{
				/*Calculate the Jacobian matrix*/
				cvProjectPoints2(&_ObjectP, &_ri, &_ti, &CaMat, &_k, &_imageP2, &_dpdr, &_dpdt, &_dpdf, &_dpdc, &_dpdk, 0);
				test = 0;
			}
			else
				cvProjectPoints2(&_ObjectP, &_ri, &_ti, &CaMat, &_k, &_imageP2);

			cvSub(&_imageP2, &_ImageP, &_imageP2);
			/*Input J^T*J*/
			if (test = (_JtJ || _JtErr))
			{
				/*intrinsic parameters have nothing to do with different points*/
				cvGetSubRect(_JtJ, &_part, cvRect(0, 0, NINTRINSIC, NINTRINSIC));
				cvGEMM(_Ji, _Ji, 1, &_part, 1, &_part, CV_GEMM_A_T);

				cvGetSubRect(_JtJ, &_part, cvRect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6));
				cvGEMM(_Je, _Je, 1, 0, 0, &_part, CV_GEMM_A_T);

				cvGetSubRect(_JtJ, &_part, cvRect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC));
				cvGEMM(_Ji, _Je, 1, 0, 0, &_part, CV_GEMM_A_T);

				cvGetRows(_JtErr, &_part, 0, NINTRINSIC);
				cvGEMM(_Ji, _err, 1, &_part, 1, &_part, CV_GEMM_A_T);

				cvGetRows(_JtErr, &_part, NINTRINSIC + i * 6, NINTRINSIC + (i + 1) * 6);
				cvGEMM(_Je, _err, 1, 0, 0, &_part, CV_GEMM_A_T);
			}

			double errNorm = cvNorm(&_imageP2, 0, CV_L2);
			reprojErr += errNorm * errNorm;
		}
		if (_errNorm)
			*_errNorm = reprojErr;
	}
	// 4. store the results
	cvConvert(&CaMat, cameraMatrix);
	cvConvert(&_k, distCoeffs);

	for (i = 0; i < nimages; i++)
	{
		CvMat src, dst;
		if (rvecs)
		{
			src = cvMat(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i * 6);
			if (rvecs->rows == nimages && rvecs->cols*CV_MAT_CN(rvecs->type) == 9)
			{
				dst = cvMat(3, 3, CV_MAT_DEPTH(rvecs->type),
					rvecs->data.ptr + rvecs->step*i);
				cvRodrigues2(&src, &CaMat);
				cvConvert(&CaMat, &dst);
			}
			else
			{
				dst = cvMat(3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
					rvecs->data.ptr + i * CV_ELEM_SIZE(rvecs->type) :
					rvecs->data.ptr + rvecs->step*i);
				cvConvert(&src, &dst);
			}
		}
		if (tvecs)
		{
			src = cvMat(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i * 6 + 3);
			dst = cvMat(3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
				tvecs->data.ptr + i * CV_ELEM_SIZE(tvecs->type) :
				tvecs->data.ptr + tvecs->step*i);
			cvConvert(&src, &dst);
		}
	}
	return std::sqrt(reprojErr / total);
}
double mycalibrateCamera(InputArrayOfArrays _objectPoints,
	InputArrayOfArrays _imagePoints,
	Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
	OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, TermCriteria criteria)
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
	for (i = 0; i < num_images; i++)
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
	for (i = 0; i < num_images; i++, j += ni)
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
	double reprojErr = myCalibrateCamera2(&c_objPt, &c_imgPt, &c_npoints, imageSize,
		&c_cameraMatrix, &c_distCoeffs, &c_rvecM,
		&c_tvecM, criteria);



	_rvecs.create((int)num_images, 1, CV_64FC3);
	_tvecs.create((int)num_images, 1, CV_64FC3);
	for (i = 0; i < (int)num_images; i++)
	{
		_rvecs.create(3, 1, CV_64F, i, true);
		Mat rv = _rvecs.getMat(i);
		memcpy(rv.data, rvecM.ptr<double>(i), 3 * sizeof(double));
		_tvecs.create(3, 1, CV_64F, i, true);
		Mat tv = _tvecs.getMat(i);
		memcpy(tv.data, tvecM.ptr<double>(i), 3 * sizeof(double));

	}
	cameraMatrix.copyTo(_cameraMatrix);
	distCoeffs.copyTo(_distCoeffs);

	return reprojErr;
}
