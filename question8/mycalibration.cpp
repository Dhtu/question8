#include"stdafx.h"
using namespace cv;

CV_IMPL void cvFindExtrinsicCameraParams2(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* A,
	const CvMat* distCoeffs, CvMat* rvec, CvMat* tvec,
	int useExtrinsicGuess)
{
	const int max_iter = 20;
	Ptr<CvMat> matM, _Mxy, _m, _mn, matL;/*objectmat and imagemat*/

	int i, count;
	double a[9], ar[9] = { 1,0,0,0,1,0,0,0,1 }, R[9];
	double MM[9], U[9], V[9], W[3];
	CvScalar Mc;
	double param[6];
	CvMat matA = cvMat(3, 3, CV_64F, a);//camera matrix
	CvMat _Ar = cvMat(3, 3, CV_64F, ar);//new camera matrix
	CvMat matR = cvMat(3, 3, CV_64F, R);
	CvMat _r = cvMat(3, 1, CV_64F, param);
	CvMat _t = cvMat(3, 1, CV_64F, param + 3);
	CvMat _Mc = cvMat(1, 3, CV_64F, Mc.val);//delta
	CvMat _MM = cvMat(3, 3, CV_64F, MM);
	CvMat matU = cvMat(3, 3, CV_64F, U);
	CvMat matV = cvMat(3, 3, CV_64F, V);//svd_v
	CvMat matW = cvMat(3, 1, CV_64F, W);//svd_w
	CvMat _param = cvMat(6, 1, CV_64F, param);
	CvMat _dpdr, _dpdt;

	CV_Assert(CV_IS_MAT(objectPoints) && CV_IS_MAT(imagePoints) &&
		CV_IS_MAT(A) && CV_IS_MAT(rvec) && CV_IS_MAT(tvec));

	count = MAX(objectPoints->cols, objectPoints->rows);
	matM = cvCreateMat(1, count, CV_64FC3);
	_m = cvCreateMat(1, count, CV_64FC2);

	cvConvertPointsHomogeneous(objectPoints, matM);
	cvConvertPointsHomogeneous(imagePoints, _m);
	cvConvert(A, &matA);

	CV_Assert((CV_MAT_DEPTH(rvec->type) == CV_64F || CV_MAT_DEPTH(rvec->type) == CV_32F) &&
		(rvec->rows == 1 || rvec->cols == 1) && rvec->rows*rvec->cols*CV_MAT_CN(rvec->type) == 3);

	CV_Assert((CV_MAT_DEPTH(tvec->type) == CV_64F || CV_MAT_DEPTH(tvec->type) == CV_32F) &&
		(tvec->rows == 1 || tvec->cols == 1) && tvec->rows*tvec->cols*CV_MAT_CN(tvec->type) == 3);

	_mn = cvCreateMat(1, count, CV_64FC2);
	_Mxy = cvCreateMat(1, count, CV_64FC2);

	// normalize image points
	// (unapply the intrinsic matrix transformation and distortion)
	cvUndistortPoints(_m, _mn, &matA, distCoeffs, 0, &_Ar);

	if (useExtrinsicGuess)
	{
		CvMat _r_temp = cvMat(rvec->rows, rvec->cols,
			CV_MAKETYPE(CV_64F, CV_MAT_CN(rvec->type)), param);
		CvMat _t_temp = cvMat(tvec->rows, tvec->cols,
			CV_MAKETYPE(CV_64F, CV_MAT_CN(tvec->type)), param + 3);
		cvConvert(rvec, &_r_temp);
		cvConvert(tvec, &_t_temp);
	}
	else
	{
		Mc = cvAvg(matM);//mean value
		cvReshape(matM, matM, 1, count);
		cvMulTransposed(matM, &_MM, 1, &_Mc);
		cvSVD(&_MM, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T);

		// initialize extrinsic parameters
		if (W[2] / W[1] < 1e-3 || count < 4)
		{
			// a planar structure case (all M's lie in the same plane)
			double tt[3], h[9], h1_norm, h2_norm;
			CvMat* R_transform = &matV;
			CvMat T_transform = cvMat(3, 1, CV_64F, tt);
			CvMat matH = cvMat(3, 3, CV_64F, h);
			CvMat _h1, _h2, _h3;

			if (V[2] * V[2] + V[5] * V[5] < 1e-10)
				cvSetIdentity(R_transform);

			if (cvDet(R_transform) < 0)
				cvScale(R_transform, R_transform, -1);

			cvGEMM(R_transform, &_Mc, -1, 0, 0, &T_transform, CV_GEMM_B_T);

			for (i = 0; i < count; i++)
			{
				const double* Rp = R_transform->data.db;
				const double* Tp = T_transform.data.db;
				const double* src = matM->data.db + i * 3;
				double* dst = _Mxy->data.db + i * 2;

				dst[0] = Rp[0] * src[0] + Rp[1] * src[1] + Rp[2] * src[2] + Tp[0];
				dst[1] = Rp[3] * src[0] + Rp[4] * src[1] + Rp[5] * src[2] + Tp[1];
			}

			cvFindHomography(_Mxy, _mn, &matH);

			if (cvCheckArr(&matH, CV_CHECK_QUIET))
			{
				cvGetCol(&matH, &_h1, 0);
				_h2 = _h1; _h2.data.db++;
				_h3 = _h2; _h3.data.db++;
				h1_norm = sqrt(h[0] * h[0] + h[3] * h[3] + h[6] * h[6]);
				h2_norm = sqrt(h[1] * h[1] + h[4] * h[4] + h[7] * h[7]);

				cvScale(&_h1, &_h1, 1. / MAX(h1_norm, DBL_EPSILON));
				cvScale(&_h2, &_h2, 1. / MAX(h2_norm, DBL_EPSILON));
				cvScale(&_h3, &_t, 2. / MAX(h1_norm + h2_norm, DBL_EPSILON));
				cvCrossProduct(&_h1, &_h2, &_h3);

				cvRodrigues2(&matH, &_r);
				cvRodrigues2(&_r, &matH);
				cvMatMulAdd(&matH, &T_transform, &_t, &_t);
				cvMatMul(&matH, R_transform, &matR);
			}
			else
			{
				cvSetIdentity(&matR);
				cvZero(&_t);
			}

			cvRodrigues2(&matR, &_r);
		}
		else
		{
			// non-planar structure. Use DLT method
			double* L;
			double LL[12 * 12], LW[12], LV[12 * 12], sc;
			CvMat _LL = cvMat(12, 12, CV_64F, LL);
			CvMat _LW = cvMat(12, 1, CV_64F, LW);
			CvMat _LV = cvMat(12, 12, CV_64F, LV);
			CvMat _RRt, _RR, _tt;
			CvPoint3D64f* M = (CvPoint3D64f*)matM->data.db;
			CvPoint2D64f* mn = (CvPoint2D64f*)_mn->data.db;

			matL = cvCreateMat(2 * count, 12, CV_64F);
			L = matL->data.db;

			for (i = 0; i < count; i++, L += 24)
			{
				double x = -mn[i].x, y = -mn[i].y;
				L[0] = L[16] = M[i].x;
				L[1] = L[17] = M[i].y;
				L[2] = L[18] = M[i].z;
				L[3] = L[19] = 1.;
				L[4] = L[5] = L[6] = L[7] = 0.;
				L[12] = L[13] = L[14] = L[15] = 0.;
				L[8] = x * M[i].x;
				L[9] = x * M[i].y;
				L[10] = x * M[i].z;
				L[11] = x;
				L[20] = y * M[i].x;
				L[21] = y * M[i].y;
				L[22] = y * M[i].z;
				L[23] = y;
			}

			cvMulTransposed(matL, &_LL, 1);
			cvSVD(&_LL, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T);
			_RRt = cvMat(3, 4, CV_64F, LV + 11 * 12);
			cvGetCols(&_RRt, &_RR, 0, 3);
			cvGetCol(&_RRt, &_tt, 3);
			if (cvDet(&_RR) < 0)
				cvScale(&_RRt, &_RRt, -1);
			sc = cvNorm(&_RR);
			cvSVD(&_RR, &matW, &matU, &matV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T);
			cvGEMM(&matU, &matV, 1, 0, 0, &matR, CV_GEMM_A_T);
			cvScale(&_tt, &_t, cvNorm(&matR) / sc);
			cvRodrigues2(&matR, &_r);
		}
	}

	cvReshape(matM, matM, 3, 1);
	cvReshape(_mn, _mn, 2, 1);

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
			cvProjectPoints2(matM, &_r, &_t, &matA, distCoeffs,
				_err, &_dpdr, &_dpdt, 0, 0, 0);
		}
		else
		{
			cvProjectPoints2(matM, &_r, &_t, &matA, distCoeffs,
				_err, 0, 0, 0, 0, 0);
		}
		cvSub(_err, _m, _err);
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
	CvSize imageSize, CvMat* cameraMatrix,
	double aspectRatio)
{
	Ptr<CvMat> matA, _b, _allH;

	int i, j, pos, nimages, ni = 0;
	double a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
	double H[9], f[2];
	CvMat _a = cvMat(3, 3, CV_64F, a);
	CvMat matH = cvMat(3, 3, CV_64F, H);//homography
	CvMat _f = cvMat(2, 1, CV_64F, f);//focal length
	/*whether the point data is legal*/
	assert(CV_MAT_TYPE(npoints->type) == CV_32SC1 &&
		CV_IS_MAT_CONT(npoints->type));
	nimages = npoints->rows + npoints->cols - 1;//the number of images

	

	matA = cvCreateMat(2 * nimages, 2, CV_64F);
	_b = cvCreateMat(2 * nimages, 1, CV_64F);
	/*the principle points*/
	a[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
	a[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;
	_allH = cvCreateMat(nimages, 9, CV_64F);

	// calculating 
	for (i = 0, pos = 0; i < nimages; i++, pos += ni)
	{
		double* Ap = matA->data.db + i * 4;
		double* bp = _b->data.db + i * 2;
		ni = npoints->data.i[i];
		double h[3], v[3], d1[3], d2[3];
		double n[4] = { 0,0,0,0 };
		CvMat _m, matM;
		cvGetCols(objectPoints, &matM, pos, pos + ni);
		cvGetCols(imagePoints, &_m, pos, pos + ni);
		

		cvFindHomography(&matM, &_m, &matH);
		memcpy(_allH->data.db + i * 9, H, sizeof(H));
		/*Translation change*/
		H[0] -= H[6] * a[2]; H[1] -= H[7] * a[2]; H[2] -= H[8] * a[2];
		H[3] -= H[6] * a[5]; H[4] -= H[7] * a[5]; H[5] -= H[8] * a[5];
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
	/*solve the linear system matA*_f=_b, where _f is the reciprocal of focal length, matA is the inverse of rotation matrix*/
	cvSolve(matA, _b, &_f, CV_NORMAL + CV_SVD);//Least-squares solution
	a[0] = sqrt(fabs(1. / f[0]));
	a[4] = sqrt(fabs(1. / f[1]));
	if (aspectRatio != 0)
	{
		double tf = (a[0] + a[4]) / (aspectRatio + 1.);
		a[0] = aspectRatio * tf;
		a[4] = tf;
	}

	cvConvert(&_a, cameraMatrix);
}
CV_IMPL double myCalibrateCamera2(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
	CvMat* rvecs, CvMat* tvecs, int flags, CvTermCriteria termCrit)
{
	const int NINTRINSIC = 12;//number of internal parameters
	Ptr<CvMat> matM, _m, _Ji, _Je, _err;
	/*Levenberg-Marquardt Algorithm*/
	CvLevMarq solver;
	double reprojErr = 0;//return value

	double A[9], k[8] = { 0,0,0,0,0,0,0,0 };
	CvMat matA = cvMat(3, 3, CV_64F, A), _k;
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
	/*Levenberg-Marquardt Algorithm */
	solver.init(nparams, 0, termCrit);
	{
		double* param = solver.param->data.db;
		uchar* mask = solver.mask->data.ptr;
		param[0] = A[0]; param[1] = A[4]; //focal length
		param[2] = A[2]; param[3] = A[5];//principle point
		/*disortions*/
		param[4] = k[0]; param[5] = k[1]; param[6] = k[2]; param[7] = k[3];
		param[8] = k[4]; param[9] = k[5]; param[10] = k[6]; param[11] = k[7];
		if (flags & CALIB_FIX_ASPECT_RATIO)
			mask[0] = 0;
		if (flags & CV_CALIB_FIX_FOCAL_LENGTH)
			mask[0] = mask[1] = 0;
		if (flags & CV_CALIB_FIX_PRINCIPAL_POINT)
			mask[2] = mask[3] = 0;
		if (flags & CV_CALIB_ZERO_TANGENT_DIST)
		{
			param[6] = param[7] = 0;
			mask[6] = mask[7] = 0;
		}
		if (!(flags & CV_CALIB_RATIONAL_MODEL))
			flags |= CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 + CV_CALIB_FIX_K6;
		if (flags & CV_CALIB_FIX_K1)
			mask[4] = 0;
		if (flags & CV_CALIB_FIX_K2)
			mask[5] = 0;
		if (flags & CV_CALIB_FIX_K3)
			mask[8] = 0;
		if (flags & CV_CALIB_FIX_K4)
			mask[9] = 0;
		if (flags & CV_CALIB_FIX_K5)
			mask[10] = 0;
		if (flags & CV_CALIB_FIX_K6)
			mask[11] = 0;
	}
	//2. initialize extrinsic parameters
	for ( i = 0,pos=0; i < nimages; i++,pos+=ni)
	{
		CvMat _Mi, _mi, _ri, _ti;
		ni = npoints->data.i[i*npstep];

		cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);//rotation vector
		cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);//translation vector

		cvGetCols(matM, &_Mi, pos, pos + ni);
		cvGetCols(_m, &_mi, pos, pos + ni);
		cvFindExtrinsicCameraParams2(&_Mi, &_mi, &matA, &_k, &_ri, &_ti);
	}
	// 3. run the optimization
	for (;;)
	{
		const CvMat* _param = 0;
		CvMat *_JtJ = 0, *_JtErr = 0;
		double* _errNorm = 0;
		bool proceed = solver.updateAlt(_param, _JtJ, _JtErr, _errNorm);
		double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;

		if (flags & CV_CALIB_FIX_ASPECT_RATIO)
		{
			param[0] = param[1] * aspectRatio;
			pparam[0] = pparam[1] * aspectRatio;
		}

		A[0] = param[0]; A[4] = param[1]; A[2] = param[2]; A[5] = param[3];
		k[0] = param[4]; k[1] = param[5]; k[2] = param[6]; k[3] = param[7];
		k[4] = param[8]; k[5] = param[9]; k[6] = param[10]; k[7] = param[11];

		if (!proceed)
			break;

		reprojErr = 0;

		for (i = 0, pos = 0; i < nimages; i++, pos += ni)
		{
			CvMat _Mi, _mi, _ri, _ti, _dpdr, _dpdt, _dpdf, _dpdc, _dpdk, _mp, _part;
			ni = npoints->data.i[i*npstep];

			cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);
			cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);

			cvGetCols(matM, &_Mi, pos, pos + ni);
			cvGetCols(_m, &_mi, pos, pos + ni);

			_Je->rows = _Ji->rows = _err->rows = ni * 2;
			cvGetCols(_Je, &_dpdr, 0, 3);
			cvGetCols(_Je, &_dpdt, 3, 6);
			cvGetCols(_Ji, &_dpdf, 0, 2);
			cvGetCols(_Ji, &_dpdc, 2, 4);
			cvGetCols(_Ji, &_dpdk, 4, NINTRINSIC);
			cvReshape(_err, &_mp, 2, 1);

			if (_JtJ || _JtErr)
			{
				cvProjectPoints2(&_Mi, &_ri, &_ti, &matA, &_k, &_mp, &_dpdr, &_dpdt,
					(flags & CV_CALIB_FIX_FOCAL_LENGTH) ? 0 : &_dpdf,
					(flags & CV_CALIB_FIX_PRINCIPAL_POINT) ? 0 : &_dpdc, &_dpdk,
					(flags & CV_CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0);
			}
			else
				cvProjectPoints2(&_Mi, &_ri, &_ti, &matA, &_k, &_mp);

			cvSub(&_mp, &_mi, &_mp);

			if (_JtJ || _JtErr)
			{
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

			double errNorm = cvNorm(&_mp, 0, CV_L2);
			reprojErr += errNorm * errNorm;
		}
		if (_errNorm)
			*_errNorm = reprojErr;
	}
	// 4. store the results
	cvConvert(&matA, cameraMatrix);
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
				cvRodrigues2(&src, &matA);
				cvConvert(&matA, &dst);
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
	double reprojErr = myCalibrateCamera2(&c_objPt, &c_imgPt, &c_npoints, imageSize,
		&c_cameraMatrix, &c_distCoeffs, &c_rvecM,
		&c_tvecM, flags, criteria);
	return 0;
}
