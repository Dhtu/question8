# question8
## Readme
本项目分为两个部分: 
* question8
* mycalibration

### question8
#### 数据存取
该模块用c++写成, 主要调用了opencv库函数. 在运行程序之前, 将数据集的路径保存在data.txt中, 将数据保存在data文件夹中, 将标定结果保存在calibration_result2.txt文件中. 
#### 读取角点
由于张正友标定要求读取黑白格的角点, 主要通过调用findChessboardCorners函数获取角点
```cpp
if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n";
			exit(1);
		}
```
#### 开始标定
经过数据的初始化后, 由于opencv中已经存在标定函数, 本项目可以在此处调用本身的程序, 即
```cpp
/*calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);*/
```
同时也可以调用将在后续进一步解释的mycalibrateCamera函数
```cpp
mycalibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat);
```
#### 保存结果
最后保存数据
```cpp
/*save the reslut*/
	std::cout << "start to save the reslut………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* the rotation matrix */
	fout << "internal matrix:" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "disortion coefficient:\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		fout << "The transpotation matrix of No." << i + 1 << endl;
		fout << tvecsMat[i] << endl;
		/* transform the rotation vector to rotation matrix */
		Rodrigues(rvecsMat[i], rotation_matrix);
		fout << "The rotation matrix of No." << i + 1 << endl;
		fout << rotation_matrix << endl;
		fout << "The rotation vector of No." << i + 1 << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	std::cout << "completed" << endl;
	fout << endl;
```
#### 以偏差值的形式评估标定结果
用得到的结果调用projectPoints函数再次投影后计算每幅图像的偏差
```cpp
for (size_t i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* Through the obtained internal and external camera parameters, the three-dimensional points
		in the space are re-projected to obtain a new projection point. */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* Calculate the error between the new projection point and the old projection point*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << "The average error of the No." << i + 1 << "\t" << err << "pixels" << endl;
		fout << "The average error of the No." << i + 1 << "\t" << err << "\t" << "pixels" << endl;
	}
```
###mycalibration
这一模块主要是mycalibratecamera函数的实现, 主要包括四个函数, mycalibrateCamera, myCalibrateCamera2, myInitIntrinsicParams2D, myFindExtrinsicCameraParams2. 其实质为<Z. Zhang, “A flexible new technique for camera calibration,” IEEE Transactions>一文的实现. 
#### mycalibrateCamera函数
本函数为借口函数, 将c++的数据结构转变为c的数据结构, 并将数据传入myCalibrateCamera2这一c语言函数
```cpp
/*Convert to C format*/
	CvMat c_objPt = objPt, c_imgPt = imgPt, c_npoints = npoints;
	CvMat c_cameraMatrix = cameraMatrix, c_distCoeffs = distCoeffs;
	CvMat c_rvecM = rvecM, c_tvecM = tvecM;

	/*start to calculate*/
	double reprojErr = myCalibrateCamera2(&c_objPt, &c_imgPt, &c_npoints, imageSize,
		&c_cameraMatrix, &c_distCoeffs, &c_rvecM,
		&c_tvecM, criteria);
```
#### myCalibrateCamera2函数
从这里开始为C语言对该论文的实现, 
##### 分配内存
首先是定义相关数据结构, 分配相机矩阵, 图像点集, 观察点集, 雅克比矩阵, 误差矩阵内存
```cpp
double Camera_Parameters[9], k[8] = { 0,0,0,0,0,0,0,0 };
	CvMat CaMat = cvMat(3, 3, CV_64F, Camera_Parameters), _k;
	int i, nimages/*the number of images*/, maxPoints = 0, total = 0;/*the total number of points*/
	int ni = 0, pos, nparams, npstep, cn;
	double aspectRatio = 0.;
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
```
##### 初始化内参矩阵和L-M算法
这里调用了myInitIntrinsicParams2D以获得相机内参矩阵, 
```cpp
myInitIntrinsicParams2D(matM, _m, npoints, imageSize, &CaMat);
```
并定义CvLevMarq类的实例solver, 使用.init方法进行初始化
##### 初始化外参(旋转向量与平移向量)
通过循环调用myFindExtrinsicCameraParams2函数获得所有图片对应的外参.
```cpp
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
```
##### 使用Levenberg-Marquardt Algorithm优化相机参数
使用.updateAlt方法进行迭代
```cpp
bool proceed = solver.updateAlt(_param, _JtJ, _JtErr, _errNorm);
```
将迭代得到的参数赋给新的相机参数:
```cpp
Camera_Parameters[0] = param[0]; Camera_Parameters[4] = param[1];//focal length
		Camera_Parameters[2] = param[2]; Camera_Parameters[5] = param[3];//principle point
		k[0] = param[4]; k[1] = param[5]; k[2] = param[6]; k[3] = param[7];
		k[4] = param[8]; k[5] = param[9]; k[6] = param[10]; k[7] = param[11];
```
通过该方法的返回值控制迭代
```cpp
if (!proceed)
			break;
```
对每一张图进行映射, 得到对应的雅克比矩阵, 这里调用了opencv的函数
```cpp
if (test = (_JtJ || _JtErr))
			{
				/*Calculate the Jacobian matrix*/
				cvProjectPoints2(&_ObjectP, &_ri, &_ti, &CaMat, &_k, &_imageP2, &_dpdr, &_dpdt, &_dpdf, &_dpdc, &_dpdk, 0);
				test = 0;
			}
```
计算雅克比矩阵与自身转置的乘积
```cpp
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
```
##### 最终将计算所得参数返回

#### myInitIntrinsicParams2D函数
不考虑x轴与y轴的倾斜度, 此函数输入数据点集, 计算内参矩阵:
##### 用平移变换去除principle point的影响
```cpp
camera_parameters[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
	camera_parameters[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;
	/*Translation change*/
		H[0] -= H[6] * camera_parameters[2]; H[1] -= H[7] * camera_parameters[2]; H[2] -= H[8] * camera_parameters[2];
		H[3] -= H[6] * camera_parameters[5]; H[4] -= H[7] * camera_parameters[5]; H[5] -= H[8] * camera_parameters[5];
```
##### 计算焦距
```cpp
/*Then the H is :
						[ f_1 0   0 ]
						[ 0   f_2 0 ] *  [R|t]
						[ 0   0   1 ]
		  According to the paper, Using the knowledge that r1 and r2 are orthonormal, we have:
						h_T_1 A−T A−1h_2 = 0 (3)
						h_T_1 A−T A−1h_1 = h_T_2 A−T A−1h_2 . (4)

		*/
```
##### 根据论文, 构造矩阵方程
```cpp
/*  we have:
			[h11h21,      h12h22,      h13h23      ] * [1/f_1^2] = 0
			[h11^2-h22^2  h12^2-h22^2  h13^2-h23^2 ]   [1/f_2^2]
													   [1	   ]
			That is to say:
			[h11h21,      h12h22,	 ] * [1/f_1^2] = [-h13h23    ]
			[h11^2-h22^2  h12^2-h22^2]   [1/f_2^2] = [h13^2-h23^2]
			*/
```
##### 解矩阵方程
调用cvSolve函数得到上述方程的最小二乘解, 并计算得到焦距
```cpp
cvSolve(matA, _b, &_focal_length, CV_NORMAL + CV_SVD);//Least-squares solution
	camera_parameters[0] = sqrt(fabs(1. / focal_len[0]));
	camera_parameters[4] = sqrt(fabs(1. / focal_len[1]));
```
#### myFindExtrinsicCameraParams2函数
该函数接受一副图片的对应点集, 并返回旋转向量与平移向量
##### 计算初始向量
根据论文, 从得到homography开始, 计算初始的向量
```cpp
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
```
##### 优化参数
使用Levenberg-Marquardt Algorithm优化不带畸变的相机参数
调用cvProjectPoints2函数获取雅克比矩阵:
```cpp
cvGetCols(matJ, &_dpdr, 0, 3);
			cvGetCols(matJ, &_dpdt, 3, 6);
			cvProjectPoints2(objectPoints, &_r, &_t, A, distCoeffs,
				_err, &_dpdr, &_dpdt, 0, 0, 0);
```
## Reference
[1]Z. Zhang, “A flexible new technique for camera calibration,” IEEE Transactions.
[2]OpenCV document.
[3]OpenCV sourse code.
