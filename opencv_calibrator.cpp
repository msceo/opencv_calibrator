#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <cctype>
#include <vector>
#include <iostream>
#include <filesystem>

static double computeReprojectionErrors(
	const std::vector<std::vector<cv::Point3f>>& objectPoints,
	const std::vector<std::vector<cv::Point2f>>& imagePoints,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs
)
{
	std::vector<cv::Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;


	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
		int n = (int)objectPoints[i].size();

		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

auto calibrate(int numCornersHor, int numCornersVer, int numSquares, std::string path_chessboard, std::string path_save) {
	std::vector<cv::Point3f> obj;
	//存放每张图的角点坐标，并存入obj中(物点)
	for (int i = 0; i < numCornersHor; i++)
		for (int j = 0; j < numCornersVer; j++)
			obj.push_back(cv::Point3f((float)j * numSquares, (float)i * numSquares, 0));

	std::vector<std::vector<cv::Point2f>> imagePoints; //像点
	std::vector<std::vector<cv::Point3f>> objectPoints; //物点
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);//进行亚像素精度的调整，获得亚像素级别的角点坐标
	//发现与绘制棋盘格
	//遍历每张图片
	cv::Size s;
	//像点
	for (const auto& file : std::filesystem::directory_iterator(path_chessboard)) {
		cv::Mat image = cv::imread(file.path().string());
		s = image.size();
		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); //转灰度
		std::vector<cv::Point2f> corners;
		auto ret = cv::findChessboardCorners(gray, cv::Size(numCornersVer, numCornersHor), corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);//该函数的功能就是判断图像内是否包含完整的棋盘图，若能检测完全，就把他们的角点坐标(从上到下，从左到右)记录，并返回true，否则为false，CALIB_CB_FILTER_QUADS用于去除检测到的错误方块。
		if (ret) {
			std::cout << "Valid image: " << file.path().string() << std::endl;
			cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria); //用于发现亚像素精度的角点位置
			cv::drawChessboardCorners(image, cv::Size(numCornersVer, numCornersHor), corners, ret); //将每一个角点出做标记，此为物点对应的像点坐标
			imagePoints.push_back(corners); //将角点坐标存入imagePoints中，此为像点坐标
			objectPoints.push_back(obj); //把存放在每张图的所有角点坐标，存在objectPoints中，物点坐标
			cv::imshow("calibration-demo", image);
			cv::waitKey(500);
		}
	}


	//计算内参与畸变系数
	cv::Mat intrinsic = cv::Mat(3, 3, CV_32FC1);
	cv::Mat distCoeffs;//畸变矩阵
	std::vector<cv::Mat> rvecs;//旋转向量R
	std::vector<cv::Mat> tvecs;//平移向量T
	//内参矩阵
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;
	calibrateCamera(objectPoints, imagePoints, s, intrinsic, distCoeffs, rvecs, tvecs);
	cv::FileStorage fs(path_save, cv::FileStorage::WRITE);
	fs << "intrinsic" << intrinsic; //存放内参矩阵
	fs << "distCoeffs" << distCoeffs; //存放畸变矩阵
	fs << "board_width" << numCornersVer; //存放标定板长度信息
	fs << "board_height" << numCornersHor; //存放标定板宽度信息
	fs << "square_size" << numSquares/1000; //存放标定板格子尺寸信息
	fs << "R" << rvecs;//R
	fs << "T" << tvecs;//T
	auto error = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, intrinsic, distCoeffs);
	return error;
}

int main(int argc, char** argv)
{
	std::cout << "=======================================================" << std::endl;
	std::cout << "          CameraCalibrator Powered by OpenCV" << std::endl;
	std::cout << "=======================================================" << std::endl;

	std::string path_chessboard = "D:\\calib\\chess";
	std::string path_save = "D:\\calib\\calibration.yaml";

	// 定义变量
	int numCornersHor = 16; //heigh
	int numCornersVer = 11; //width
	numCornersHor -= 1; numCornersVer -= 1;
	int numSquares = 10; //单位mm
	std::cout << "Image path: " << path_chessboard << std::endl;
	std::cout << "Cheeseboard size: " << numCornersHor+1 << "x" << numCornersVer+1 << std::endl;
	std::cout << "Sqaure size: " << numSquares << "mm" << std::endl;
	std::cout << "=======================================================" << std::endl;
	auto error = calibrate(numCornersHor, numCornersVer, numSquares, path_chessboard, path_save);
	std::cout << "=======================================================" << std::endl;
	std::cout << "Process Complete" << std::endl;
	std::cout << "Calibration Error: " << error << std::endl;
	std::cout << "File saved at: " << path_save << std::endl;
	std::cout << "=======================================================" << std::endl;
}
