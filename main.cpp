#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>

using namespace cv;

int main() {
	//读取训练器haar
	CascadeClassifier clsface;
	std::string strFileFace("D:\\cppsoft\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");

	//判断xml文件是否成功
	if (!clsface.load(strFileFace)) {
		std::cerr << "xml open err ! " << std::endl;
		return -1;
	}

	Mat msrc_image, mdst_image, mgray_image;

	msrc_image = imread("test3.png");

	mdst_image = msrc_image.clone();

	//将图像生成灰度，以提高检查效率s
	cv::cvtColor(msrc_image,mgray_image,COLOR_BGR2BGRA);

	//实现人脸检测
	std::vector<cv::Rect> vface_rect;
	
	//调用分类器
	clsface.detectMultiScale(mgray_image, vface_rect, 1.1, 3, 0);
	std::cout << "已经检测出该图像人脸个数为:" << vface_rect.size() << std::endl;

	if (vface_rect.size()) {
		for (int i = 0; i < vface_rect.size(); i++) {
			//将图像上的人脸标注矩形
			cv::rectangle(mdst_image, vface_rect[i], cv::Scalar(0, 0, 255), 1);

			//获取人脸矩形图像
			cv::Mat face_img = mdst_image(vface_rect[i]);
			imshow("test : face_img",face_img);
		}
	}

	cv::imwrite("facedetectionfile.png", mdst_image);

	cv::imshow("test ： ", mdst_image);
	waitKey(0);
	return 0;
}