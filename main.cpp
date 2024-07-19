#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>

using namespace cv;

int main() {
	//��ȡѵ����haar
	CascadeClassifier clsface;
	std::string strFileFace("D:\\cppsoft\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");

	//�ж�xml�ļ��Ƿ�ɹ�
	if (!clsface.load(strFileFace)) {
		std::cerr << "xml open err ! " << std::endl;
		return -1;
	}

	Mat msrc_image, mdst_image, mgray_image;

	msrc_image = imread("test3.png");

	mdst_image = msrc_image.clone();

	//��ͼ�����ɻҶȣ�����߼��Ч��s
	cv::cvtColor(msrc_image,mgray_image,COLOR_BGR2BGRA);

	//ʵ���������
	std::vector<cv::Rect> vface_rect;
	
	//���÷�����
	clsface.detectMultiScale(mgray_image, vface_rect, 1.1, 3, 0);
	std::cout << "�Ѿ�������ͼ����������Ϊ:" << vface_rect.size() << std::endl;

	if (vface_rect.size()) {
		for (int i = 0; i < vface_rect.size(); i++) {
			//��ͼ���ϵ�������ע����
			cv::rectangle(mdst_image, vface_rect[i], cv::Scalar(0, 0, 255), 1);

			//��ȡ��������ͼ��
			cv::Mat face_img = mdst_image(vface_rect[i]);
			imshow("test : face_img",face_img);
		}
	}

	cv::imwrite("facedetectionfile.png", mdst_image);

	cv::imshow("test �� ", mdst_image);
	waitKey(0);
	return 0;
}