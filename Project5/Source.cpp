#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

struct cVector {
	uint x;
	uint y;
	cVector(int _x, int _y) {
		x = _x;
		y = _y;
	}
	cVector() {
		x = 0;
		y = 0;
	}
};

struct glyphObj {
	vector<cVector> list;
	int nrOfPixels;
	cVector bBoxStart;
	cVector bBowEnd;
	cVector center;
	cVector rotation;
	int nr;
};

void dropFire(uchar * pixel, glyphObj &store, int width, int y, int x ) {
	*pixel = store.nr;
	store.list.push_back(cVector(x, y));
	if (x < 0 || y < 0) {
		cout << "WTF:::::" << x << " : " << y << endl;
	}
	if (*(pixel + 1) == 255) {
		dropFire((pixel + 1), store,  width, y, x+1);
	}
	if(*(pixel + width) == 255){
		dropFire(pixel + width, store, width, y +1, x);
	}
	
	if (*(pixel - 1) == 255) {
		dropFire((pixel - 1), store,  width, y , x -1);
	}
	
	if (*(pixel - width) == 255) {
		dropFire(pixel - width, store, width, y-1 , x);
	}
}


int main() {


	Mat cameraFrame;

	VideoCapture cap(0); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}
	
	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
												/*
	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;


	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);
	*/


	int green= 0;
	int red = 227;
	

	cvCreateTrackbar("R", "Control", &red, 255); //Value (0 - 255)
	cvCreateTrackbar("G", "Control", &green, 255);
	
	cap.set( CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set( CV_CAP_PROP_FRAME_HEIGHT, 480);
	while (true)
	{
		Mat imgOriginal;
		
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		//imgOriginal = imread("blobs.jpg", CV_LOAD_IMAGE_COLOR);		
		
		Mat rgbNorm(imgOriginal.rows, imgOriginal.cols, CV_8UC3);

		//convert to normalized rgb space
		int nRows = rgbNorm.rows;
		int nCols = rgbNorm.cols*3;
		float sum = 0;
		uchar * p;
		uchar * cp; 
		for (int i = 0; i < nRows; i++) {
			p = imgOriginal.ptr<uchar>(i);
			cp = rgbNorm.ptr<uchar>(i);
			for (int j = 0; j < nCols; j += 3) {
				sum = p[j] + p[j + 1] + p[j + 2];
				if (sum < 150) {
					cp[j] = 0;
					cp[j + 1] = 0;
					cp[j + 2] = 0;
					continue;
				}
				cp[j] = (uchar)(p[j]*255/sum);
				cp[j+1] = (uchar)(p[j+1] * 255 / sum);
				cp[j + 2] = (uchar)(p[j + 2] * 255 / sum);
			}
		}

		//threshold the image
		Mat thresImg(imgOriginal.rows, imgOriginal.cols, CV_8UC1);
		
		nRows = rgbNorm.rows;
		nCols = rgbNorm.cols;

		
		for (int i = 0; i < nRows; i++) {
			p = rgbNorm.ptr<uchar>(i);
			cp = thresImg.ptr<uchar>(i);
		
			for (int j = 0 ; j < nCols; j += 1) {
				int color = j * 3;
				if ((p[color+1] - green)*(p[color + 1] - green) + (p[color + 2] - red)*(p[color + 2] - red) < 1600) {
					cp[j] = 255;
					continue;
				}
				cp[j] = 0;
			}
		}

		copyMakeBorder(thresImg, thresImg, 3, 3, 3, 3, BORDER_CONSTANT, 0);
		
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(2, 2));
		morphologyEx(thresImg, thresImg, MORPH_DILATE, element );

		nRows = thresImg.rows;
		nCols = thresImg.cols;

		vector<glyphObj> blobs;
		int col = 245;
		for (int i = 1; i < nRows-1; i++) {
			p = thresImg.ptr<uchar>(i);
			for (int j = 1; j < nCols-1; j++) {
				if (p[j+2] == 255) {
					glyphObj currentBlob;
					currentBlob.nr = col;
					dropFire(p + j, currentBlob, nCols, i , j);
					col -= 10;
					if (col < 20) {
						col = 245;
					}
					blobs.push_back(currentBlob);
				}
			}
		}
		
		//blob detection
		
		//printing out objects

		for ( auto &i : blobs) {
			//find center
			int size;
			size = i.list.size();
			if (size < 500 || size > 6000) { continue; }
			long centerX = 0;
			long centerY = 0;
			for (auto &v : i.list) {
				centerX += v.x;
				centerY += v.y;
			}
			cout << "new blob" << endl;
			cout << size << endl;
			cout << centerX << endl;
			cout << centerY << endl;
			centerX = centerX / (float)size;
			centerY = centerY / (float)size;
			i.center.x = centerX;
			i.center.y = centerY; 
			cout << centerX << endl;
			cout << centerY << endl;
			circle(imgOriginal, Point(centerX, centerY), 50, Scalar(0, 0, 255), 5);
		}
	
		cv::imshow("original", imgOriginal);
		cv::imshow("normalized", rgbNorm);
		cv::imshow("thresholded", thresImg);

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}






	return 0;
}