#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const int GRIDSIZE = 1;

int blue = 0;
int green = 0;
int red = 227;

void createTrackBars();

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


	if (*(pixel + GRIDSIZE) == 255) {
		dropFire(pixel + GRIDSIZE, store,  width, y, x+GRIDSIZE);
	}
	if(*(pixel + (width*GRIDSIZE)) == 255){
		dropFire(pixel + (width*GRIDSIZE), store, width, y +GRIDSIZE, x);
	}
	
	if (*(pixel - GRIDSIZE) == 255) {
		dropFire(pixel - GRIDSIZE, store,  width, y , x -GRIDSIZE);
	}
	
	if (*(pixel - (width*GRIDSIZE)) == 255) {
		dropFire(pixel - (width*GRIDSIZE), store, width, y-GRIDSIZE , x);
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
	
	createTrackBars();

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

	//	imgOriginal = imread("white.png", CV_LOAD_IMAGE_COLOR);		
		
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

		copyMakeBorder(thresImg, thresImg, GRIDSIZE +1, GRIDSIZE +1 , GRIDSIZE +1 , GRIDSIZE +1 , BORDER_CONSTANT, 0);
		
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(2, 2));
		morphologyEx(thresImg, thresImg, MORPH_DILATE, element );

		//blob detection
		nRows = thresImg.rows;
		nCols = thresImg.cols;

		//create lookuptable to get acces to each row;
		uchar ** lut = new uchar*[nRows];
		for (int i = 0; i < nRows; i++) {
			*(lut + i) = thresImg.ptr<uchar>(i);
		}

		vector<glyphObj> blobs;
		int col = 245;
		for (int i = GRIDSIZE + 1; i < nRows-GRIDSIZE - 1; i += GRIDSIZE) {
			p = thresImg.ptr<uchar>(i);
			for (int j = GRIDSIZE; j < nCols-GRIDSIZE; j += GRIDSIZE) {
				if (p[j] == 255) {
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
		
		Mat thresImgHolder(nRows, nCols, CV_8UC1);
		

		

		//printing out objects
		int counter = 0;
		int minSize = 400/GRIDSIZE;
		int maxSize = 4000/GRIDSIZE;
		for ( auto &i : blobs) {

			//find center
			int size;
			size = i.list.size();
			if (size < minSize || size > maxSize) { continue; }
			long centerX = 0;
			long centerY = 0;
			for (auto &v : i.list) {
				centerX += v.x;
				centerY += v.y;
				thresImgHolder.at<uchar>(v.y, v.x) = i.nr;
			}
			counter++;
			/*cout << "new blob" << endl;
			cout << size << endl;
			cout << centerX << endl;
			cout << centerY << endl;*/
			centerX = centerX / (float)size;
			centerY = centerY / (float)size;
			i.center.x = centerX;
			i.center.y = centerY; 
			/*cout << centerX << endl;
			cout << centerY << endl;*/
			circle(imgOriginal, Point(centerX-GRIDSIZE, centerY-GRIDSIZE), 50, Scalar(0, 0, 255), 5);

			//find closest pixel
			int ClosesDist = 10000;
			int dist = 10000;
			for (auto &v : i.list) {
				dist = (v.x - i.center.x) * (v.x - i.center.x) + (v.y - i.center.y) * (v.y - i.center.y);
				if (dist < ClosesDist) {
					ClosesDist = dist;
					i.rotation = v;
				}
			}
			i.rotation.x = i.rotation.x - i.center.x;
			i.rotation.y = i.rotation.y - i.center.y;
			line(imgOriginal, Point(i.center.x-GRIDSIZE, i.center.y-GRIDSIZE), Point(i.center.x + 4*i.rotation.x-GRIDSIZE, i.center.y + 4*i.rotation.y-GRIDSIZE), Scalar(0, 255, 0), 3);
			
		}
		if (counter != 0) {
			cout << "nr of objects : " << counter << "off :" << blobs.size() << endl;
		}
		cv::imshow("original", imgOriginal);
		cv::imshow("normalized", thresImgHolder);
		cv::imshow("thresholded", thresImg);

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}






	return 0;
}

void createTrackBars() {
	namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("R", "Control", &red, 255);
	cvCreateTrackbar("G", "Control", &green, 255);
	cvCreateTrackbar("B", "Control", &blue, 255);
}