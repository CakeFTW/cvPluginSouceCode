#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const int GRIDSIZE = 3;

int blue = 0;
int green = 0;
int red = 227;
bool timeKeeping = true;

void createTrackBars();

struct cVector {
	int x;
	int y;
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

	double t = (double)getTickCount();

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
		t = (double)getTickCount();
		system("CLS");
		
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:readCam	: " << t << endl;
			t = (double)getTickCount();
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
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:RGBnorm	: " << t << endl;
			t = (double)getTickCount();
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
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:Threshold	: " << t << endl;
			t = (double)getTickCount();
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
					blobs.push_back(currentBlob);
					blobs.back().nr = col;
					dropFire(p + j, blobs.back(), nCols, i , j);
					col -= 10;
					if (col < 20) {
						col = 245;
					}
				}
			}
		}
		
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:BlobDetect	: " << t << endl;
			t = (double)getTickCount();
		}

		

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
			int largestX = 0;
			int smallestX = 10000;
			int largestY = 0;
			int smallestY = 10000;
			int radiusDist = 0;

			for (auto &v : i.list) {
				if (v.x < smallestX) { smallestX = v.x; }
				if (v.x > largestX) { largestX = v.x; }
				if (v.y < smallestY) { smallestY = v.y; }
				if (v.y > largestY) { largestY = v.y; }
				centerX += v.x;
				centerY += v.y;
			}
			counter++;
			/*cout << "new blob" << endl;
			cout << size << endl;
			cout << centerX << endl;
			cout << centerY << endl;*/
		
			cout << "radi dist " << radiusDist << endl;
			centerX = centerX / (float)size;
			centerY = centerY / (float)size;
			radiusDist = ((float)((float)(largestX - centerX) + (centerX - smallestX) + (largestY - centerY)+(centerY - smallestY)))/4;
			i.center.x = centerX;
			i.center.y = centerY; 
			/*cout << centerX << endl;
			cout << centerY << endl;*/
			circle(imgOriginal, Point(centerX-GRIDSIZE, centerY-GRIDSIZE), radiusDist, Scalar(0, 0, 255), 5);
			radiusDist = (float)radiusDist * 0.3;
			radiusDist = radiusDist * radiusDist;
			//find closest pixel
			int dist = 10000;
			vector<cVector> points;
			for (auto &v : i.list) {
				dist = (v.x - i.center.x) * (v.x - i.center.x) + (v.y - i.center.y) * (v.y - i.center.y);
				if (dist < radiusDist) {
					points.push_back(v);
				}
			}
			i.rotation.x = 0;
			i.rotation.y = 0;
			for (auto &p : points) {
				i.rotation.x += p.x-centerX;
				i.rotation.y += p.y - centerY;
			}
			if (points.size() != 0) {
				i.rotation.x = i.rotation.x / (float)points.size();
				i.rotation.y = i.rotation.y / (float)points.size();

				cout << "----ROTATION------" << endl;
				cout << i.rotation.x << ":" << i.rotation.y << endl;

				cout << "close points:" << points.size() << "withing" << radiusDist << endl;
				line(imgOriginal, Point(i.center.x - GRIDSIZE, i.center.y - GRIDSIZE), Point(i.center.x + i.rotation.x - GRIDSIZE, i.center.y + i.rotation.y - GRIDSIZE), Scalar(0, 255, 0), 3);
			}
		}

		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:BlobAnalysis: " << t << endl;
			t = (double)getTickCount();
		}

		if (counter != 0) {
			cout << "nr of objects : " << counter << "off :" << blobs.size() << endl;
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

void createTrackBars() {
	namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("R", "Control", &red, 255);
	cvCreateTrackbar("G", "Control", &green, 255);
	cvCreateTrackbar("B", "Control", &blue, 255);
}