#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const int GRIDSIZE = 1;

int blue = 0;
int green = 40;
int red = 180;
bool timeKeeping = false;
const float discrimHW = 0.2;

void createTrackBars();


//Finding border edge
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
int largest_area = 0;
int largest_contour_index = 0;
Rect bounding_rect;
vector<vector<Point>> biggestContour;
vector<Vec4i> permHiearchy;


void findBorder(Mat src);

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

void grassFireBlobDetection(Mat &biImg, vector<glyphObj> &blobs) {
	int nRows = biImg.rows;
	int nCols = biImg.cols;
	uchar * p;

	int col = 245;
	for (int i = GRIDSIZE + 1; i < nRows - GRIDSIZE - 1; i += GRIDSIZE) {
		p = biImg.ptr<uchar>(i);
		for (int j = GRIDSIZE; j < nCols - GRIDSIZE; j += GRIDSIZE) {
			if (p[j] == 255) {
				glyphObj currentBlob;
				blobs.push_back(currentBlob);
				blobs.back().nr = col;
				dropFire(p + j, blobs.back(), nCols, i, j);
				col -= 10;
				if (col < 20) {
					col = 245;
				}
			}
		}
	}
}


void blobAnalysis(vector<glyphObj> &blobs, Mat &drawImg) {


	//printing out objects
	int minSize = 30 / GRIDSIZE;
	int maxSize = 4000 / GRIDSIZE;
	for (auto &i : blobs) {
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
		float heightWidth = ((largestX - smallestX) / (float)(largestY - smallestY));

		//check discriminate basedd on height width relation
		if (heightWidth > (1 + discrimHW) || heightWidth <(1 - discrimHW)) { continue; }
		centerX = centerX / (float)size;
		centerY = centerY / (float)size;
		radiusDist = ((float)((float)(largestX - centerX) + (centerX - smallestX) + (largestY - centerY) + (centerY - smallestY))) / 4;
		i.center.x = centerX;
		i.center.y = centerY;
		circle(drawImg, Point(centerX - GRIDSIZE, centerY - GRIDSIZE), radiusDist, Scalar(0, 0, 255), 5);
		radiusDist = (float)radiusDist * 0.5;
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
			i.rotation.x += p.x - centerX;
			i.rotation.y += p.y - centerY;
		}
		if (points.size() != 0) {
			i.rotation.x = i.rotation.x / (float)points.size();
			i.rotation.y = i.rotation.y / (float)points.size();

			line(drawImg, Point(i.center.x - GRIDSIZE, i.center.y - GRIDSIZE), Point(i.center.x + i.rotation.x - GRIDSIZE, i.center.y + i.rotation.y - GRIDSIZE), Scalar(0, 255, 0), 2);

			//use vectors to find bit pixels.
			cVector rotCclock;
			rotCclock.x = -i.rotation.y*0.4;
			rotCclock.y = i.rotation.x*0.4;
			cVector rotClock;
			rotClock.x = i.rotation.y*0.4;
			rotClock.y = -i.rotation.x*0.4;
			cVector reverse;
			reverse.x = -i.rotation.x*0.8;
			reverse.y = -i.rotation.y*0.8;

			const Scalar cirCol(0, 255, 0);
			const int cirSize = 1;
			circle(drawImg, Point(centerX - GRIDSIZE, centerY - GRIDSIZE), sqrt(radiusDist), Scalar(255, 0, 255), 2);


			vector<cVector> searchPoints;
			cVector point(i.center.x + rotCclock.x - GRIDSIZE, i.center.y + rotCclock.y - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotClock.x - GRIDSIZE, i.center.y + rotClock.y + -GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotCclock.x + reverse.x - GRIDSIZE, i.center.y + rotCclock.y + reverse.y - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotClock.x + reverse.x - GRIDSIZE, i.center.y + rotClock.y + reverse.y - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotCclock.x * 3 - GRIDSIZE, i.center.y + rotCclock.y * 3 - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotClock.x * 3 - GRIDSIZE, i.center.y + rotClock.y * 3 - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotCclock.x * 3 + reverse.x - GRIDSIZE, i.center.y + rotCclock.y * 3 + reverse.y - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			point = cVector(i.center.x + rotClock.x * 3 + reverse.x - GRIDSIZE, i.center.y + rotClock.y * 3 + reverse.y - GRIDSIZE);
			circle(drawImg, Point(point.x, point.y), cirSize, cirCol, 1);
			searchPoints.push_back(point);

			int bitCounter = 0;
			uchar * colPtr;
			int colCounter = 0;
			int iterations = 0;
			for (auto &sp : searchPoints) {
				colCounter = 0;
				colPtr = drawImg.ptr(sp.y);
				colCounter += *(colPtr + sp.x);
				colCounter += *(colPtr + sp.x + 1);
				colCounter += *(colPtr + sp.x + 2);
				if (colCounter < 100) {
					bitCounter = pow(2, iterations);
				}
				else {
				}
				iterations++;
			}
			i.nr = bitCounter;

		}
	}

}

void normRGBthres(Mat &input, Mat &output , int threshold) {

	//create lookup table



}

int main() {

	double t = (double)getTickCount();

	Mat cameraFrame;

	VideoCapture cap(1); //capture the video from web cam

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
		
		system("CLS");
		t = (double)getTickCount();
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
		//Finding the biggest border, just showing it for now, find a way to save it somehow
		findBorder(imgOriginal);

		//The rgb normalised container material
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
				if ((p[color + 1] - green)*(p[color + 1] - green) + (p[color + 2] - red)*(p[color + 2] - red) < 1600) {
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
		
		

		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:DanielsStuf	: " << t << endl;
			t = (double)getTickCount();
		}


		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(2, 2));
		morphologyEx(thresImg, thresImg, MORPH_DILATE, element );


		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:MORPH	: " << t << endl;
			t = (double)getTickCount();
		}




		//blob detection
		vector<glyphObj> blobs;
		grassFireBlobDetection(thresImg, blobs);

		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:BlobDetect	: " << t << endl;
			t = (double)getTickCount();
		}

		//blob analysis
		blobAnalysis(blobs, imgOriginal);
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:BlobAnalysis: " << t << endl;
			t = (double)getTickCount();
		}

		cv::imshow("original", imgOriginal);
		cv::imshow("normalized", rgbNorm);
		cv::imshow("thresholded", thresImg);

		waitKey(3);
	}


	return 0;
}

void createTrackBars() {
	namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("R", "Control", &red, 255);
	cvCreateTrackbar("G", "Control", &green, 255);
	cvCreateTrackbar("B", "Control", &blue, 255);
}

void findBorder(Mat src) {
	Mat tempImg, canny_output;
	//Making a clone of the camera feed image
	tempImg = src.clone();
	vector<vector<Point>> contours;

	vector<Vec4i> hiearchy;
	//Converting to HSV
	cvtColor(tempImg, tempImg, CV_BGR2HSV);

	//Sensitivity of threshold, higher number = bigger area to take in
	int sensitivity = 20;
	//Thresholding
	inRange(tempImg, Scalar(73 - sensitivity, 18, 18), Scalar(73 + sensitivity, 255, 255), canny_output);
	//Median blur to remove some noise
	medianBlur(canny_output, canny_output, 7);
	
	//Find the contours, and save them in contours vector vector
	findContours(canny_output, contours, hiearchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	//Empty material to store the drawing of the contour
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	// iterate through each contour.
	for (int i = 0; i < contours.size(); i++)
	{
		//  Find the area of contour
		double contour_area = contourArea(contours[i], false);
		if (contour_area > largest_area) {
			//Save the new biggest contour
			largest_area = contour_area;
			//Emptying the biggest contour container, as a newer, bigger one has been found
			biggestContour.empty();
			biggestContour.insert(biggestContour.begin(), contours[i]);
			
			// Store the index of largest contour
			largest_contour_index = 0;
			// Find the bounding rectangle for biggest contour
			bounding_rect = boundingRect(biggestContour[0]);
		}
	}
	//Green colour
	Scalar color = Scalar(0, 255, 0);
	if (largest_area > 1000) {
		//Draw the found contours
		drawContours(drawing, biggestContour, 0, color, CV_FILLED, 8, hiearchy, 0, Point());
		//Draw the bounding box the found "object"
		rectangle(drawing, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
	}
	//Show the resulting biggest contour "object"
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

}