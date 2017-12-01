#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const int GRIDSIZE = 1;

int r = 180;
int g = 40;

bool timeKeeping = false;
const float discrimHW = 0.2;
const int rgConvThreshold = 150;

void createTrackBars();


void findBorder(int, void*, Mat src);

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

void dropFire(uchar * pixel, glyphObj &store, int &width, int y, int x, cVector &from) {
	*pixel = store.nr;
	from.x = x;
	from.y = y;
	store.list.push_back(from);

	if (*(pixel + GRIDSIZE) == 255) {
		dropFire(pixel + GRIDSIZE, store, width, y, x + GRIDSIZE, from);
	}
	if (*(pixel + width) == 255) {
		dropFire(pixel + width, store, width, y + GRIDSIZE, x, from);
	}

	if (*(pixel - GRIDSIZE) == 255) {
		dropFire(pixel - GRIDSIZE, store, width, y, x - GRIDSIZE, from);
	}

	if (*(pixel - width) == 255) {
		dropFire(pixel - width, store, width, y - GRIDSIZE, x, from);
	}
}

void dropFireNew(uchar * pixel, glyphObj &store, int &width, int y, int x, cVector &from) {
	*pixel = store.nr;
	from.x = x;
	from.y = y;
	store.list.push_back(from);

	if (*(pixel + GRIDSIZE) == 255) {
		dropFireNew(pixel + GRIDSIZE, store, width, y, x + GRIDSIZE, from);
	}
	if (*(pixel + width) == 255) {
		dropFireNew(pixel + width, store, width, y + GRIDSIZE, x, from);
	}

	if (*(pixel - GRIDSIZE) == 255) {
		dropFireNew(pixel - GRIDSIZE, store, width, y, x - GRIDSIZE, from);
	}

	if (*(pixel - width) == 255) {
		dropFireNew(pixel - width, store, width, y - GRIDSIZE, x, from);
	}
}

void grassFireBlobDetection(Mat &biImg, vector<glyphObj> &blobs) {
	int nRows = biImg.rows;
	int nCols = biImg.cols;
	int rowSize = nCols * GRIDSIZE;
	uchar * p;
	uchar * passer;
	glyphObj currentBlob;
	cVector assigner;
	int col = 245;
	for (int i = GRIDSIZE + 1; i < nRows - GRIDSIZE - 1; i += GRIDSIZE) {
		p = biImg.ptr<uchar>(i);
		for (int j = GRIDSIZE+1; j < nCols - GRIDSIZE-1; j += GRIDSIZE) {
			if (p[j] == 255) {
				blobs.push_back(currentBlob);
				blobs.back().nr = col;
				passer = &p[j];
				dropFire(passer, blobs.back(), rowSize, i, j, assigner);
				col -= 10;
				if (col < 20) {
					col = 245;
				}
			}
		}
	}
}



void grassFireBlobDetectionNew(Mat &biImg, vector<glyphObj> &blobs) {
	int nRows = biImg.rows;
	int nCols = biImg.cols;
	int rowSize = nCols * GRIDSIZE;
	uchar * p;
	uchar * passer;
	glyphObj currentBlob;
	cVector assigner;
	int col = 245;
	for (int i = GRIDSIZE + 1; i < nRows - GRIDSIZE - 1; i += GRIDSIZE) {
		p = biImg.ptr<uchar>(i);
		for (int j = GRIDSIZE+1; j < nCols - GRIDSIZE-1; j += GRIDSIZE) {
			if (p[j] == 255) {
				blobs.push_back(currentBlob);
				blobs.back().nr = col;
				passer = &p[j];
				dropFireNew(passer, blobs.back(), rowSize, i, j, assigner);
				col -= 10;
				if (col < 20) {
					col = 245;
				}
			}
		}
	}
}

void preLookUpBgr2rg(Mat &in, Mat &out, int (&divLUT)[768][256]) {
	//convert to normalized rgb space

	
	//then convert using LUT
	int nRows = in.rows;
	int nCols = in.cols * 3;
	int sum = 0;
	uchar * p;
	uchar * cp;

	uchar red;
	uchar green;
	uchar blue;
	int * lutptr;
	
	for (int i = 0; i < nRows; i += GRIDSIZE) {
		p = in.ptr<uchar>(i);
		cp = out.ptr<uchar>(i);

		for (int j = 0; j < nCols; j += 3 * GRIDSIZE) {
			blue = p[j];
			green = p[j+1];
			red = p[j+2];
			sum = blue + green + red;
			lutptr = divLUT[sum];
			cp[j] = *(lutptr + blue);
			cp[j + 1] = *(lutptr + green);
			cp[j + 2] = *(lutptr + red);
		}
	}
}


void lookUpBgr2rg(Mat &in, Mat &out) {
	//convert to normalized rgb space

	//start by creating lookup table
	int divLUT[768][256]; //division lookuptavle;
	for (int i = rgConvThreshold; i < 768; i++) {
		for (int j = 0; j < 256; j++) {
			divLUT[i][j] = (j * 255)/i;
		}
	}
	//then convert using LUT
	int nRows = in.rows;
	int nCols = in.cols * 3;
	int sum = 0;
	uchar * p;
	uchar * cp;

	for (int i = 0; i < nRows; i += GRIDSIZE) {
		p = in.ptr<uchar>(i);
		cp = out.ptr<uchar>(i);
		for (int j = 0; j < nCols; j += 3*GRIDSIZE) {
			sum = p[j] + p[j + 1] + p[j + 2];
			if (sum < rgConvThreshold) {
				cp[j] = 0;
				cp[j + 1] = 0;
				cp[j + 2] = 0;
				continue;
			}
			cp[j] = divLUT[sum][p[j]];
			cp[j+1] = divLUT[sum][p[j+1]];
			cp[j+2] = divLUT[sum][p[j+2]];
		}
	}
}

void Bgr2rg(Mat &in, Mat &out) {
	//convert to normalized rgb space

	//start by creating lookup table
	//then convert using LUT
	int nRows = in.rows;
	int nCols = in.cols * 3;
	float sum = 0;
	uchar * p;
	uchar * cp;

	for (int i = 0; i < nRows; i += GRIDSIZE) {
		p = in.ptr<uchar>(i);
		cp = out.ptr<uchar>(i);
		for (int j = 0; j < nCols; j += 3 * GRIDSIZE) {
			sum = p[j] + p[j + 1] + p[j + 2];
			if (sum < rgConvThreshold) {
				cp[j] = 0;
				cp[j + 1] = 0;
				cp[j + 2] = 0;
				continue;
			}
			cp[j] = 255 * p[j]/sum;
			cp[j + 1] = 255 * p[j+1] / sum;
			cp[j + 2] = 255 * p[j+2] / sum;
		}
	}
}

void blobAnalysis(vector<glyphObj> &blobs, Mat &drawImg) {

	//printing out objects
	int minSize = 200 / GRIDSIZE;
	int maxSize = 8000 / GRIDSIZE;
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
		int searchDist = 0;
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
		centerX = (smallestX + largestX) / 2;
		centerY = (smallestY + largestY)/ 2;
		radiusDist = ((float)((float)(largestX - centerX) + (centerX - smallestX) + (largestY - centerY) + (centerY - smallestY))) / 4;
		i.center.x = centerX;
		i.center.y = centerY;
		circle(drawImg, Point(centerX - GRIDSIZE, centerY - GRIDSIZE), 2, Scalar(0, 0, 255), 5);
		searchDist = (float)radiusDist * 0.7;
		searchDist = searchDist * searchDist;
		//find closest pixel
		int dist = 10000;
		vector<cVector> points;
		for (auto &v : i.list) {
			dist = (v.x - i.center.x) * (v.x - i.center.x) + (v.y - i.center.y) * (v.y - i.center.y);
			if (dist < searchDist) {
				points.push_back(v);
			}
		}
		if (points.size() == 0) { continue; }
		float rotX = 0;
		float rotY = 0;

		for (auto &p : points) {
			rotX+= p.x - centerX;
			rotY += p.y - centerY;
		}
	
		rotX /= points.size();
		rotY /= points.size();
	
		//set vector size to be radius
		float vecDist = sqrt(rotX * rotX + rotY * rotY);
		if (vecDist == 0) {
			continue;
		}
		vecDist = radiusDist/ vecDist;

		i.rotation.x = rotX * vecDist;
		i.rotation.y = rotY * vecDist;

		line(drawImg, Point(i.center.x - GRIDSIZE, i.center.y - GRIDSIZE), Point(i.center.x + i.rotation.x - GRIDSIZE, i.center.y + i.rotation.y - GRIDSIZE), Scalar(0, 255, 0), 2);

		//use vectors to find bit pixels.

		i.center.x = centerX + rotX*0.4;
		i.center.y = centerY + rotY*0.4;


		cVector rotCclock;
		rotCclock.x = -rotY*0.38;
		rotCclock.y = rotX*0.38;
		cVector rotClock;
		rotClock.x = rotY*0.38;
		rotClock.y = -rotX*0.38;
		cVector reverse;
		reverse.x = -rotX*0.9;
		reverse.y = -rotY*0.9;

		
		const int cirSize = 1;

		vector<cVector> searchPoints;
		cVector point(i.center.x + rotCclock.x -GRIDSIZE, i.center.y + rotCclock.y - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotClock.x - GRIDSIZE, i.center.y + rotClock.y - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotCclock.x + reverse.x - GRIDSIZE, i.center.y + rotCclock.y + reverse.y - GRIDSIZE);
		
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotClock.x + reverse.x - GRIDSIZE, i.center.y + rotClock.y + reverse.y - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotCclock.x * 3 - GRIDSIZE, i.center.y + rotCclock.y * 3 - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotClock.x * 3 - GRIDSIZE, i.center.y + rotClock.y * 3 - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotCclock.x * 3 + reverse.x - GRIDSIZE, i.center.y + rotCclock.y * 3 + reverse.y - GRIDSIZE);
		searchPoints.push_back(point);

		point = cVector(i.center.x + rotClock.x * 3 + reverse.x - GRIDSIZE, i.center.y + rotClock.y * 3 + reverse.y - GRIDSIZE);
		searchPoints.push_back(point);
		
		int bitCounter = 0;
		uchar * colPtr;
		int iterations = 0;
		for (auto &sp : searchPoints) {
			
			Vec3b intensity = drawImg.at<Vec3b>(sp.y, sp.x);

			if (intensity[0]< 125 && intensity[1] < 125 && intensity[2] < 125) {
				bitCounter += pow(2, iterations);
				circle(drawImg, Point(sp.x, sp.y), cirSize, Scalar(0,255,0), 1);
			}
			else {
				circle(drawImg, Point(sp.x, sp.y), cirSize, Scalar(0,0,255), 1);
			}
			iterations++;
		}
		circle(drawImg, Point(centerX - GRIDSIZE, centerY - GRIDSIZE), sqrt(searchDist), Scalar(255, 0, 255), 2);

	
		i.nr = bitCounter;
		putText(drawImg, to_string(bitCounter), Point(centerX, centerY - sqrt(radiusDist) - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0));
		//cout << "found an object with value " << i.nr << endl;

	}

}

void createTrackBars();
void findBorder(Mat src);

//Finding border edge
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
int largest_area = 0;
int largest_contour_index = 0;
Rect bounding_rect;
vector<vector<Point>> biggestContour;
vector<Vec4i> permHiearchy;


void thresholdSpeedy(Mat &in, Mat &out ) {

	uchar * p;
	uchar * cp;
	int * ip; 
	int nRows = in.rows;
	int nCols = in.cols;

	int lookup[255][255];
	int minPossibleValue = 0;


	for (int i = minPossibleValue; i < 255; i++) {
		ip = lookup[i];
		for (int j = minPossibleValue; j < 255; j++) {
			if (((i - g)*(i - g) + (j - r)*(j - r)) < 3600) {
				*(ip + j) = 255;
			}
			else {
				*(ip + j) = 0;
			}
		}
	}
	int color = 0;
	for (int i = 0; i < nRows; i+=GRIDSIZE) {
		p = in.ptr<uchar>(i);
		cp = out.ptr<uchar>(i);
		for (int j = 0; j < nCols; j += GRIDSIZE) {
			color = j * 3;
			cp[j] = lookup[p[color + 1]][p[color + 2]];
		}
	}
}

int main() {

	double t = (double)getTickCount();
	double compa;
	double tots = 0;
	long counter = 0;
	double startTime;


	Mat cameraFrame;
	//start by creating lookup table
	int divLUT[768][256]; //division lookuptavle;
	for (int i = rgConvThreshold; i < 768; i++) {
		for (int j = 0; j < 256; j++) {
			if (i < rgConvThreshold) { 
				divLUT[i][j] = 0;
			}
			else {
				divLUT[i][j] = (j * 255) / i;
			}
		}
	}

	VideoCapture cap(0); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}
	
	createTrackBars();
	Mat imgOriginal;
	bool bSuccess = cap.read(imgOriginal);
	imgOriginal = imread("city.jpg", CV_LOAD_IMAGE_COLOR);
	Mat rgbNorm;
	rgbNorm.create(imgOriginal.rows, imgOriginal.cols, CV_8UC3);

	cap.set( CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set( CV_CAP_PROP_FRAME_HEIGHT, 480);
	startTime = ((double)getTickCount()) / getTickFrequency();
	while (true)
	{
	
		
		//system("CLS");
		//cout << "TIMEKEEPING:dif	: " << tots << endl;
		
	//	bool bSuccess = cap.read(imgOriginal); // read a new frame from video
		
		//
		/*if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}*/
		

		
		/*

		Mat thresImg(imgOriginal.rows, imgOriginal.cols, CV_8UC1);

		vector<glyphObj> blobs2;
		vector<glyphObj> blobs;
	
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:instanceMat	: " << t << endl;
			t = (double)getTickCount();
		}
		*/
		preLookUpBgr2rg(imgOriginal, rgbNorm, divLUT);
	
		/*
		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "TIMEKEEPING:RG lookup	: " << t << endl;
			tots += t;
			t = (double)getTickCount();
		}

		
		thresholdSpeedy(rgbNorm, thresImg);

		if (timeKeeping) {
			
			t = ((double)getTickCount() - t) / getTickFrequency();
			compa = t;
			cout << "TIMEKEEPING:thresspeedy	: " << t << endl;
			t = (double)getTickCount();
		}



		copyMakeBorder(thresImg, thresImg, GRIDSIZE+1, GRIDSIZE+1, GRIDSIZE+1, GRIDSIZE+1, BORDER_CONSTANT, 0);
		//Test border shit here
		//findBorder(0, 0, thresImg);

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


grassFireBlobDetectionNew(thresImg2, blobs2);

		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			compa = t;
			cout << "TIMEKEEPING:Blobnew	: " << t << endl;

			t = (double)getTickCount();
		}
		//blob detection
	
		grassFireBlobDetection(thresImg, blobs);


		if (timeKeeping) {
			t = ((double)getTickCount() - t) / getTickFrequency();
		
			tots += (t - compa) * 1000;
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
		*/
		
		//cv::imshow("RG NORM ", rgbNorm);
		//cv::imshow("grassfire", thresImg);

		
		
		counter++;
		if (counter % 10000 == 0) {
			t = ((double)getTickCount()) / getTickFrequency();
			cout << ((t - startTime) / counter) << endl;
			counter = 0;
			cv::imshow("original", rgbNorm);
			waitKey(1);
			startTime = ((double)getTickCount()) / getTickFrequency();
		
		}
	}

	return 0;
}

void createTrackBars() {
	namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("red color", "Control", &r, 255);
	cvCreateTrackbar("green color", "Control", &g, 255);
}

void findBorder(int, void*, Mat src) {
	Mat canny_output;

	vector<vector<Point>> contours;

	vector<Vec4i> hiearchy;

	Canny(src, canny_output, thresh, thresh*2, 3);

	findContours(canny_output, contours, hiearchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(0, 0, 255);
		drawContours(drawing, contours, i, color, 2, 8, hiearchy, 0, Point());
	}

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}

/*
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
	findContours(canny_output, contours, hiearchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	
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

}*/