/*
 * headpose.cpp
 *
 *  Created on: Jul 29, 2014
 *      Author: tesfu
 */

#include <string>
#include <map>
#include <iostream>
#include <stdio.h>
#include<cstdlib>
#include<array>
#include<vector>
#include<map>
#include<vector>
#include <memory>

#define _USE_MATH_DEFINES // for C++/
//#include <math.h>
//#include<cmath>
//#include <tbb/tbb.h>

#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


#include "symmetryDetector.h"

using namespace std;
using namespace cv;

//int thresh = 100;Mat src, dst;
Mat src_gray,gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
Mat frame, dst;
	Mat GetImg;
	Mat prvs;
	Mat flow;
	Mat cflow;
	Mat frameN;

String face_cascade_name = "cascadeface.xml";
String window_image= "IMAGE_WINDOW";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "FACE_DETECTION";
String filename = "/test.png";


static Point accumIndex(-1, -1);
static void onMouse(int event, int x, int y, int, void * data);
string windowt = "edge";
void testImage();
void testVideo();
void test(Mat framet );
void detectAndDisplay(Mat framei);
void dense_optical_flow(const Mat& flow, Mat& cflowmap, int step, const Scalar& color);
void testImage_detection(Mat framef);



//=================================main function ======================================================

int main(int argc,  char ** argv)
{
	// Load the cascades

	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };

	   testVideo();

   return 0;
     }

//===================uses the default pc camera===========================================================
void testVideo()

{
	namedWindow("");
	cvMoveWindow("", 0, 0);
	Mat frame1;

	VideoCapture cap(0);

     int s =2;

	if ((cap.read(frame)))

		//	return 0;

		resize(frame, prvs, Size(frame.size().width / s, frame.size().height / s));
		cvtColor(prvs, gray, CV_RGB2GRAY);


	while (frame1.empty())
		cap >> frame1;

	/* Resize the image accordingly */
	resize(frame1, frame1, Size(), 0.7, 0.7);

	/* Determine the shape of Hough accumulation matrix */
	float rho_divs = hypot(frame1.rows, frame1.cols) + 1;
	float theta_divs = 180.0;

// ============create the object detector to instatiate symmetrydetector class==================

	SymmetryDetector detector(frame1.size(), Size(rho_divs, theta_divs), 1);

	/* Adjustable parameters, depending on the scene condition */
	int canny_thresh_1 = 30;
	int canny_thresh_2 = 90;
	int min_pair_dist = 25;
	int max_pair_dist = 500;
	int no_of_peaks = 1;

	createTrackbar("canny_thresh_1", "", &canny_thresh_1, 500);
	createTrackbar("canny_thresh_2", "", &canny_thresh_2, 500);
	createTrackbar("min_pair_dist", "", &min_pair_dist, 500);
	createTrackbar("max_pair_dist", "", &max_pair_dist, 500);
	createTrackbar("no_of_peaks", "", &no_of_peaks, 10);

	Mat edge;
	while (true){
		cap >> frame1;
		test(frame1);
		flip(frame1, frame1, 1);
		resize(frame1, frame1, Size(), 0.7, 0.7);

		/* Find the edges of the image */
		cvtColor(frame1, edge, CV_BGR2GRAY);
		Canny(edge, edge, canny_thresh_1, canny_thresh_2);


		/* Vote for the hough matrix */

		detector.vote(edge, min_pair_dist, max_pair_dist);
		Mat accum = detector.getAccumulationMatrix();


		/* Get the result and draw the symmetrical line */
		vector<pair<Point, Point>> result = detector.getResult(no_of_peaks);

		for (auto point_pair : result)

			line(frame1, point_pair.first, point_pair.second, Scalar(0, 0, 255), 3);


		/* Convert our Hough accum matrix to heat map */

		double minVal, maxVal;

		minMaxLoc(accum, &minVal, &maxVal);

		accum.convertTo(accum, CV_8UC3);// 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal)


		applyColorMap(accum, accum, COLORMAP_JET);
		resize(accum, accum, Size(), 2.0, 0.5);

		//* Show the original, edge and the accumulation image
		Mat appended = Mat::zeros(frame1.rows + accum.rows, frame1.cols * 2, CV_8UC3);
		frame1.copyTo(Mat(appended, Rect(0, 0, frame1.cols, frame1.rows)));
		cvtColor(edge, Mat(appended, Rect(frame1.cols, 0, edge.cols, edge.rows)), CV_GRAY2BGR);
		accum.copyTo(Mat(appended, Rect(0, frame1.rows, accum.cols, accum.rows)));


		imshow("", appended);
		imwrite("appended11.jpg", appended);
		if (waitKey(10) == 'q')
			break;
	}
}

/*
* =============test on the image stored on the disc===========================================
*/
void testImage() {
	namedWindow("");
	cvMoveWindow("", 0, 0);

	Mat frameI= imread("/home/tesfu/workspace/whohead/test.jpg");

	/* Determine the shape of Hough accumulationmatrix */
	float rho_divs = hypot(frameI.rows, frameI.cols) + 1;
	float theta_divs = 180.0;

	SymmetryDetector detector(frameI.size(), Size(rho_divs, theta_divs), 1);


	Rect region(0, frameI.rows, theta_divs * 2.0, rho_divs * 0.5);
	setMouseCallback("", onMouse, static_cast<void*>(&region));
	Mat temp, edgeI;

	/* Adjustable parameters, depending on the scene condition */
	int canny_thresh_1 = 30;
	int canny_thresh_2 = 90;
	int min_pair_dist = 25;
	int max_pair_dist = 500;
	int no_of_peaks = 1;

	createTrackbar("canny_thresh_1", "", &canny_thresh_1, 500);
	createTrackbar("canny_thresh_2", "", &canny_thresh_2, 500);
	createTrackbar("min_pair_dist", "", &min_pair_dist, 500);
	createTrackbar("max_pair_dist", "", &max_pair_dist, 500);
	createTrackbar("no_of_peaks", "", &no_of_peaks, 10);

	while (true) {
		temp = frameI.clone();

		/* Find the edges */
		cvtColor(temp, edgeI, CV_BGR2GRAY);
		//	Mat edgeIB = edgeI.clone();
		Canny(edgeI, edgeI, canny_thresh_1, canny_thresh_2);

		/* Vote for the accumulation matrix */

		detector.vote(edgeI, min_pair_dist, max_pair_dist);

		/* Draw the symmetrical line */


		vector<pair<Point, Point>> result = detector.getResult(no_of_peaks);

		for (auto point_pair : result)

			line(temp, point_pair.first, point_pair.second, Scalar(0, 0, 255), 2);


		/* Visualize the Hough accum matrix */
		Mat accum = detector.getAccumulationMatrix();

		//double minVal;
		//double maxVal;

		//minMaxLoc(accum, &minVal, &maxVal, NULL, NULL, NULL);
		accum.convertTo(accum, CV_8UC3);//out,type, 1,0
		applyColorMap(accum, accum, COLORMAP_JET);
		resize(accum, accum, Size(), 2.0, 0.5);

		/* Draw lines based on cursor position */

		if (accumIndex.x != -1 && accumIndex.y != -1) {
			pair<Point, Point> point_pair = detector.getLine(accumIndex.y, accumIndex.x);
			line(temp, point_pair.first, point_pair.second, CV_RGB(0, 255, 0), 2);
		}

		/* Show the original and edge images */
		Mat appended = Mat::zeros(temp.rows + accum.rows, temp.cols * 2, CV_8UC3);
		temp.copyTo( Mat(appended, Rect(0, 0, temp.cols, temp.rows)));
		cvtColor(edgeI, Mat(appended, Rect(temp.cols, 0, edgeI.cols, edgeI.rows)), CV_GRAY2BGR);
		accum.copyTo(Mat(appended, Rect(0, temp.rows, accum.cols, accum.rows)));

		imshow("", appended);
		  imwrite("append11.jpg", appended);
		if (waitKey(10) == 'q')
			break;
	}
}

/**
* Mouse callback, to show the line based on which part of accumulation matrix the cursor is.
*/
static void onMouse(int event, int x, int y, int, void * data) {
	Rect *region = (Rect*)data;
	Point point(x, y);

	if ((*region).contains(point)) {
		accumIndex.x = (point.x - region->x) / 2.0;
		accumIndex.y = (point.y - region->y) * 2.0;
	}
	else {
		accumIndex.x = -1;
		accumIndex.y = -1;
	}
}


void test(Mat framed)
{
	Mat imagp = imread(filename,1);

	Mat next;
  int s =2;
	//-- 3. Apply the classifier to the frame
		if (!framed.empty())
			//break;
		detectAndDisplay(frame);

		dst.create(framed.size(), frame.type());

		//=====================================================================================================

		//   Resize  optical flow estimation

		resize(framed, next, Size(framed.size().width / s, framed.size().height / s));
		cvtColor(next, next, CV_BGR2GRAY);

		calcOpticalFlowFarneback(gray, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		cvtColor(gray, cflow, CV_GRAY2BGR);

		dense_optical_flow(flow, cflow, 10, CV_RGB(0, 255, 0));

		gray = next.clone();


      		 imshow("OpticalFlowFarneback", cflow);
				imshow("prvs11", prvs);
				imshow("next111", next);
				imwrite("first11.jpg",prvs );
				imwrite("nextim11.jpg",next);
				imwrite("opticalgradient11.jpg",cflow);
				/*cvWaitKey(100000);
			  int c = waitKey(100);
			if ((char)c == 'c') { break; }  */
	       }


//=================================##############################################======================
//face detection algorithm

//======================================================================================================
void detectAndDisplay(Mat img_face)
{
std::vector<Rect> faces;
Mat frame_gray;

cvtColor(img_face, frame_gray, CV_BGR2GRAY);
equalizeHist(frame_gray, frame_gray);


face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

for (size_t i = 0; i < faces.size(); i++)
{

Point p1(faces[i].x, faces[i].y);
Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
rectangle(img_face, p1, p2, Scalar(255, 255, 0), 1, 8, 0);
Mat faceROI = frame_gray(faces[i]);
std::vector<Rect> eyes;

//-- In each face, detect eyes

eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

for (size_t j = 0; j < eyes.size(); j++)
{
Point p1(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
Point p2(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);
rectangle(img_face, p1, p2, Scalar(255, 255, 0), 1, 8, 0);

   }

   }

imshow("FACEDETECTOR", img_face);
imwrite("facedet1.jpg",img_face);


}

void testImage_detection(Mat img)

{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(img, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{

	Point p1(faces[i].x, faces[i].y);
	Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
	rectangle(img, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
	//Mat faceROI = frame_gray(faces[i]);


		imshow("FACEDETECTOR", img);
	}



}

void  dense_optical_flow(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
for (int y = 0; y < cflowmap.rows; y += step)
for (int x = 0; x < cflowmap.cols; x += step)
{
const Point2f& fxy = flow.at< Point2f>(y, x);
line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
color);
circle(cflowmap, Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1, color, -1);
}
}



