/*
 * SymmetryDetector.h
 *
 *  Created on: Jul 29, 2014
 *      Author: tesfu
 */

#ifndef SYMMETRYDETECTOR_H_
#define SYMMETRYDETECTOR_H_

#include <iostream>
#include <opencv2/opencv.hpp>

//#include <tbb/tbb.h>
//#include <string>
#include <stdlib.h>
#include<limits>
#include<map>
#include<set>
#define _USE_MATH_DEFINES // for C++/
#include <math.h>
#include<cmath>
#include<array>
#include<memory>
#include<vector>
#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/core/core.hpp"
#include"opencv2/features2d/features2d.hpp"


using namespace std;
using namespace cv;
//using namespace tbb;


class
SymmetryDetector {
public:
	SymmetryDetector(const Size image_size, const Size hough_size, const int rot_resolution = 1);

	void vote(Mat& image, int min_pair_dist, int max_pair_dist);

	inline void rotateEdges(vector<Point2f>& edges, int theta);

	Mat getAccumulationMatrix(float thresh = 0.0);

	vector<pair<Point, Point>> getResult(int no_of_peaks, float threshold = -1.0f);
	pair<Point, Point> getLine(float rho, float theta);

private:
	vector<Mat> rotMatrices;
	Mat rotEdges;
	vector<float*> reRows;
	Mat accum;

	Size imageSize;
	Point2f center;
	float diagonal;
	int rhoDivision;
	int rhoMax;
	int thetaMax;
};


#endif /* SYMMETRYDETECTOR_H_ */
