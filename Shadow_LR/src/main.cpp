// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include"iostream"
#include "ChromacityShadRem.h"
#include "GeometryShadRem.h"
#include "LrTextureShadRem.h"
#include "PhysicalShadRem.h"
#include "SrTextureShadRem.h"

using namespace std;
using namespace cv;

#define BKG_LEARNING_HISTORY_LENGTH  100
#define BKG_LEARNING_THRESHOLD       25
#define BKG_LEARNING_RATE            0.00

int FRAME_RESIZE_WIDTH = 400;
int FRAME_RESIZE_HEIGHT = 400;

int main(int argc, char **argv) {
	if (argc != 3)
		return -1;
	 //load frame, background and foreground
	string base_path = "D:\\MyProject\\WANG_Shadow\\shadowvideo\\";
	string path = base_path + argv[1];
	
	VideoWriter writer;
	writer.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(FRAME_RESIZE_WIDTH, FRAME_RESIZE_HEIGHT), true);
	VideoCapture cap(path);
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}
	Mat bg, fg, frame, lrTexMask;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2(BKG_LEARNING_HISTORY_LENGTH, BKG_LEARNING_THRESHOLD, false);
	LrTextureShadRem lrTex;

//预更新一帧背景，免得第一帧获得的前景错误
	string avg2 = argv[2];
	if(avg2 == "NULL")
		cap >> frame;
	else
		frame = cv::imread(base_path + avg2);

	pBackSub->apply(frame, fg, BKG_LEARNING_RATE);
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	while (1)
	{
		cap >> frame;
		if (frame.empty())
			break;
		//remove noise
		pBackSub->apply(frame, fg, BKG_LEARNING_RATE);
		pBackSub->getBackgroundImage(bg);	
		
		morphologyEx(fg, fg, MORPH_OPEN, element);
		morphologyEx(fg, fg, MORPH_CLOSE, element);

		lrTex.removeShadows(frame, fg, bg, lrTexMask);
		imshow("lrTexMask", lrTexMask);
		Mat opFrame = frame.clone();
		Scalar blue(255, 0, 0);
		Scalar red(0, 0, 255);
		opFrame.setTo(blue, fg);
		opFrame.setTo(red, lrTexMask);
		resize(opFrame, opFrame, Size(FRAME_RESIZE_WIDTH, FRAME_RESIZE_HEIGHT));
		writer.write(opFrame);
		imshow("Shadow Detection", opFrame);
		//imshow("frame", frame);
		//imshow("bg", bg);
		cv::waitKey(10);
	}

	

	return 0;
}
