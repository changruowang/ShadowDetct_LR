// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include "ConnCompGroup.h"
//临时添加

ConnCompGroup::ConnCompGroup(const cv::Mat& fgMask) {
	if (!fgMask.empty()) {
		update(fgMask);
	}
}

ConnCompGroup::~ConnCompGroup() {
}

void ConnCompGroup::update(const cv::Mat& fgMask, bool clean, bool fill, int minPerim) {
	mask.create(fgMask.size(), CV_8U);
	mask.setTo(cv::Scalar(0));
	comps.clear();

	// clean up input mask
	cv::Mat cleanMask;
	if (clean) {
		cv::morphologyEx(fgMask, cleanMask, cv::MORPH_OPEN, cv::Mat());
		cv::morphologyEx(cleanMask, cleanMask, cv::MORPH_CLOSE, cv::Mat());
	}
	else {
		fgMask.copyTo(cleanMask);
	}
	//while循环的意义：首先外部的大循环是寻找同级的最外面的轮廓，即n个连通域所产生的轮廓，因此在此循环中计算boundrect等信息。
	//                 通过 hierarchy[i][0];  [i][0]表示轮廓i同级的下一轮廓的序号   [i][0]
	//  内部j的for小循环：[i][2]表示第i个轮廓的子轮廓。通过for循环便便利了i轮廓内部的所有第一层子轮廓。
	//  内部最后一个双层for循环，记录了联通域区域的像素位置
	//整个循环下来，记录了途中所有最外层轮廓及其对应的连通域的信息 以及 内部一级子轮廓的信息（如果参数为CV_RETR_CCOMP的话）
	// find contours
	int mode = (fill ? CV_RETR_EXTERNAL : CV_RETR_CCOMP);
	cv::findContours(cleanMask, contours, hierarchy, mode, CV_CHAIN_APPROX_NONE);
	int i = 0;
	while (i >= 0 && i < (int) contours.size()) {
		cv::Rect box = cv::boundingRect(cv::Mat(contours[i]));
		if ((2 * box.width + 2 * box.height) >= minPerim) {
			comps.push_back(ConnComp());
			ConnComp& cc = comps.back();

			cc.contours.push_back(contours[i]);
			for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0]) {
				cc.contours.push_back(contours[j]);
			}

			cc.box = box;
			cc.boxMask.create(box.size(), CV_8U);
			cc.boxMask.setTo(cv::Scalar(0));
			cv::drawContours(cc.boxMask, cc.contours, -1, 255, CV_FILLED, 8,
					std::vector<cv::Vec4i>(), std::numeric_limits<int>::max(),
					cv::Point(-box.x, -box.y));

			// calculate center of mass
			cv::Moments moments = cv::moments(cc.boxMask, true);
			cc.center.x = (int) (moments.m10 / moments.m00) + box.x;
			cc.center.y = (int) (moments.m01 / moments.m00) + box.y;

			for (int y = 0; y < cc.boxMask.rows; ++y) {
				uchar* ccMaskPtr = cc.boxMask.ptr(y);

				for (int x = 0; x < cc.boxMask.cols; ++x) {
					if (ccMaskPtr[x] > 0) {
						cc.pixels.push_back(cv::Point(x + box.x, y + box.y));
					}
				}
			}
			//绘制填充的轮廓 
			cc.draw(mask);
		}

		i = hierarchy[i][0];
	}
}

void ConnCompGroup::draw(cv::Mat& dst, const cv::Scalar& color, bool filled) const {
	cv::drawContours(dst, contours, -1, color, (filled ? CV_FILLED : 1));
}
