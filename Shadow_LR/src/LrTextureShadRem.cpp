// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "LrTextureShadRem.h"

using namespace cv;

#define MXSIZE 10000
#define SHOW_CANDIDATE_SHADOWS  0
#define SHOW_CANNY_DIFF         0
#define SHOW_GRAD_REGION        0
#define PRINT_REGION_CORR       0  //如果输出太多的0 则梯度大小阈值选大了

const std::vector<cv::Mat> LrTextureShadRem::skeletonKernels = getSkeletonKernels();

LrTextureShadRem::LrTextureShadRem(const LrTextureShadRemParams& params) {
	this->params = params;

	frameCount = 0;
	avgAtten = 0;
	avgSat = 0;
	avgPerim = 0;
}

LrTextureShadRem::~LrTextureShadRem() {
}

void LrTextureShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	ConnCompGroup fg(fgMask);  //计算前景连通域
	fg.mask.copyTo(srMask);      
	printf("\n\n");
	cv::Mat grayFrame, grayBg, hsvFrame, hsvBg;
	cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
	cv::cvtColor(bg, grayBg, CV_BGR2GRAY);
	cv::cvtColor(frame, hsvFrame, CV_BGR2HSV);
	cv::cvtColor(bg, hsvBg, CV_BGR2HSV);
    //avgAtten视屏大区域阴影的平均亮度V
	//avgSat 视频平均每帧 S^2/(S^2 + V^2)
	//统计视频每帧轮廓的平均长度
	// calculate global frame properties                               fg.mask前景掩膜  
	avgAtten = ((avgAtten * frameCount) + frameAvgAttenuation(hsvFrame, hsvBg, fg.mask)) / (frameCount + 1);   //计算前n帧大区域阴影部分的平均亮度（V）    
	avgSat = ((avgSat * frameCount) + frameAvgSaturation(hsvFrame, fg.mask)) / (frameCount + 1);
	avgPerim = ((avgPerim * frameCount) + fgAvgPerim(fg)) / (frameCount + 1);
	++frameCount;

	//粗略得到候选阴影区域 find candidate shadow pixels
	getCandidateShadows(hsvFrame, hsvBg, fg.mask, candidateShadows);
#if SHOW_CANDIDATE_SHADOWS
	imshow("candidateShadows", candidateShadows);
#endif
	// split using foreground edges cannyDiffWithBorders
	getEdgeDiff(grayFrame, grayBg, fg, candidateShadows, cannyFrame, cannyBg, cannyDiffWithBorders, borders, cannyDiff);
#if SHOW_CANNY_DIFF
	cv::imshow("cannyDiff", cannyDiff);
#endif
	cv::Mat splitCandidateShadowsMask;
	maskDiff(candidateShadows, cannyDiff, splitCandidateShadowsMask, params.splitRadius);
	
	// connected components are candidate shadow regions 更新候选阴影的连通域
	splitCandidateShadows.update(splitCandidateShadowsMask, false, true);
#if SHOW_GRAD_REGION
	cv::imshow("splitCandidateShadows", splitCandidateShadows.mask);
#endif
	shadows.create(grayFrame.size(), CV_8U);
	shadows.setTo(cv::Scalar(0));

	// classify regions with high correlation as shadows
	gradCorrThresh = (
		avgAtten < params.avgAttenThresh ? params.gradCorrThreshLowAtten : params.gradCorrThreshHighAtten);
	for (int sr = 0; sr < (int) splitCandidateShadows.comps.size(); ++sr) {
		ConnComp& cc = splitCandidateShadows.comps[sr];      //遍历每个连通域信息

		float regionCorr = getGradDirCorr(grayFrame, cc, grayBg);
#if PRINT_REGION_CORR
		printf("frame %d regionCorr:%.3f pixels:%d posy:%d\n", frameCount, regionCorr , cc.pixels.size(),cc.center.y);
#endif
		if (regionCorr > gradCorrThresh) {
			cc.draw(shadows, 255);
		}
	}

	bool cleanShadows = (avgAtten < params.avgAttenThresh ? params.cleanShadows : false);
	if (cleanShadows || params.fillShadows || params.minShadowPerim > 0) {
		if (cleanShadows) {
			cv::morphologyEx(shadows, shadows, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
		}   //cv::MORPH_CLOSE 闭运算 先膨胀再腐蚀 可以清楚图像中的小黑点
		postShadows.update(shadows, false, params.fillShadows, params.minShadowPerim);
		postShadows.mask.copyTo(shadows);

	}
	srMask.setTo(0, shadows);

	if (params.cleanSrMask || params.fillSrMask) {
		postSrMask.update(srMask, params.cleanSrMask, params.fillSrMask);
		postSrMask.mask.copyTo(srMask);
	}
}


float LrTextureShadRem::frameAvgAttenuation(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg) {
	float avgAtten = 0;
	int count = 0;
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* fgPtr = fg.ptr(y);
		const uchar* framePtr = hsvFrame.ptr(y);
		const uchar* bgPtr = hsvBg.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				float atten = (float) (10 + bgPtr[x * 3 + 2]) / (10 + framePtr[x * 3 + 2]);
				bool vIsShadow = (atten > 1 && atten < 5);

				int hDiff = CV_IABS(framePtr[x * 3] - bgPtr[x * 3]);
				if (hDiff > 90) {
					hDiff = 180 - hDiff;
				}
				bool hIsShadow = (hDiff < 4);

				if (vIsShadow && hIsShadow) {
					avgAtten += atten;
					++count;
				}
			}
		}
	}

	if (count > 0) {
		avgAtten /= count;
	}

	return avgAtten;
}

float LrTextureShadRem::frameAvgSaturation(const cv::Mat& hsvFrame, const cv::Mat& fg) {
	float avgSat = 0;
	int count = 0;
	int vSum = 0;
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* fgPtr = fg.ptr(y);
		const uchar* framePtr = hsvFrame.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				avgSat += framePtr[x * 3 + 1] * framePtr[x * 3 + 2];     //s^2 + v^2
				vSum += framePtr[x * 3 + 2];                             //v^2
				++count;
			}
		}
	}

	if (count > 0) {
		avgSat /= vSum;
	}

	return avgSat;
}

float LrTextureShadRem::fgAvgPerim(const ConnCompGroup& fg) {
	float avgPerim = 0;

	for (int i = 0; i < (int) fg.comps.size(); ++i) {
		avgPerim += fg.comps[i].contours[0].size();
	}

	if (fg.comps.size() > 0) {
		avgPerim /= fg.comps.size();
	}

	return avgPerim;
}
//m2Radius 1or0   m1:前景的canny结果边缘  m2背景且不属于前景的边缘
//作用：判断属于m1而不属于m2的像素掩膜结果存于diff中
void LrTextureShadRem::maskDiff(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& diff, int m2Radius) {
	diff.create(m1.size(), CV_8U);

	for (int y = 0; y < m1.rows; ++y) {
		const uchar* m1Ptr = m1.ptr(y);
		const uchar* m2Ptrs[MXSIZE];
		int count = 0;
		for (int y2 = y - m2Radius; y2 <= y + m2Radius; ++y2) {
			if (y2 < 0 || y2 >= m1.rows) {
				m2Ptrs[count] = NULL;
			}
			else {
				m2Ptrs[count] = m2.ptr(y2);
			}

			++count;
		}
		uchar* diffPtr = diff.ptr(y);

		for (int x = 0; x < m1.cols; ++x) {
			bool isInBg = false;
			for (int i = 0; i < count && !isInBg; ++i) {
				if (m2Ptrs[i]) {
					for (int x2 = x - m2Radius; x2 <= x + m2Radius && !isInBg; ++x2) {
						if (x2 >= 0 && x2 < m1.cols) {
							if (m2Ptrs[i][x2] > 0) {
								isInBg = true;
							}
						}
					}
				}
			}

			if (m1Ptr[x] > 0 && !isInBg) {
				diffPtr[x] = 255;
			}
			else {
				diffPtr[x] = 0;
			}
		}
	}
}

void LrTextureShadRem::getSkeleton(const cv::Mat& mask, cv::Mat& skeleton) {
	cv::Mat tmpMask = mask.clone();
	tmpMask.copyTo(skeleton);

	bool changed = true;
	while (changed) {
		changed = false;

		for (int k = 0; k < (int) skeletonKernels.size(); ++k) {
			for (int y = 1; y < tmpMask.rows - 1; ++y) {
				uchar* skeletonPtr = skeleton.ptr(y);

				for (int x = 1; x < tmpMask.cols - 1; ++x) {
					bool allMatch = true;
					for (int dy = -1; dy <= 1 && allMatch; ++dy) {
						uchar* maskPtr = tmpMask.ptr(y + dy);
						const uchar* kernelPtr = skeletonKernels[k].ptr(1 + dy);

						for (int dx = -1; dx <= 1 && allMatch; ++dx) {
							uchar maskVal = maskPtr[x + dx];
							uchar kernelVal = kernelPtr[1 + dx];

							if (kernelVal != 127 && maskVal != kernelVal) {
								allMatch = false;
							}
						}
					}

					if (allMatch) {
						skeletonPtr[x] = 0;
						changed = true;
						break;
					}
				}
			}

			if (changed) {
				skeleton.copyTo(tmpMask);
			}
		}
	}
}

std::vector<cv::Mat> LrTextureShadRem::getSkeletonKernels() {
	std::vector<cv::Mat> skeletonKernels(8);

	skeletonKernels[0].create(3, 3, CV_8U);
	skeletonKernels[0].at<uchar>(0, 0) = 0;
	skeletonKernels[0].at<uchar>(0, 1) = 0;
	skeletonKernels[0].at<uchar>(0, 2) = 0;
	skeletonKernels[0].at<uchar>(1, 0) = 127;
	skeletonKernels[0].at<uchar>(1, 1) = 255;
	skeletonKernels[0].at<uchar>(1, 2) = 127;
	skeletonKernels[0].at<uchar>(2, 0) = 255;
	skeletonKernels[0].at<uchar>(2, 1) = 255;
	skeletonKernels[0].at<uchar>(2, 2) = 255;

	skeletonKernels[2] = skeletonKernels[0].t();
	cv::flip(skeletonKernels[0], skeletonKernels[4], 0);  //将Mat沿X轴翻转
	cv::flip(skeletonKernels[2], skeletonKernels[6], 1);  //将Mat沿y轴翻转

	skeletonKernels[1].create(3, 3, CV_8U);
	skeletonKernels[1].at<uchar>(0, 0) = 127;
	skeletonKernels[1].at<uchar>(0, 1) = 0;
	skeletonKernels[1].at<uchar>(0, 2) = 0;
	skeletonKernels[1].at<uchar>(1, 0) = 255;
	skeletonKernels[1].at<uchar>(1, 1) = 255;
	skeletonKernels[1].at<uchar>(1, 2) = 0;
	skeletonKernels[1].at<uchar>(2, 0) = 127;
	skeletonKernels[1].at<uchar>(2, 1) = 255;
	skeletonKernels[1].at<uchar>(2, 2) = 127;

	cv::flip(skeletonKernels[1], skeletonKernels[3], 1);
	cv::flip(skeletonKernels[3], skeletonKernels[5], 0);
	cv::flip(skeletonKernels[5], skeletonKernels[7], 1);

	return skeletonKernels;
}
//使用色度信息HSV来粗略得到候选阴影区域（包含真实的阴影和前景）
void LrTextureShadRem::getCandidateShadows(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg,
		cv::Mat& hsvMask) {
	float vThreshLower = (
			avgAtten < params.avgAttenThresh ? params.vThreshLowerLowAtten : params.vThreshLowerHighAtten);
	float vThreshUpper = (
			avgAtten < params.avgAttenThresh ? params.vThreshUpperLowAtten : params.vThreshUpperHighAtten);
	float hThresh = (avgSat < params.avgSatThresh ? params.hThreshLowSat : params.hThreshHighSat);
	float sThresh = (avgSat < params.avgSatThresh ? params.sThreshLowSat : params.sThreshHighSat);

	hsvMask.create(hsvFrame.size(), CV_8U);
	hsvMask.setTo(cv::Scalar(0));
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* hsvFramePtr = hsvFrame.ptr(y);
		const uchar* hsvBgPtr = hsvBg.ptr(y);
		const uchar* fgPtr = fg.ptr(y);
		uchar* hsvMaskPtr = hsvMask.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				float vRatio = (float) hsvFramePtr[x * 3 + 2] / hsvBgPtr[x * 3 + 2];
				bool vIsShadow = (vRatio > vThreshLower && vRatio < vThreshUpper);

				int hDiff = CV_IABS(hsvFramePtr[x * 3] - hsvBgPtr[x * 3]);
				if (hDiff > 90) {
					hDiff = 180 - hDiff;
				}
				bool hIsShadow = (hDiff < hThresh);

				int sDiff = hsvFramePtr[x * 3 + 1] - hsvBgPtr[x * 3 + 1];
				bool sIsShadow = (sDiff < sThresh);

				if (vIsShadow && hIsShadow && sIsShadow) {
					hsvMaskPtr[x] = 255;
				}
			}
		}
	}
}


/*
入口参数： 前景信息fg  candidateShadows粗略的候选阴影区域
目的： 得到运动目标和阴影连接的边缘而尽可能去掉阴影和目标的外轮廓以及前景内部杂乱的边缘
	   理想输出的 cannyDiff 仅含有阴影和目标连接的边缘，这样在后续步骤中可以减操作可以将真实阴影和目标混合的连通域分割为两个而不
	   损失其他前景部分的外围或者内部信息。
*/
void LrTextureShadRem::getEdgeDiff(const cv::Mat& grayFrame, const cv::Mat& grayBg, const ConnCompGroup& fg,
		const cv::Mat& candidateShadows, cv::Mat& cannyFrame, cv::Mat& cannyBg, cv::Mat& cannyDiffWithBorders,
		cv::Mat& borders, cv::Mat& cannyDiff) {         //只关注候选阴影处的边缘
	cv::Mat invCandidateShadows(candidateShadows.size(), CV_8U, cv::Scalar(255));
	invCandidateShadows.setTo(0, candidateShadows);     //invCandidateShadows 与 candidateShadows元素值相反

	cv::Canny(grayFrame, cannyFrame, params.cannyThresh1, params.cannyThresh2, params.cannyApertureSize,
			params.cannyL2Grad);
	
	cannyFrame.setTo(0, invCandidateShadows);     //invCandidateShadow中属于前景的像素为0  因此将cannyFrame中不属于前景的像素值置0
                                                  //等价于cannyFrame只表示运动前景的当前图像边缘？
	cv::Canny(grayBg, cannyBg, params.cannyThresh1, params.cannyThresh2, params.cannyApertureSize, params.cannyL2Grad);
	cannyBg.setTo(0, invCandidateShadows);        //同上cannyBg只表示不包含前景的背景的边缘
											  //edgeDiffRadius是否四邻域判断
	int edgeDiffRadius = (avgPerim > params.avgPerimThresh ? params.edgeDiffRadius : 0);

	maskDiff(cannyFrame, cannyBg, cannyDiffWithBorders, edgeDiffRadius);
	          //cannyDiffWithBorders 表示运动前景在背景图像上产生的最外边缘  params.borderDiffRadius 0
	int borderDiffRadius = (avgAtten < params.avgAttenThresh ? params.borderDiffRadius : 2);
	borderDiffRadius = (avgPerim > params.avgPerimThresh ? borderDiffRadius : 0);
	borders.create(fg.mask.size(), CV_8U);
	borders.setTo(cv::Scalar(0));
	fg.draw(borders, cv::Scalar(255, 255, 255), false);
	if (borderDiffRadius > 0) {                   //borderDiffRadius 视频平均亮度大，前景平均轮廓长，则borders膨胀次数多   
		cv::dilate(borders, borders, cv::Mat(borderDiffRadius, borderDiffRadius, CV_8U, cv::Scalar(255)));
	}
	int splitIncrement = (avgAtten < params.avgAttenThresh ? params.splitIncrement : 0);
	splitIncrement = (avgPerim > params.avgPerimThresh ? splitIncrement : 2);
	cannyDiffWithBorders.copyTo(cannyDiff);
	cannyDiff.setTo(0, borders);    //borders:前景的轮廓
	if (splitIncrement > 0) {
		cv::dilate(cannyDiff, cannyDiff, cv::Mat(splitIncrement, splitIncrement, CV_8U, cv::Scalar(255)));
		//getSkeleton(cannyDiff, cannyDiff); //？
	}

}

/*
函数解析：


*/
float LrTextureShadRem::getGradDirCorr(const cv::Mat& grayFrame, const ConnComp& cc, const cv::Mat& grayBg) {
	std::vector<int> deltas(params.gradScales);
	deltas[0] = 1;
	for (int i = 1; i < params.gradScales; ++i) {
		deltas[i] = pow(2, i);  //2^i
	}
	int minCorrPoints = cc.pixels.size()/3;
	//int minCorrPoints = (avgPerim > params.avgPerimThresh ? params.minCorrPoints : 3);
	int round = 1;
	int nGradDirs = -1;
	int nSimilarGradDirs = 0;
	//外层while循环，如果一轮里得到的有效梯度数据点的数目较小 再round一轮，且梯度大小阈值自适应降低
	while (nGradDirs < minCorrPoints && round <= params.maxCorrRounds) {
		nGradDirs = 0;
		nSimilarGradDirs = 0;
        //依次处理连通域中的像素点
		for (int i = 0; i < (int) cc.pixels.size(); ++i) {  
			int x = cc.pixels[i].x;
			int y = cc.pixels[i].y;

			if (x < grayFrame.cols - 1 && y < grayFrame.rows - 1) {
				int boxMaskX = x - cc.box.x;
				int boxMaskY = y - cc.box.y;

				float frMaxGradMag = -1;
				float frMaxGradDir;
				float bgMaxGradMag = -1;
				float bgMaxGradDir;
				//不同间隔的梯度求解 梯度 并记录梯度值最大的值 。比如：Gx = x - (x+1) / Cx = x - (x+2) / Gx = x - (x+4)
				for (int d = 0; d < (int) deltas.size(); ++d) {
					// avoid pixels in the border of the component
					int boxMaskX1 = boxMaskX - params.corrBorder;
					int boxMaskX2 = boxMaskX + params.corrBorder + deltas[d];
					int boxMaskY1 = boxMaskY - params.corrBorder;
					int boxMaskY2 = boxMaskY + params.corrBorder + deltas[d];
					if (boxMaskX1 >= 0 && boxMaskX2 < cc.box.width && boxMaskY1 >= 0 && boxMaskY2 < cc.box.height) {
						const uchar* ccMaskU = cc.boxMask.ptr(boxMaskY1) + boxMaskX;
						const uchar* ccMaskD = cc.boxMask.ptr(boxMaskY2) + boxMaskX;
						const uchar* ccMaskL = cc.boxMask.ptr(boxMaskY) + (boxMaskX1);
						const uchar* ccMaskR = cc.boxMask.ptr(boxMaskY) + (boxMaskX2);
						//确保搜索的像素都在连通域内
						if ((*ccMaskU > 0) && (*ccMaskD > 0) && (*ccMaskL > 0) && (*ccMaskR > 0)) {
							// search scale with largest gradient

							// calculate gradients
							float frVal = grayFrame.ptr(y)[x];
							float frNextXVal = grayFrame.ptr(y)[x + deltas[d]];
							float frNextYVal = grayFrame.ptr(y + deltas[d])[x];
							float bgVal = grayBg.ptr(y)[x];
							float bgNextXVal = grayBg.ptr(y)[x + deltas[d]];
							float bgNextYVal = grayBg.ptr(y + deltas[d])[x];
							float frGx = frNextXVal - frVal;
							float frGy = frVal - frNextYVal;
							float bgGx = bgNextXVal - bgVal;
							float bgGy = bgVal - bgNextYVal;
							float frMagnitude = sqrt(frGx * frGx + frGy * frGy);
							float bgMagnitude = sqrt(bgGx * bgGx + bgGy * bgGy);

							if (frMagnitude + bgMagnitude > frMaxGradMag + bgMaxGradMag) {
								frMaxGradMag = frMagnitude;
								bgMaxGradMag = bgMagnitude;
								frMaxGradDir = atan2(frGy, frGx);
								bgMaxGradDir = atan2(bgGy, bgGx);
							}
						}
					}
				}

				// use gradient if magnitude is big enough 梯度大小大于一定阈值（随着round自适应减小）的数据才使用 认为是有效信息 以滤除噪声
				//记录有效点 nGradDirs 的个数，并记录其中相似点的个数 nSimilarGradDirs。 
				float magThresh = params.gradMagThresh / pow(2, round);
				if (frMaxGradMag > magThresh || bgMaxGradMag > magThresh) {
					if (bgMaxGradMag / frMaxGradMag > params.gradAttenThresh) {
						float angle = fabs(frMaxGradDir - bgMaxGradDir);
						angle = std::min(angle, (float) (2 * CV_PI - angle));

						// add to gradient directions sums
						if (angle < params.gradDistThresh) {
							++nSimilarGradDirs;
						}
					}
					//printf("%.2f ", fabs(frMaxGradDir - bgMaxGradDir));
					++nGradDirs;
				}
			}
		}

		++round;
	}

	if (nGradDirs < minCorrPoints)
		return -2;
	if (nGradDirs == 0)
		return -1;

	return (float) nSimilarGradDirs / nGradDirs;
}
