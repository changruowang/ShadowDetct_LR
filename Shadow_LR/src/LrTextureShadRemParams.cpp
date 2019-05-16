// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <cxcore.h>
#include "LrTextureShadRemParams.h"

/*
参数理解：
1. avgPerimThresh：整个视频轮廓长短的阈值，统计得到的平均阈值大于它则轮廓较长 否则较短
   avgAttenThresh：整个视频平均亮度阈值，大于亮，小于暗
   avgSat        ：整个视频平均S^2/(S^2 + V^2)的阈值
（1）. 如果图像较亮，HSV粗筛选阴影阈值自动切换为高低阈值（第（1）组参数），保证粗筛选的候选阴影区域包含全部的实际阴影区域
（2）. 如果avgSat > avgSatThresh, 切换（第（2）组参数）高低，保证粗筛选的候选阴影区域包含全部的实际阴影区域
（3）. 如果平均边缘长度较长，使用edgeDiffRadius较大，增加maskDiff采样宽度，使得做差后实际阴影内部区域较平滑
       如果平均亮度较大且边缘长度较长 borderDiffRadius=2，如果边缘长度较短，borderDiffRadius=0，如果边缘长度较长且亮度较小使用初始值
	                                  borderDiffRadius越大，board连通域外边缘轮廓越粗，后面损失的边缘信息越少  
       如果平均亮度较大且边缘长度较长 splitIncrement=2，如果边缘长度较短，splitIncrement=0，如果边缘长度较长且亮度较小使用初始值
	                                  splitIncrement越大，值较大，目标和阴影的连接处边缘越强。
（4）.目标和阴影处的边缘强化，越大，间隔分割的越大

（6）.梯度阈值，平均亮度较大时，相似梯度的像素数目的比例阈值较大，意味着某区域内需要有较大比例的梯度相似像素点才行
（7）. gradMagThresh梯度大小阈值，只有当梯度大小大于该值，才计入有梯度效数据点，才会接着判断相似性
	   gradScales  不同间隔的梯度求解 梯度 并记录梯度值最大的值 。比如：Gx = x - (x+1) / Cx = x - (x+2) / Gx = x - (x+4)
       gradAttenThresh前景背景梯度大小相似性阈值  只有当 bgMaxGradMag/frMaxGradMag>Thresh 时才接着判断角度相似性阈值
	   gradDistThresh 角度相似性阈值，前景背景向量的夹角
	   minCorrPoints 最小有用数据点阈值。在一轮循环后若统计得到的有效的数据点数小于该值，则降低梯度大小阈值，再循环一轮
	   maxCorrRounds 循环的最大轮数 与上一个参数配合
	   corrBorder    跳过连通域边缘corrBorder个像素不统计
minCorrPoint最小游泳的梯度数据。       
 */

LrTextureShadRemParams::LrTextureShadRemParams() {
	avgAttenThresh = 1.6;  ////平均亮度阈值1.58
	avgPerimThresh = 2;     ////Line335  平均单个前景的轮廓长度阈值 100
	avgSatThresh = 35;
//参数 （1）
	vThreshUpperLowAtten = 1;
	vThreshUpperHighAtten = 0.99;
	vThreshLowerLowAtten = 0.33;
	vThreshLowerHighAtten = 0.33;
//参数（2）
	hThreshLowSat = 90;
	hThreshHighSat = 90;
	sThreshLowSat = 90;
	sThreshHighSat = 90;
//参数（3）  主要在 getEdgeDiff 中应用
	edgeDiffRadius = 1;     ////Line352  平均单个轮廓的周长大于阈值选定 否则为0 增加该值会使Line353 cannyDiffWithBorders轮廓平滑
	borderDiffRadius = 0;   ////Line355  平均亮度小于阈值时选定 否则为2
	splitIncrement = 1; 

	cannyThresh1 = 72;       //canny参数
	cannyThresh2 = 94;		 //canny参数	
	cannyApertureSize = 3;   //canny参数
	cannyL2Grad = true;      //canny参数
//参数（4）    
	splitRadius = 1;

//参数（6）
	gradCorrThreshLowAtten = 0.1; //Line373 低光照度阈值    =>gradCorrThresh代表候选阴影region中前背景纹理相似的像素点的比例 阈值 大于该值为阴影
	gradCorrThreshHighAtten = 0.08;//Line373 高光照度阈值    =>gradCorrThresh
//参数（7）
	gradMagThresh = 8;       //梯度阈值，当像素梯度大于该值时才计入统计  6
	gradAttenThresh = 0.1;   //bgMaxGradMag / frMaxGradMag > gradAttenThresh 梯度大小相似度阈值
	gradDistThresh = CV_PI/5;  //角度相似度阈值，单位为角度，前景和背景相应位置的梯度向量的夹角，小于该值认为纹理相似
	gradScales = 3;          //梯度尺寸
	minCorrPoints = 30;
	maxCorrRounds = 3;       //Line380
	corrBorder = 1;   //跳过太靠边缘的像素不统计 越大跳的像素越多


	cleanShadows = true;     //是否对阴影区域进行闭操作（去除小黑点）
	fillShadows = true;      
	minShadowPerim = 19;     //最小的阴影像素数目阈值 连通域操作中对联通区像素数目的滤波
	cleanSrMask = false;
	fillSrMask = false;
}
/*
底线参数：  CV_PI/5   0.08
*/

LrTextureShadRemParams::~LrTextureShadRemParams() {
}
