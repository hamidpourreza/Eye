//
// Created by MSI on 8/24/2023.
//

#ifndef CPP_CONFIG_H
#include <opencv2/opencv.hpp>
#define CPP_CONFIG_H

const int patchSize=512;
const cv::Scalar pMean=cv::Scalar(0.485, 0.456, 0.406);
const cv::Scalar pStd=cv::Scalar(0.229, 0.224, 0.225);
const int padd=32;
const int batchSize=12;
const std::array<float, 3> subVals {0.485, 0.456, 0.406};
const std::array<float, 3> divVals {0.229, 0.224, 0.225};
const int mxbs=12;
const bool normalize = true;
#endif //CPP_CONFIG_H
