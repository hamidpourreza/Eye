//
// Created by MSI on 8/24/2023.
//

#ifndef CPP_DETECTOR_H
#define CPP_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "config.h"
#include "engine.h"
using namespace cv::dnn;

class Detector {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat SK = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat LK = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6));
    double mClipLimit = 32;
    cv::Size mTitleGridSize;
    Options options;

    Engine engine;
    cv::Ptr<cv::CLAHE> mClahe;
    int count=0;
public:
    Detector(const std::string& modelPath);

    cv::Mat patchInference(cv::cuda::GpuMat patch);
    std::pair<cv::Mat,cv::Mat> postProcess(cv::Mat input, double ignoreThresh, double areaThresh);

    


    cv::Mat inference(cv::Mat img);
    std::pair<float,float> calcSize(const cv::Mat& input,float scaleX,float scaleY);
    cv::Mat enhanced(const cv::Mat &img);
    std::map<std::string, int> report(cv::Mat mask);


};


#endif //CPP_DETECTOR_H
