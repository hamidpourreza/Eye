//
// Created by veerayco on 9/21/23.
//

#ifndef MULTI_CAMERA_CIMAGEMANAGER_H
#define MULTI_CAMERA_CIMAGEMANAGER_H
#include <opencv2/opencv.hpp>

class CImageManager {
    mutable std::mutex mMutex{};
    std::queue<cv::Mat> mImagesQueue{};
    uint mQueueMaxSize{};
    uint mQueueCurrentSize{};
    cv::Mat mLastImage;
public:
    explicit CImageManager(uint size) : mQueueMaxSize(size) {
        mLastImage = cv::Mat(10, 10, CV_8UC3, cv::Scalar(255, 255, 255));
    }

    void Push(const cv::Mat &input);

    cv::Mat Last();

    size_t CurrentLength() const;
    cv::Mat Front();
};


#endif //MULTI_CAMERA_CIMAGEMANAGER_H
