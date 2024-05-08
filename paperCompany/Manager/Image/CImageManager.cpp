//
// Created by veerayco on 9/21/23.
//

#include "CImageManager.h"

void CImageManager::Push(const cv::Mat &input) {
    std::lock_guard<std::mutex> guard(mMutex);
    mImagesQueue.push(input.clone());
    if (mImagesQueue.size() > mQueueMaxSize) {
        mImagesQueue.pop();
    }
}

cv::Mat CImageManager::Last() {
    std::lock_guard<std::mutex> guard(mMutex);
    if (mImagesQueue.empty()) {
        return mLastImage;
    }
    auto img = mImagesQueue.back();
//    mImagesQueue.pop();
//    if (mImagesQueue.empty()) {
//        mLastImage = img.clone();
//    }
    return img;
}

size_t CImageManager::CurrentLength() const {
    std::lock_guard<std::mutex> guard(mMutex);
    auto len = mImagesQueue.size();
    return len;
}

cv::Mat CImageManager::Front() {
    std::lock_guard<std::mutex> guard(mMutex);
    auto img = mImagesQueue.front();
    mImagesQueue.pop();
    return img;
}
