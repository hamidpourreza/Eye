//
// Created by veerayco on 10/4/23.
//

#ifndef MULTI_CAMERA_CIMAGE_H
#define MULTI_CAMERA_CIMAGE_H
#include <opencv2/opencv.hpp>
#include "../detectorGPU/Detector.h"

class CImage {
    cv::Mat mImage{};
    uint mWidth{};
    uint mHeight{};
    uint mNumOfInsert{};
    uint mCurrentRow{};

    void checkNumOfRows() {
        if (mCurrentRow >= mHeight) {
            ResetCurrentRow();
        }
    }

public:
    CImage(uint rows, uint cols) {
        mImage = cv::Mat::ones(rows + padd, cols, CV_8UC1) * 255;
        mWidth = cols;
        mHeight = rows;
    }

    void InsertRows(uchar *data, uint numberOfRows, uint start, uint end) {
        checkNumOfRows();
        auto img = cv::Mat(cv::Size(mWidth, numberOfRows), CV_8UC1, data);
        auto size = end - start;
        img.rowRange(start, end).copyTo(mImage.rowRange(mCurrentRow, mCurrentRow + size));
        IncrementCurrentRow(size);
    }

    uchar *ImageData() const {
        return mImage.data;
    }

    void NumOfRows() const {

    }

    cv::Mat ImageMat() const {
        return mImage;
    }

    uint CurrentRow() const {
        return mCurrentRow;
    }

    uint IncrementCurrentRow(uint currentRow) {
        mCurrentRow += currentRow;
        return mCurrentRow;
    }

    void ResetCurrentRow() {
        mCurrentRow = 0;
    }
};

#endif //MULTI_CAMERA_CIMAGE_H
