//
// Created by ai-innovate on 29/11/23.
//

#ifndef MULTI_CAMERA_CWRITER_H
#define MULTI_CAMERA_CWRITER_H

#include <opencv2/opencv.hpp>

class CWriter {
private:
    inline static uint index{};
public:
    static void Write(const cv::Mat &image, std::string_view filename);
};


#endif //MULTI_CAMERA_CWRITER_H
