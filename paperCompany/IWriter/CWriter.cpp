//
// Created by ai-innovate on 29/11/23.
//

#include "CWriter.h"

void CWriter::Write(const cv::Mat &image, std::string_view filename) {
    auto path = "./result/" + std::string{filename} + ".jpg";
    cv::imwrite(path, image);
}
