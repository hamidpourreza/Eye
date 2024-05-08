//
// Created by ai-innovate on 30/11/23.
//

#ifndef MULTI_CAMERA_CONSTANT_H
#define MULTI_CAMERA_CONSTANT_H
#include <bits/stdc++.h>
#include "../detectorGPU/config.h"

constexpr uint NUM_OF_DETECTOR = 1;
constexpr int MIN_NUM_OF_LINES_PER_REQUEST = 16;
constexpr int NUM_OF_ROWS = 32 * MIN_NUM_OF_LINES_PER_REQUEST + padd;
constexpr uint MAX_IMAGE_INDEX = 3;
constexpr uint MAX_BATCH = 4;

#endif //MULTI_CAMERA_CONSTANT_H
