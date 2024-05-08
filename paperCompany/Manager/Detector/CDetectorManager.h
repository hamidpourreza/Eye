//
// Created by veerayco on 9/21/23.
//

#ifndef MULTI_CAMERA_CDETECTORMANAGER_H
#define MULTI_CAMERA_CDETECTORMANAGER_H

#include <atomic>
#include <opencv2/opencv.hpp>
#include "../../detectorGPU/Detector.h"

const std::string PATH = "./model/wide.onnx";

struct DetectorUnit {
    DetectorUnit() {
        mDetector = std::make_unique<Detector>(PATH);
    }

    void resetProcessFinished() {
        mIsProcessFinished = true;
    }

    std::unique_ptr<Detector> mDetector{};
    std::atomic<bool> mIsProcessFinished{true};

    ~DetectorUnit() {
        std::cout << "******Detector destroyed.******";
    }
};

class CDetectorManager {
    std::mutex mMutex{};
    std::vector<std::unique_ptr<DetectorUnit>> mDetectorUnitList{};
    uint mNumberOfTotalProcess{};
    std::atomic<uint> mNumberOfRunningProcess{};

    bool isAllProcessRunning();

    DetectorUnit *CreateNewProcess();

public:
    explicit CDetectorManager(uint numberOfTotalProcess) : mNumberOfTotalProcess(numberOfTotalProcess) {

    }

    void resetDetector(DetectorUnit *dt);

    void InitProcess();

    uint NumberOfTotalProcess();

    DetectorUnit *GetFreeDetector();

    int NumberOfRunningProcess();
};


#endif //MULTI_CAMERA_CDETECTORMANAGER_H
