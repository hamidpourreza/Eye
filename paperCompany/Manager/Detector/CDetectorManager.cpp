//
// Created by veerayco on 9/21/23.
//

#include "CDetectorManager.h"

bool CDetectorManager::isAllProcessRunning() {
    return false;
}

DetectorUnit *CDetectorManager::CreateNewProcess() {
    std::cout << __func__ << std::endl;
    mDetectorUnitList.emplace_back(new DetectorUnit);
    return mDetectorUnitList.back().get();
}

void CDetectorManager::InitProcess() {
    mDetectorUnitList.resize(mNumberOfTotalProcess);
    for (auto &item: mDetectorUnitList) {
        item = std::make_unique<DetectorUnit>();
    }
}

uint CDetectorManager::NumberOfTotalProcess() {
//    std::lock_guard<std::mutex> lock(mMutex);
    return mDetectorUnitList.size();
}

DetectorUnit *CDetectorManager::GetFreeDetector() {
    DetectorUnit *detector = nullptr;
    std::lock_guard<std::mutex> guard(mMutex);
    for (auto const &item: mDetectorUnitList) {

        if (item->mIsProcessFinished) {
            detector = item.get();
            break;
        }
    }
    if (!detector) {
        detector = CreateNewProcess();
    }
    detector->mIsProcessFinished = false;
    return detector;
}

int CDetectorManager::NumberOfRunningProcess() {
    std::lock_guard<std::mutex> lock(mMutex);
    int running{};
    for (auto const &item: mDetectorUnitList) {
        if (!item->mIsProcessFinished) {
            running++;
        }
    }
    return running;
}

void CDetectorManager::resetDetector(DetectorUnit *dt) {
    std::lock_guard<std::mutex> lock(mMutex);
//    auto point = mDetectorUnitList.begin();
//    if (mDetectorUnitList.size() > mNumberOfTotalProcess) {
//        for (auto it = mDetectorUnitList.begin(); it != mDetectorUnitList.end(); it++) {
//            if (dt == it->get()) {
//                point = it;
//                std::cout << "******************REMOVE POINTER******************" << std::endl;
//                break;
//            }
//        }
////        point->get()->resetProcessFinished();
//        point->reset();
//
//        mDetectorUnitList.erase(point);
//
//
//    } else {
        dt->resetProcessFinished();
//    }
}
