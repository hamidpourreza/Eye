//
// Created by veerayco on 10/8/23.
//

#include "CDeviceStatus.h"

bool CDeviceStatus::isCamConnected() const {
    return mCamStatus;
}

void CDeviceStatus::setCamConnect(bool status) {
    CDeviceStatus::mCamStatus = status;
    mNotifyCallback("camConnected", status);
}

bool CDeviceStatus::isPaperEnter() const {
    return mPaperEnter;
}

void CDeviceStatus::setPaperEnter(bool status) {
    CDeviceStatus::mPaperEnter = status;
    mNotifyCallback("paperEntered", status);
}

bool CDeviceStatus::isProcessStart() const {
    return mProcessStart;
}

void CDeviceStatus::setProcessStart(bool status) {
    CDeviceStatus::mProcessStart = status;
    mNotifyCallback("processStarted", status);
}

bool CDeviceStatus::isSoftwareStart() const {
    return mSoftwareStatus;
}

void CDeviceStatus::setSoftwareStatus(bool status) {
    CDeviceStatus::mSoftwareStatus = status;
    mNotifyCallback("softwareStarted", status);
}

void CDeviceStatus::addNotifyCallback(const std::function<void(std::string_view, int)> &callback) {
    mNotifyCallback = callback;
}

void CDeviceStatus::setTotalCounter(int counter) {
    mTotalCounter = counter;
    mNotifyCallback("totalCounter", counter);

}

int CDeviceStatus::totalCounter() const {
    return mTotalCounter;
}
