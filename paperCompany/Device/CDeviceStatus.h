//
// Created by veerayco on 10/8/23.
//

#ifndef MULTI_CAMERA_CDEVICESTATUS_H
#define MULTI_CAMERA_CDEVICESTATUS_H
#include <atomic>
#include <string>
#include <map>
#include <functional>

class CDeviceStatus {
    inline static std::function<void(std::string_view, int)> mNotifyCallback;
public:

    static void addNotifyCallback(const std::function<void(std::string_view, int)> &callback);

    bool isCamConnected() const;

    void setCamConnect(bool status);

    bool isPaperEnter() const;

    void setPaperEnter(bool status);

    bool isProcessStart() const;

    void setProcessStart(bool status);

    bool isSoftwareStart() const;

    void setSoftwareStatus(bool status);

    void setTotalCounter(int counter);

    int totalCounter() const;

private:
    std::atomic<bool> mCamStatus{};
    std::atomic<bool> mPaperEnter{};
    std::atomic<bool> mProcessStart{};
    std::atomic<bool> mSoftwareStatus{};
    std::atomic<int>  mTotalCounter{};
};


#endif //MULTI_CAMERA_CDEVICESTATUS_H
