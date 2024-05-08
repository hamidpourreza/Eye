//
// Created by ai-innovate on 9/25/23.
//

#ifndef MULTI_CAMERA_CDEVICE_H
#define MULTI_CAMERA_CDEVICE_H

#include <iostream>
#include <unistd.h>
#include <mutex>
#include <atomic>
#include <thread>
//#include <JetsonGPIO.h>

enum class EJECTOR_STATUS {
    HIGH,
    LOW
};

class CDevice {
    std::mutex mMutex{};
protected:
    CDevice() = default;
    const static int mGPIO = 18;
    inline static CDevice *mCDevice = nullptr;
    inline static bool mIsEjectorEnable{false};

    bool mEnable{};
    EJECTOR_STATUS mHighLowSignal{};
    std::atomic<uint> mAlarmDuration{};
    std::atomic<uint> mAlarmDelay{};
public:
    CDevice(const CDevice &) = delete;

    void operator=(const CDevice &) = delete;

    static CDevice *GetDevice();

    void SetAlarmDuration(uint duration);

    void SetAlarmDelay(uint delay);

    uint GetAlarmDelay() const;

    uint GetAlarmDuration() const;

    void SetEjectorDefaultStatus(EJECTOR_STATUS status);

    void EnableEjector();

    void DisableEjector();

    bool IsEjectorEnable();

    void setEnable(bool value);

    void setDefault();
};


#endif //MULTI_CAMERA_CDEVICE_H
