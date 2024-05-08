//
// Created by ai-innovate on 9/25/23.
//

#include "CDevice.h"

void CDevice::SetAlarmDuration(uint duration) {
    mAlarmDuration.store(duration);
}

void CDevice::SetAlarmDelay(uint delay) {
    mAlarmDelay.store(delay);
}

void CDevice::EnableEjector() {
//    std::this_thread::sleep_for(std::chrono::milliseconds(mAlarmDelay));
//    if (mEnable) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mIsEjectorEnable = true;
        }
        if (mHighLowSignal == EJECTOR_STATUS::HIGH) {
            // set pin to low
            std::system("echo omid1234 | sudo -S ./low.sh");

//            GPIO::output(mGPIO, GPIO::LOW);
//            gpioWrite(GPIO, 0);
        } else {
            std::system("echo omid1234 | sudo -S ./high.sh");
            // set pin to high
//            GPIO::output(mGPIO, GPIO::HIGH);
//            gpioWrite(GPIO, 1);

        }
//    }
//    mEnable = false;
}

void CDevice::DisableEjector() {
//    std::this_thread::sleep_for(std::chrono::milliseconds(mAlarmDuration));
    mIsEjectorEnable = false;

//    if (mEnable) {
//        std::lock_guard<std::mutex> lock(mMutex);
        if (mHighLowSignal == EJECTOR_STATUS::HIGH) {
            std::system("echo omid1234 | sudo -S ./high.sh");
            // set pin to high
//            GPIO::output(mGPIO, GPIO::HIGH);
        } else {
            std::system("echo omid1234 | sudo -S ./low.sh");
//            GPIO::output(mGPIO, GPIO::LOW);
            // set pin to low
        }
//    }
}

void CDevice::SetEjectorDefaultStatus(EJECTOR_STATUS status) {
    mHighLowSignal = status;
    if (mHighLowSignal == EJECTOR_STATUS::HIGH) {
        std::system("echo omid1234 | sudo -S ./high.sh");
    } else {
        std::system("echo omid1234 | sudo -S ./low.sh");
    }
}

CDevice *CDevice::GetDevice() {
    if (!mCDevice) {
        std::system("echo omid1234 | sudo -S ./gpio.sh");

        mCDevice = new CDevice();
//        GPIO::setmode(GPIO::TEGRA_SOC);
//        GPIO::setup(mGPIO, GPIO::OUT);
////        auto ret = gpioInitialise();
//        if (ret < 0) {
//            std::cerr << "Failed to init gpio." << std::endl;
//        }
//        ret = gpioSetMode(GPIO, JET_OUTPUT);
//        if (ret < 0 ) {
//            std::cerr << "gpio setting up failed." << std::endl;
//        }
    }
    return mCDevice;
}

bool CDevice::IsEjectorEnable() {
    return mIsEjectorEnable;
}

void CDevice::setEnable(bool value) {
    mEnable = value;
}

void CDevice::setDefault() {

}

uint CDevice::GetAlarmDelay() const {
    return mAlarmDelay.load();
}

uint CDevice::GetAlarmDuration() const {
    return mAlarmDuration.load();
}
