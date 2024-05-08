//
// Created by veerayco on 10/21/23.
//

#ifndef MULTI_CAMERA_CDEVICEPARAMS_H
#define MULTI_CAMERA_CDEVICEPARAMS_H

#include <bits/stdc++.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class CDeviceParams {
    std::atomic<float> mExposureTime;
    std::atomic<float> mDivider;
    std::atomic<float> mMultiplier;
    std::atomic<float> mAlarmDelay;
    std::atomic<float> mAlarmDuration;
    std::atomic<bool> mAlarmNormallyHighLow;
    std::atomic<bool> mImageContrastEnhancement;
    std::atomic<float> mBSBPLeft;
    std::atomic<float> mBSBPVLeft;
    std::atomic<float> mBSBPRight;
    std::atomic<float> mBSBPVRight;
    std::atomic<float> mBSDPCenter;
    std::atomic<float> mBSDPVCenter;
    std::atomic<float> mEnterDebouncing;
    std::atomic<float> mExitDebouncing;
    std::atomic<float> mDefectSizeFilter;
    std::atomic<float> mDefectStrengthFilter;
    std::atomic<float> mMinDefectSizeForAlarm;
    std::atomic<float> mScaleX;
    std::atomic<float> mScaleY;

public:
    float BSBPLeft() const;

    void setBSBPLeft(float mBsbpLeft);

    float BSBPVLeft() const;

    void setBSBPVLeft(float mBsbpvLeft);

    float BSBPRight() const;

    void setBSBPRight(float mBsbpRight);

    float BSBPVRight() const;

    void setBSBPVRight(float mBsbpvRight);

    float BSDPCenter() const;

    void setBSDPCenter(float mBsdpCenter);

    float BSDPVCenter() const;

    void setBSDPVCenter(float mBsdpvCenter);

    float enterDebouncing() const;

    void setEnterDebouncing(float enterDebouncing);

    float exitDebouncing() const;

    void setExitDebouncing(float exitDebouncing);

    float scaleX() const;

    void setScaleX(float scaleX);

    float scaleY() const;

    void setScaleY(float scaleY);

    void Fill(std::string_view input);

    float ExposureTime() const;

    void ExposureTime(float ExposureTime);

    float Divider() const;

    void Divider(float Divider);

    float Multiplier() const;

    void Multiplier(float Multiplier);

    float AlarmDelay() const;

    void AlarmDelay(float AlarmDelay);

    float AlarmDuration() const;

    void AlarmDuration(float AlarmDuration);

    bool AlarmNormallyHighLow() const;

    void AlarmNormallyHighLow(bool AlarmNormallyHighLow);

    bool ImageContrastEnhancement() const;

    void ImageContrastEnhancement(bool ImageContrastEnhancement);

    float DefectSizeFilter() const;

    void DefectSizeFilter(float DefectSizeFilter);

    float DefectStrengthFilter() const;

    void DefectStrengthFilter(float DefectStrengthFilter);

    float MinDefectSizeForAlarm() const;

    void MinDefectSizeForAlarm(float MinDefectSizeForAlarm);

};


#endif //MULTI_CAMERA_CDEVICEPARAMS_H
