//
// Created by veerayco on 10/21/23.
//

#include "CDeviceParams.h"

float CDeviceParams::ExposureTime() const {
    return mExposureTime;
}

void CDeviceParams::ExposureTime(float ExposureTime) {
    mExposureTime = ExposureTime;
}

float CDeviceParams::Divider() const {
    return mDivider;
}

void CDeviceParams::Divider(float Divider) {
    mDivider = Divider;
}

float CDeviceParams::Multiplier() const {
    return mMultiplier;
}

void CDeviceParams::Multiplier(float Multiplier) {
    mMultiplier = Multiplier;
}

float CDeviceParams::AlarmDelay() const {
    return mAlarmDelay;
}

void CDeviceParams::AlarmDelay(float AlarmDelay) {
    mAlarmDelay = AlarmDelay;
}

float CDeviceParams::AlarmDuration() const {
    return mAlarmDuration;
}

void CDeviceParams::AlarmDuration(float AlarmDuration) {
    mAlarmDuration = AlarmDuration;
}

bool CDeviceParams::AlarmNormallyHighLow() const {
    return mAlarmNormallyHighLow;
}

void CDeviceParams::AlarmNormallyHighLow(bool AlarmNormallyHighLow) {
    mAlarmNormallyHighLow = AlarmNormallyHighLow;
}

bool CDeviceParams::ImageContrastEnhancement() const {
    return mImageContrastEnhancement;
}

void CDeviceParams::ImageContrastEnhancement(bool ImageContrastEnhancement) {
    mImageContrastEnhancement = ImageContrastEnhancement;
}

float CDeviceParams::DefectSizeFilter() const {
    return mDefectSizeFilter;
}

void CDeviceParams::DefectSizeFilter(float DefectSizeFilter) {
    mDefectSizeFilter = DefectSizeFilter;
}

float CDeviceParams::DefectStrengthFilter() const {
    return mDefectStrengthFilter;
}

void CDeviceParams::DefectStrengthFilter(float DefectStrengthFilter) {
    mDefectStrengthFilter = DefectStrengthFilter;
}

float CDeviceParams::MinDefectSizeForAlarm() const {
    return mMinDefectSizeForAlarm;
}

void CDeviceParams::MinDefectSizeForAlarm(float MinDefectSizeForAlarm) {
    mMinDefectSizeForAlarm = MinDefectSizeForAlarm;
}

void CDeviceParams::Fill(std::string_view input) {
    auto j = json::parse(input);

#ifdef DEBUG
    std::cout << "Recv Json: " << j.dump(4) << std::endl;
#endif
    mExposureTime = j["exposureTime"];
    mDivider = j["divider"];
    mMultiplier = j["multiplier"];
    mAlarmDelay = j["alarmDelay"];
    mAlarmDuration = j["alarmDuration"];
    mAlarmNormallyHighLow = static_cast<bool>(j["alarmNormallyHighLow"]);
    mImageContrastEnhancement = static_cast<bool>(j["imageContrastEnhancement"]);
    mBSBPLeft = j["BSBPLeft"];
    mBSBPVLeft = j["BSBPVLeft"];
    mBSBPRight = j["BSBPRight"];
    mBSBPVRight = j["BSBPVRight"];
    mBSDPCenter = j["BSDPCenter"];
    mBSDPVCenter = j["BSDPVCenter"];
    mEnterDebouncing = j["enterDebouncing"];
    mExitDebouncing = j["exitDebouncing"];
    mDefectSizeFilter = j["defectSizeFilter"];
    mDefectStrengthFilter = j["defectStrengthFilter"];
    mMinDefectSizeForAlarm = j["minDefectSizeForAlarm"];
    mScaleX = j["scaleX"];
    mScaleY = j["scaleY"];

}

float CDeviceParams::scaleX() const {
    return mScaleX;
}

void CDeviceParams::setScaleX(float scaleX) {
    CDeviceParams::mScaleX = scaleX;
}

float CDeviceParams::scaleY() const {
    return mScaleY;
}

void CDeviceParams::setScaleY(float scaleY) {
    CDeviceParams::mScaleY = scaleY;
}

float CDeviceParams::BSBPLeft() const {
    return mBSBPLeft;
}

void CDeviceParams::setBSBPLeft(float mBsbpLeft) {
    mBSBPLeft = mBsbpLeft;
}

float CDeviceParams::BSBPVLeft() const {
    return mBSBPVLeft;
}

void CDeviceParams::setBSBPVLeft(float mBsbpvLeft) {
    mBSBPVLeft = mBsbpvLeft;
}

float CDeviceParams::BSBPRight() const {
    return mBSBPRight;
}

void CDeviceParams::setBSBPRight(float mBsbpRight) {
    mBSBPRight = mBsbpRight;
}

float CDeviceParams::BSBPVRight() const {
    return mBSBPVRight;
}

void CDeviceParams::setBSBPVRight(float mBsbpvRight) {
    mBSBPVRight = mBsbpvRight;
}

float CDeviceParams::BSDPCenter() const {
    return mBSDPCenter;
}

void CDeviceParams::setBSDPCenter(float mBsdpCenter) {
    mBSDPCenter = mBsdpCenter;
}

float CDeviceParams::BSDPVCenter() const {
    return mBSDPVCenter;
}

void CDeviceParams::setBSDPVCenter(float mBsdpvCenter) {
    mBSDPVCenter = mBsdpvCenter;
}

float CDeviceParams::enterDebouncing() const {
    return mEnterDebouncing;
}

void CDeviceParams::setEnterDebouncing(float enterDebouncing) {
    mEnterDebouncing = enterDebouncing;
}

float CDeviceParams::exitDebouncing() const {
    return mExitDebouncing;
}

void CDeviceParams::setExitDebouncing(float exitDebouncing) {
    mExitDebouncing = exitDebouncing;
}
