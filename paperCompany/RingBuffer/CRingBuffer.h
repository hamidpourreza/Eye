//
// Created by linux on 11/24/23.
//

#ifndef DOUBLEBUFFER_CRINGBUFFER_H
#define DOUBLEBUFFER_CRINGBUFFER_H

#include <bits/stdc++.h>

template<typename T, size_t N>
class CRingBuffer {
public:

    bool Insert(const T &item, bool isFinished);

    void ReadFinished();

    T *Extract() noexcept;

    bool IsFinished();
private:
    T mBuffer[N]{};
    std::atomic<bool> mIsFinished{};
    std::atomic<int32_t> mQueued{};
    std::atomic<uint64_t> mWriteIndex{};
    std::atomic<uint64_t> mReadIndex{};

};

template<typename T, size_t N>
bool CRingBuffer<T, N>::IsFinished() {
    return mIsFinished;
}

template<typename T, size_t N>
bool CRingBuffer<T, N>::Insert(const T &item, bool isFinished) {
    auto pos = mWriteIndex.load();
    mIsFinished = isFinished;
    if (mQueued >= N) {
//        pos = (pos + 1) % N;
        return false;
    }
//    std::cout << "Write Index: " << mWriteIndex << std::endl;

    mBuffer[pos] = item;
    mWriteIndex = (mWriteIndex + 1) % N;
    mQueued += 1;
//    mIsWrite[pos].store(true);
//    mIsRead[pos].store(true);
    return true;
}

template<typename T, size_t N>
T *CRingBuffer<T, N>::Extract() noexcept {
    auto pos = mReadIndex.load();
    if (!mQueued) {
        return nullptr;
    }
//    std::cout << "Read Index: " << mReadIndex << std::endl;

    mReadIndex = (mReadIndex.load() + 1) % N;

    return &mBuffer[pos];
}

template<typename T, size_t N>
void CRingBuffer<T, N>::ReadFinished() {
//    auto pos = mIndex.load() % N;
//    mIsRead[pos].store(false);
//    mIsWrite[pos].store(false);
    mQueued -= 1;
//    std::cout << "Queue Index: " << mQueued << std::endl;

}

#endif //DOUBLEBUFFER_CRINGBUFFER_H
