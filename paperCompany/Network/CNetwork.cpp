//
// Created by veerayco on 10/4/23.
//

#include "CNetwork.h"

CNetwork::CNetwork() {
    sockpp::initialize();
}

bool CNetwork::connect(std::string_view HOST, uint32_t PORT) {
    std::lock_guard<std::mutex> guard(mMutex);
    sockpp::inet_address address(HOST.data(), PORT);
    auto ret = conn.connect(address, 5s);
    return ret;
}

long CNetwork::sendData(std::string_view data) {
    std::lock_guard<std::mutex> guard(mMutex);
    auto ret = conn.write(data.data());
    return ret;
}

bool CNetwork::close() {
    std::lock_guard<std::mutex> guard(mMutex);
    auto ret = conn.close();
    return ret;
}

std::string CNetwork::readData() {
    std::lock_guard<std::mutex> guard(mMutex);
    std::string data(2048, '\0');
    auto ret = conn.read(data.data(), 2048);
    return data;
}
