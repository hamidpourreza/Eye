//
// Created by veerayco on 10/4/23.
//

#ifndef MULTI_CAMERA_CNETWORK_H
#define MULTI_CAMERA_CNETWORK_H

#include <bits/stdc++.h>
#include <sockpp/tcp_connector.h>

using namespace std::chrono_literals;

class CNetwork {
//    inline static const uint PORT = 40001;
//    inline static const char *HOST = "localhost";
    std::mutex mMutex{};
    sockpp::tcp_connector conn;
public:
    CNetwork();

    bool connect(std::string_view HOST, uint32_t PORT);

    long sendData(std::string_view data);

    std::string readData();

    bool close();

};


#endif //MULTI_CAMERA_CNETWORK_H
