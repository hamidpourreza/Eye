//
// Created by mj on 06.07.23.
//

#include <thread>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include "Network/CNetwork.h"
#include "Image/CImage.h"
#include "Manager/Image/CImageManager.h"
#include "Manager/Detector/CDetectorManager.h"
#include <opencv2/opencv.hpp>
#include <CameraService.h>
#include "detectorGPU/Detector.h"
#include "Device/CDeviceStatus.h"
#include "Parameters/CDeviceParams.h"
#include <fstream>
#include <sockpp/tcp_acceptor.h>
#include "Device/CDevice.h"
#include "RingBuffer/CRingBuffer.h"
#include "constant/constant.h"
#include "IWriter/CWriter.h"
#include <mutex>
#include "colorSensor/uart.h"

#ifdef TEST
cv::Mat testRawImage;
cv::Mat testProcessImage;
#endif

using namespace std::chrono_literals;
const uint32_t MAX_COLOR_BUFFER = 200;
std::atomic<uint64_t> globalCounter = 0;
CRingBuffer<cv::Mat, 2> ringBuffer;
std::mutex gMutex;
std::mutex gColorMutex;
std::queue<bool> paperFinishedQueue;
std::queue<bool> faultPaperQueue;

std::unique_ptr<CDetectorManager> detectorManager{};
std::unique_ptr<CImageManager> processImageManager{};
std::unique_ptr<CImageManager> rawImageManager{};
std::unique_ptr<CImageManager> writerImageManger{};
int startOfRow = 0;
std::atomic<int32_t> gOffset{};
std::atomic<uint32_t> colorCurrentIndex{};

std::atomic<uint> imageIndex;
uint64_t writeIndex{};

cv::Mat restPadd;
cv::Mat inputImg;
cv::Mat gProcessImage[MAX_IMAGE_INDEX];
cv::Mat gRawImage[MAX_IMAGE_INDEX];
std::atomic<bool> gEndOfPaper{};

int gDebouncing = 0;
std::atomic<uint> batchNumber{};
CNetwork network;
auto device = CDevice::GetDevice();
auto deviceStatus = CDeviceStatus();
auto deviceParameter = CDeviceParams();

struct CustomData {
    std::unique_ptr<CameraService> cameraService{};
    std::unique_ptr<CImage> cImage{};
};

std::unique_ptr<CustomData> GCustomData;

struct SChunkData {
    double Min{};
    double Max{};
    double Len{};
    double Width{};
    std::atomic<uint32_t> R{};
    std::atomic<uint32_t> G{};
    std::atomic<uint32_t> B{};
};

SChunkData chunkData[MAX_COLOR_BUFFER]{}, tempChunkData;

Uart colorSensor;

cv::Mat topLayer;
bool isPaper = false;

std::map<std::string, std::function<std::string(const json &content, sockpp::tcp_socket &sock)>> handleRequestMap;

struct ReportStatus {
    static json TOJSON(CDeviceStatus &status) {
        json result{};
        result["camConnected"] = (int) status.isCamConnected();
        result["paperEntered"] = (int) status.isPaperEnter();
        result["processStarted"] = (int) status.isProcessStart();
        result["softwareStarted"] = (int) status.isSoftwareStart();
        result["totalCounter"] = status.totalCounter();
        return result;
    }
};

void notifyEjectorChanged() {
    auto ejectorStatus = 0;
    if (deviceParameter.AlarmNormallyHighLow()) {
        if (device->IsEjectorEnable()) {
            ejectorStatus = 0;
        } else {
            ejectorStatus = 1;
        }
    } else {
        if (device->IsEjectorEnable()) {
            ejectorStatus = 1;
        } else {
            ejectorStatus = 0;
        }
    }
    json j;
    j["Command"] = "ejectorStatus";
    j["Content"] = ejectorStatus;
    auto ret = network.connect("localhost", 40001);
    if (ret) {
        network.sendData(j.dump());
        network.close();
    }
};


void activateEjector() {
    while (true) {
        bool value{};
        {
            std::lock_guard<std::mutex> lock(gMutex);
            if (!paperFinishedQueue.empty()) {
                paperFinishedQueue.pop();
                value = true;

            }
        }

        if (value) {
            auto current = globalCounter.load();
            auto expected = current + (ulong) device->GetAlarmDelay();
            while (current < expected) {
                current = globalCounter.load();
                std::this_thread::sleep_for(1ms);
            }
//            std::this_thread::sleep_for(std::chrono::milliseconds(device->GetAlarmDelay()));
            if (faultPaperQueue.empty()) {
                spdlog::critical("Queue Empty");

                continue;
            }
            value = faultPaperQueue.front();
            faultPaperQueue.pop();

            if (value) {
                spdlog::critical("Ejector Enable");
                device->EnableEjector();
                notifyEjectorChanged();

                current = globalCounter.load();
                expected = current + (ulong) device->GetAlarmDuration();
                while (current < expected) {
                    current = globalCounter.load();
                    std::this_thread::sleep_for(1ms);
                }
                device->DisableEjector();
                spdlog::critical("Ejector Disable");
                notifyEjectorChanged();
            }

        } else {
            std::this_thread::sleep_for(50ms);
        }
    }
}

void sendDataToProxy(const json &j) {
    auto ret = network.connect("localhost", 40001);
    if (ret) {
        network.sendData(j.dump());
        network.close();
    };
}

void notifyProcessFinished() {
    json j;
    j["Command"] = "processFinished";
    j["Content"] = 1;
    sendDataToProxy(j);
}


ulong dateTimeNow() {
    std::time_t t = std::time(0);   // get time now
    std::tm *now = std::localtime(&t);
    ulong dateTime = 0;
    dateTime = (now->tm_year + 1900) * 10'000'000'000 +
               (now->tm_mon + 1) * 100'000'000 +
               (now->tm_mday) * 1'000'000 +
               (now->tm_hour) * 10'000 + now->tm_min * 100 +
               now->tm_sec;
    return dateTime;
}

void sendTotal(ulong total) {
    auto dateTime = dateTimeNow();
    json j;
    j["Command"] = "saveTotal";
    j["Content"] = {
            {"dateTime", dateTime},
            {"count",    total}
    };
    auto ret = network.connect("localhost", 40001);
    if (ret) {
        network.sendData(j.dump());
        network.close();
    }
}

void sendResult(float minSize, float maxSize, float area, float count, int r, int g, int b) {
    std::string sensorName = "SENSOR ONE";
    auto dateTime = dateTimeNow();
    json j;
    j["Command"] = "saveData";
    j["Content"] = {
            {"dateTime", dateTime},
            {"minSize",  minSize},
            {"maxSize",  maxSize},
            {"area",     area},
            {"count",    count},
            {"r",        r},
            {"g",        g},
            {"b",        b}
    };
    auto ret = network.connect("localhost", 40001);
    if (ret) {
        network.sendData(j.dump());
        network.close();
    }
}

std::string setCameraSettings() {
    auto error = "";
    if (deviceStatus.isCamConnected()) {
        GCustomData->cameraService->setMultiplier(deviceParameter.Multiplier());
        GCustomData->cameraService->setDivider(deviceParameter.Divider());
        GCustomData->cameraService->setExposureTime(deviceParameter.ExposureTime());
        return error;
    }
    return "failed to set camera settings";
}

std::string heartBit(const json &content, sockpp::tcp_socket &sock) {
    return "";
}

std::string applySettings(const json &content, sockpp::tcp_socket &sock) {

    deviceParameter.Fill(content["Content"].dump());
    auto ret = setCameraSettings();
    EJECTOR_STATUS status;
    if (deviceParameter.AlarmNormallyHighLow()) {
        status = EJECTOR_STATUS::HIGH;
    } else {
        status = EJECTOR_STATUS::LOW;
    }
    device->SetEjectorDefaultStatus(status);
    device->SetAlarmDuration(deviceParameter.AlarmDuration());
    device->SetAlarmDelay(deviceParameter.AlarmDelay());
    return ret;
}

std::string loadSettings(const json &content, sockpp::tcp_socket &sock) {

    return "";
}

std::string getSensorStatus(const json &content, sockpp::tcp_socket &sock) {
    auto ret = ReportStatus::TOJSON(deviceStatus);
    spdlog::info("{} {}", __func__, ret.dump(4));
    sock.write(ret.dump());
    return "";
}

//TODO check for last '/' in the route
std::string getEjectorStatus(const json &content, sockpp::tcp_socket &sock) {
    json j;
    auto ejectorStatus = 0;
    if (deviceParameter.AlarmNormallyHighLow()) {
        if (device->IsEjectorEnable()) {
            ejectorStatus = 0;
        } else {
            ejectorStatus = 1;
        }
    } else {
        if (device->IsEjectorEnable()) {
            ejectorStatus = 1;
        } else {
            ejectorStatus = 0;
        }
    }
    j["ejectorStatus"] = ejectorStatus;
    std::cout << "Ejector Status: " << ejectorStatus << std::endl;
//    j["default"] = deviceParameter.AlarmNormallyHighLow();
    sock.write(j.dump());
    return "";
}

void sendChunk(sockpp::tcp_socket &socket, char *chunk, size_t chunkSize) {
    size_t remain = chunkSize;
    size_t offset = 0;
    int len;
    while ((remain > 0) && ((len = socket.write_n(chunk + offset, remain)) > 0)) {
        remain -= len;
        offset += len;
    }
}

std::string getRawImage(const json &content, sockpp::tcp_socket &sock) {
//    spdlog::info()
#ifdef TEST
    if (testRawImage.empty()) {
//        testRawImage = cv::imread("/home/linux/Pictures/Big_&_Small_Pumkins.JPG");
    }
    testRawImage = cv::imread("./result/1.png");

#else
    auto testRawImage = rawImageManager->Last().clone();
#endif

    cv::resize(testRawImage, testRawImage, cv::Size(), 0.125, 0.125, cv::INTER_NEAREST);
    std::vector<uchar> PNG;
    cv::imencode(".PNG", testRawImage, PNG);
    auto chunkSize = PNG.size();
    json j;
    j["type"] = "testRawImage";
    j["size"] = chunkSize;
    sock.write(j.dump());
    std::array<char, 2048> buffer{};
    sock.read(buffer.data(), 2048);
    sendChunk(sock, (char *) PNG.data(), chunkSize);
    return "";
}

std::string getChunkData(const json &content, sockpp::tcp_socket &sock) {
    json j;
    std::string len(' ', 8);
    std::string width(' ', 8);
    std::string min(' ', 8);
    std::string max(' ', 8);

    sprintf(len.data(), "%0.2f", tempChunkData.Len);
    sprintf(width.data(), "%0.2f", tempChunkData.Width);
    sprintf(min.data(), "%0.2f", tempChunkData.Min);
    sprintf(max.data(), "%0.2f", tempChunkData.Max);

    j["dim"] = {
            {"L", len + " inch"},
            {"W", width + " inch"}
    };

    j["scoreMap"] = "Min:" + min + ", Max:" + max;

    j["color"] = {
            {"R", tempChunkData.R.load()},
            {"G", tempChunkData.G.load()},
            {"B", tempChunkData.B.load()}
    };
    spdlog::critical("R: {}, G: {}, B: {}",
                     tempChunkData.R.load(),
                     tempChunkData.G.load(),
                     tempChunkData.B.load());

    sendChunk(sock, (char *) j.dump().c_str(), j.dump().length());
    return "";
}

std::string getProcessedImage(const json &content, sockpp::tcp_socket &sock) {
#ifdef TEST
    if (testProcessImage.empty()) {
    }
        testProcessImage = cv::imread("./result/Proc2.png");

#else
    auto testProcessImage = processImageManager->Last().clone();
#endif
    std::vector<uchar> PNG;
    cv::imencode(".PNG", testProcessImage, PNG);
    auto chunk_size = PNG.size();

    json j;
    j["type"] = "testProcessImage";
    j["size"] = chunk_size;
    sock.write(j.dump());
    std::array<char, 2048> buffer{};
    sock.read(buffer.data(), 2048);
    sendChunk(sock, (char *) PNG.data(), chunk_size);
    return "";
}

void handleRequest(sockpp::tcp_socket sock) {
    ssize_t n;
    char buf[2048]{};
    n = sock.read(buf, sizeof(buf));
    try {
        static uint counter = 0;
        spdlog::info("Request Number: {}", ++counter);
        json j = json::parse(std::string(buf));
        auto command = j["Command"];
        handleRequestMap[command](j, sock);
        spdlog::info("Request: {}", command);
    } catch (...) {
        sock.close();
    }
    sock.close();
}

void runTcpServer() {
    sockpp::tcp_acceptor acc(8181);

    if (!acc) {
        std::cerr << "Error creating the acceptor: " << acc.last_error_str() << std::endl;
        return;
    }

    while (true) {
        sockpp::inet_address peer;

        // Accept a new client connection
        sockpp::tcp_socket sock = acc.accept(&peer);
        spdlog::warn("Received a connection request from {}", peer.address());

        if (!sock) {
            std::cerr << "Error accepting incoming connection: "
                      << acc.last_error_str() << std::endl;
        } else {
            // Create a thread and transfer the new stream to it.
            handleRequest(std::move(sock));
        }
    }
}

void writer() {
#ifdef DEBUG
    while (true) {
        auto size = writerImageManger->CurrentLength();
        if (size > 0) {
            spdlog::info("Image Write");
            auto image = writerImageManger->Front();
            CWriter::Write(image, std::string_view());
        }
        std::this_thread::sleep_for(5ms);
    }
#endif
}

void writeCountToFile() {
    std::fstream file;
    file.open("./totalCounter.txt", std::ios::out | std::ios::trunc);
    file << deviceStatus.totalCounter();
}

void processThread() {
    while (true) {
        auto img = ringBuffer.Extract();
        if (!img) {
            std::this_thread::sleep_for(1ms);
            continue;
        }
        auto isPaperFinished = gEndOfPaper.load();
        auto index = imageIndex.load();
        if (isPaperFinished) {
            imageIndex = (imageIndex + 1) % MAX_IMAGE_INDEX;
            gEndOfPaper = false;
        }
        cv::Mat res;

        cv::Mat temp;
        cv::cvtColor(*img, temp, cv::COLOR_GRAY2RGB);

        auto dt = detectorManager->GetFreeDetector();
        res = dt->mDetector->inference(temp);
        ringBuffer.ReadFinished();

        auto resizedFrame = img->rowRange(padd, img->rows - padd);

        gRawImage[index].push_back(resizedFrame);
        gProcessImage[index].push_back(res);
        detectorManager->resetDetector(dt);

        if (isPaperFinished) {
            uint64_t localImageIndex = writeIndex;
            writeIndex = (writeIndex +1) % 200;
            std::thread([index, dt, localImageIndex] {
                auto tempRaw = gRawImage[index].clone();
                auto tempProcess = gProcessImage[index].clone();
                gRawImage[index].release();
                gProcessImage[index].release();

#ifdef DEBUG
                writerImageManger->Push(tempRaw.clone());
#endif
                auto t1 = std::chrono::high_resolution_clock::now();

                auto start = 0;
                int end;
                if (gOffset - padd > 0) {
                    end = tempProcess.rows - abs(gOffset - padd);

                } else {
                    end = tempProcess.rows;
                }
                tempProcess = tempProcess.rowRange(start, end);
                end = tempRaw.rows - gOffset;
                if (end < 0) {
                    end = tempRaw.rows;
                }
                cv::Mat forwrite;
                tempRaw.copyTo(forwrite);
                tempRaw = tempRaw.rowRange(start, end);

                cv::Mat processforWrite;
                auto min = 0.0, max = 0.0;
                cv::minMaxIdx(tempProcess, &min, &max);
                tempChunkData.Min = min;
                tempChunkData.Max = max;
                tempRaw = tempRaw.rowRange(padd, tempRaw.rows - padd);
                tempProcess = tempProcess.rowRange(padd, tempProcess.rows - padd);

                cv::normalize(tempProcess, processforWrite, 0, 255, cv::NORM_MINMAX,
                              CV_8UC1);

                cv::threshold(tempProcess, tempProcess, deviceParameter.DefectStrengthFilter(), 1.0,
                              cv::THRESH_BINARY);
                cv::normalize(tempProcess, tempProcess, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::cvtColor(tempProcess, tempProcess, cv::COLOR_GRAY2BGR);

                std::pair<cv::Mat, cv::Mat> result;
                result = dt->mDetector->postProcess(tempProcess,
                                                    deviceParameter.DefectSizeFilter(),
                                                    deviceParameter.MinDefectSizeForAlarm());

                cv::Mat secondMat, maskMat;
                cv::cvtColor(result.first, secondMat, cv::COLOR_BGR2GRAY);

                auto ret = cv::countNonZero(secondMat);


                if (ret) {
                    faultPaperQueue.push(true);
                    auto name = dateTimeNow();
                    auto report = dt->mDetector->report(result.first);
                    sendResult(report["minArea"], report["maxArea"],
                               report["area"], report["count"],
                               tempChunkData.R, tempChunkData.G, tempChunkData.B);
                }
                auto counter = deviceStatus.totalCounter() + 1;
                deviceStatus.setTotalCounter(counter);
                if (counter % 10 == 0) {
                    sendTotal(10);
                    writeCountToFile();
                }

                cv::resize(processforWrite, processforWrite, cv::Size(), 1,
                           deviceParameter.scaleX() / deviceParameter.scaleY(), cv::INTER_NEAREST);

                processImageManager->Push(processforWrite);
                std::string filename = std::to_string(localImageIndex) + "-processImage";
                CWriter::Write(processforWrite, filename);

                if (deviceParameter.ImageContrastEnhancement()) {
                    tempRaw = dt->mDetector->enhanced(tempRaw);
                } else {
                    cv::cvtColor(tempRaw, tempRaw, cv::COLOR_GRAY2BGR);
                }
                cv::Mat mask;
                cv::resize(result.second, result.second, cv::Size(tempRaw.cols, tempRaw.rows), 0, 0, cv::INTER_NEAREST);
                std::pair<float, float> sizes = dt->mDetector->calcSize(tempRaw,
                                                                        deviceParameter.scaleX(),
                                                                        deviceParameter.scaleY());

                cv::cvtColor(result.second, mask, cv::COLOR_RGB2GRAY);
                result.second.copyTo(tempRaw, mask);

                cv::resize(tempRaw, tempRaw, cv::Size(), 1,
                           deviceParameter.scaleX() / deviceParameter.scaleY(),
                           cv::INTER_NEAREST);

                rawImageManager->Push(tempRaw);
                filename = std::to_string(localImageIndex) + "-rawImage";


                std::cout << "Size = " << sizes.first << " " << sizes.second << std::endl;
                tempChunkData.Width = std::get<0>(sizes);
                tempChunkData.Len = std::get<1>(sizes);
                spdlog::critical("Width: {}, Len: {}", tempChunkData.Width, tempChunkData.Len);

                auto t2 = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                std::cout << "Thread Processing Time:" << elapsed << std::endl;

                notifyProcessFinished();
                CWriter::Write(tempRaw, filename);

#ifdef DEBUG
                writerImageManger->Push(tempRaw);
#endif
            }).detach();
        }
    }
}

#ifdef GPU

void GrabFrameCallback(IMV_Frame *imvFrame, void *value) {
    auto now = std::chrono::high_resolution_clock::now();
    auto customData = (CustomData *) value;
    auto height = imvFrame->frameInfo.height;
    auto width = imvFrame->frameInfo.width;
    auto num = std::get<0>(customData->cameraService->GetNumOfLostPacket());
    if (num) {
        std::cerr << "Frame Lost" << std::endl;
    }
//    std::cout << "Frame received" << std::endl;

    auto frameData = customData->cameraService->getConvertedFrame(imvFrame, false);
    if (!frameData) {
        std::cerr << "Empty frame." << std::endl;
        return;
    }
    globalCounter.fetch_add(1);
    auto subImage = cv::Mat(height, width, CV_8UC1, frameData);

    auto brightPointLeft = deviceParameter.BSBPLeft();
    auto brightPointRight = deviceParameter.BSBPRight();
    auto darkPointCenter = deviceParameter.BSDPCenter();

    auto brightPointLeftValue = deviceParameter.BSBPVLeft();
    auto brightPointRightValue = deviceParameter.BSBPVRight();
    auto darkPointCenterValue = deviceParameter.BSDPVCenter();

    auto enterDebouncing = deviceParameter.enterDebouncing();
    auto exitDebouncing = deviceParameter.exitDebouncing();

    int found = -1;
    int lastCol = 0;
    if (!isPaper) {
        for (int col = 0; col < subImage.rows - 1; ++col) {
            if (!(cv::mean(subImage(cv::Range(col, col + 1), cv::Range(brightPointLeft, brightPointLeft + 50))).val[0] >
                  brightPointLeftValue &&
                  cv::mean(subImage(cv::Range(col, col + 1),
                                    cv::Range(brightPointRight, brightPointRight + 50))).val[0] >
                  brightPointRightValue &&
                  cv::mean(
                          subImage(cv::Range(col, col + 1), cv::Range(darkPointCenter, darkPointCenter + 250))).val[0] <
                  darkPointCenterValue)) {
                gDebouncing += 1;
                lastCol = col;
                if (gDebouncing == enterDebouncing) {
                    found = col;
                    static int paperSheet = 0;
                    spdlog::critical("PAPER SHEET {}", ++paperSheet);
                    isPaper = !isPaper;
                    deviceStatus.setPaperEnter(true);
                    break;
                }

            } else {
                gDebouncing = 0;
            }
        }
    } else {
        for (int col = 0; col < subImage.rows - 1; ++col) {
            if (cv::mean(subImage(cv::Range(col, col + 1), cv::Range(brightPointLeft, brightPointLeft + 50))).val[0] >
                brightPointLeftValue &&
                cv::mean(subImage(cv::Range(col, col + 1), cv::Range(brightPointRight, brightPointRight + 50))).val[0] >
                brightPointRightValue &&
                cv::mean(subImage(cv::Range(col, col + 1), cv::Range(darkPointCenter, darkPointCenter + 250))).val[0] <
                darkPointCenterValue) {
                gDebouncing -= 1;
                lastCol = col;

                if (gDebouncing == 0) {
                    found = col;
                    isPaper = !isPaper;
                    gDebouncing = 0;
                    deviceStatus.setPaperEnter(false);
                    spdlog::critical("PAPER EXIT");
                    deviceStatus.setProcessStart(false);

                    break;
                }

            } else {
                gDebouncing = exitDebouncing;
            }
        }
    }

    if (found == -1 && !isPaper) {
        return;
    }

    auto start = 0, end = 0;
    start = 0;
    end = height;
    if (found == -1 && isPaper) {
        start = 0;
        end = height;
    } else if (found != -1 && isPaper) {
        start = found;
        end = height;
    } else if (found != -1 && !isPaper) {
        start = 0;
        end = (found == 0) ? 1 : found;
    }
    customData->cImage->InsertRows(frameData, height, start, end);
    startOfRow += static_cast<int>(end - start);
//    auto processStatus = deviceStatus.isProcessStart();
//    if (!processStatus) {
//        deviceStatus.setProcessStart(true);
//    }
    int offsetSize = 0;
    auto condition = (startOfRow - NUM_OF_ROWS) < 0;
    auto isPaperFinished = (found != -1 && !isPaper);
    if (isPaperFinished && condition) {
        gOffset = NUM_OF_ROWS - startOfRow;
        offsetSize = gOffset;
        cv::Mat rest = cv::Mat::ones(gOffset, width, CV_8UC1);

        customData->cImage->InsertRows(rest.data, rest.rows, 0, rest.rows);
        startOfRow += static_cast<int>(rest.rows);
    }

    if (startOfRow >= NUM_OF_ROWS) {
        batchNumber.fetch_add(1);
        if (batchNumber < MAX_BATCH && isPaperFinished) {
            startOfRow = 0;
            batchNumber.store(0);
            customData->cImage->ResetCurrentRow();
            auto index = imageIndex.load();
            gProcessImage[index].release();
            gRawImage[index].release();
            spdlog::critical("Image Invalid");
            return;
        }
        spdlog::critical("Batch number {}", batchNumber.load());

        cv::Mat rest;
        int last = startOfRow;
        auto testRawImage = customData->cImage->ImageMat().clone();


        inputImg = cv::Mat(NUM_OF_ROWS + topLayer.rows, testRawImage.cols, CV_8UC1);
        testRawImage.rowRange(0, NUM_OF_ROWS).copyTo(inputImg.rowRange(padd, inputImg.rows));
        if (batchNumber.load() == 1) {
            testRawImage.rowRange(0, 32).copyTo(topLayer);
            cv::flip(topLayer, topLayer, 0);
        }
        topLayer.copyTo(inputImg.rowRange(0, padd));
        testRawImage.rowRange((NUM_OF_ROWS) - 2 * padd, NUM_OF_ROWS - padd).copyTo(topLayer);
        customData->cImage->ResetCurrentRow();
        customData->cImage->InsertRows(testRawImage.data, NUM_OF_ROWS, NUM_OF_ROWS - padd, NUM_OF_ROWS);
        startOfRow = padd;


        if (last > NUM_OF_ROWS) {
            startOfRow += last - NUM_OF_ROWS;
            auto temp = cv::Mat(cv::Size(width, startOfRow), CV_8UC1);
            testRawImage.rowRange(NUM_OF_ROWS, last).copyTo(temp);
            customData->cImage->InsertRows(temp.data, temp.rows, 0, temp.rows);
        }
        cv::Mat padded;
        if (isPaperFinished && condition && offsetSize != 0) {
            std::cout << " offsetsize " << offsetSize << std::endl;
            offsetSize += 2 * padd;
            cv::Mat part = inputImg.rowRange(0, padd + NUM_OF_ROWS - offsetSize).clone();
            int partcount = (576 / part.rows) + 1;
            cv::Mat flippedPart;
            cv::flip(part, flippedPart, 0);
            for (int i = 0; i < partcount; i++) {
                if (i % 2 == 0) {
                    padded.push_back(part);
                } else {
                    padded.push_back(flippedPart);
                }
            }

            padded.rowRange(0, inputImg.rows).copyTo(inputImg);
        }

        inputImg.copyTo(restPadd);
#ifdef DEBUG
        static cv::Mat raw;
        raw.push_back(inputImg.rowRange(padd, 576 - padd));
#endif
        auto ret = ringBuffer.Insert(inputImg, isPaperFinished);
        if (!ret) {
            spdlog::critical("System is not real-time");
            deviceStatus.setProcessStart(true);
            std::this_thread::sleep_for(5ms);
            deviceStatus.setProcessStart(false);

        }
        if (isPaperFinished) {
            paperFinishedQueue.push(false);
            offsetSize = 0;
            batchNumber.store(0);
            gEndOfPaper = true;
            customData->cImage->ResetCurrentRow();
            startOfRow = 0;
            topLayer = cv::Mat(32, width, CV_8UC1, cv::Scalar(0, 0, 0));
//            std::this_thread::sleep_for(50ms);
            size_t offsetOne = 50;
            const uint32_t MAX_COLOR_OFFSET = 10;
            uint32_t r{}, g{}, b{};
            {
                std::lock_guard<std::mutex> lockGuard(gColorMutex);
                size_t startColorIndex = MAX_COLOR_BUFFER + colorCurrentIndex - offsetOne;
                size_t endColorIndex = startColorIndex + MAX_COLOR_OFFSET;
                for (size_t idx = startColorIndex; idx < endColorIndex; idx++) {
                    size_t index = idx % MAX_COLOR_BUFFER;
                    r += chunkData[index].R;
                    g += chunkData[index].G;
                    b += chunkData[index].B;
                }
            }

            tempChunkData.R.store(r / MAX_COLOR_OFFSET);
            tempChunkData.G.store(g / MAX_COLOR_OFFSET);
            tempChunkData.B.store(b / MAX_COLOR_OFFSET);
            spdlog::critical("temp R {}, G {}, B {}", tempChunkData.R, tempChunkData.G, tempChunkData.B);

#ifdef DEBUG
            static auto counter = 0;
            auto filename = "./result/raw-image-" + std::to_string(counter++) + ".jpg";
            writerImageManger->Push(raw);
            raw.release();
#endif

        }
    }
}

#endif

void onCameraDisconnect(IMV_DeviceInfo info, CameraService *service) {
    deviceStatus.setCamConnect(false);
    deviceStatus.setPaperEnter(false);
    deviceStatus.setProcessStart(false);
    service->stopGrabbing();
//    service->close();
}

void onCameraConnect(IMV_DeviceInfo info, CameraService *service) {
    spdlog::critical(__FUNCTION__);
//    auto ret = service->open();
//    if (ret) {
    service->close();
    service->open();
    service->set_max_width();
    service->set_height(32);
    service->set_acquisition_mode_continuous();
    setCameraSettings();
    service->allocateMem(false);
    service->set_on_disconnect_callback(onCameraDisconnect);
    service->set_on_connect_callback(onCameraConnect);
    service->set_grab_callback(GrabFrameCallback,
                               GCustomData.get());
    GCustomData->cameraService->subscribe_connect_arg();
    service->startGrabbing();
    deviceStatus.setCamConnect(true);
//    }
}

void openCamera() {
    auto num = CameraService::numFoundDevices();
    std::vector<std::string> cameraIPS;


    GCustomData = std::make_unique<CustomData>();
    GCustomData->cameraService = std::make_unique<CameraService>();
    while (true) {
        cameraIPS = CameraService::getDevicesList();
        if (!cameraIPS.empty()) {
            std::cout << "Num of Found camera: " << std::get<0>(num) << std::endl;
            GCustomData->cameraService->createHandle(cameraIPS[0]);

            GCustomData->cameraService->open();
            if (GCustomData->cameraService->isOpen()) {
                std::cout << "Camera opened." << std::endl;
                break;
            }
        }

        std::this_thread::sleep_for(1s);
    }
    deviceStatus.setCamConnect(true);

    auto width = std::get<0>(GCustomData->cameraService->getWidth());
    std::cout << "WIDTH: " << width << std::endl;
    topLayer = cv::Mat(32, width, CV_8UC1, cv::Scalar(0, 0, 0));
    auto colorSpace = std::get<0>(GCustomData->cameraService->get_pixel_format());
    GCustomData->cImage = std::make_unique<CImage>(NUM_OF_ROWS, width);
    GCustomData->cameraService->set_max_width();
    GCustomData->cameraService->set_height(32);
    GCustomData->cameraService->set_acquisition_mode_continuous();
    setCameraSettings();
    GCustomData->cameraService->allocateMem(false);
    GCustomData->cameraService->set_on_disconnect_callback(onCameraDisconnect);
    GCustomData->cameraService->set_on_connect_callback(onCameraConnect);
    GCustomData->cameraService->set_grab_callback(GrabFrameCallback,
                                                  GCustomData.get());
    GCustomData->cameraService->subscribe_connect_arg();
    GCustomData->cameraService->startGrabbing();

}

void getParameters() {
    auto ret = false;
    while (!ret) {
        ret = network.connect("localhost", 40001);
        json j;
        j["Command"] = "loadLastSettings";
        if (ret) {
            network.sendData(j.dump());
            auto read = network.readData();
            json content;
            try {
                content = json::parse(read);
            } catch (...) {
                network.close();
                continue;
            }
            deviceParameter.Fill(content.dump());
            network.close();

            device->SetAlarmDuration(deviceParameter.AlarmDuration());
            device->SetAlarmDelay(deviceParameter.AlarmDelay());

            EJECTOR_STATUS status;
            if (deviceParameter.AlarmNormallyHighLow()) {
                status = EJECTOR_STATUS::HIGH;
            } else {
                status = EJECTOR_STATUS::LOW;
            }
            device->SetEjectorDefaultStatus(status);
        }
        std::this_thread::sleep_for(1s);
    }
}

void onNotifyDeviceStatusChange(std::string_view name, int status) {
    json j;
    j["Command"] = std::string{name};
    j["Content"] = status;

    sendDataToProxy(j);
}

void openCounterFile() {
    std::fstream file;
    file.open("./totalCounter.txt", std::ios::in | std::ios::out);
    if (!file.is_open()) {
        spdlog::critical("Failed to open file.");
    }
    auto count = 0;
    file >> count;
    deviceStatus.setTotalCounter(count);
}

void colorSensorThread() {
    while (true) {
        auto ret = colorSensor.readUart();
        if (ret) {
            auto rgb = colorSensor.RGB;
            spdlog::critical("part R: {}, G: {}, B: {}",
                             rgb[0],
                             rgb[1],
                             rgb[2]);
            std::lock_guard<std::mutex> lockGuard(gColorMutex);
            chunkData[colorCurrentIndex].R = rgb[0];
            chunkData[colorCurrentIndex].G = rgb[1];
            chunkData[colorCurrentIndex].B = rgb[2];
            colorCurrentIndex = (colorCurrentIndex + 1) % MAX_COLOR_BUFFER;
        }
        std::this_thread::sleep_for(5ms);
    }
}

int main() {

    sockpp::initialize();
    std::cout << "Version: " << 1.0 << std::endl;
    spdlog::set_level(spdlog::level::critical);
    handleRequestMap["applySettings"] = applySettings;
    handleRequestMap["getSensorStatus"] = getSensorStatus;
    handleRequestMap["loadSettings"] = loadSettings;
    handleRequestMap["getEjectorStatus"] = getEjectorStatus;
    handleRequestMap["getRawImage"] = getRawImage;
    handleRequestMap["getProcessedImage"] = getProcessedImage;
    handleRequestMap["getChunkData"] = getChunkData;
    handleRequestMap["heartBit"] = heartBit;

    CDeviceStatus::addNotifyCallback(onNotifyDeviceStatusChange);
    deviceStatus.setSoftwareStatus(true);
    openCounterFile();

    detectorManager = std::make_unique<CDetectorManager>(NUM_OF_DETECTOR);
    detectorManager->InitProcess();
    processImageManager = std::make_unique<CImageManager>(3);
    rawImageManager = std::make_unique<CImageManager>(3);
#ifdef DEBUG
    writerImageManger = std::make_unique<CImageManager>(10);
#endif

    std::thread writerThread(writer);
    writerThread.detach();

    std::thread tcpThread(runTcpServer);
    tcpThread.detach();

    std::thread trd(activateEjector);
    trd.detach();

    std::thread trdProcess(processThread);
    trdProcess.detach();

    std::thread colorSensorTrd(colorSensorThread);
    colorSensorTrd.detach();

    getParameters();
//    freopen("/dev/null", "w", stdout);
    openCamera();


    while (true) {
        std::this_thread::sleep_for(5ms);
    }
}

