//

// Created by MSI on 8/24/2023.

//



#include "Detector.h"



Detector::Detector(const std::string& modelPath): engine(options) {

    auto start = std::chrono::high_resolution_clock::now();

    if (!Util::doesFileExist(modelPath)) {

        std::cout << "Error: Unable to find file at path: " << modelPath << std::endl;

        exit(-1);

    }



    bool succ =engine.build(modelPath,subVals,divVals,normalize);



    if (!succ) {

        throw std::runtime_error("Unable to build TRT engine.");

    }



    // Load the TensorRT engine file from disk

    succ = engine.loadNetwork();

    if (!succ) {

        throw std::runtime_error("Unable to load TRT engine.");

    }

    cv::cuda::GpuMat img(576,576,CV_8UC3);



    std::vector<std::vector<cv::cuda::GpuMat>> infBatch;

    std::vector<std::vector<cv::Mat>> featureVectors;

    std::vector<cv::cuda::GpuMat> batch;

    batch.emplace_back(img);

infBatch.emplace_back(batch);

for(int i=0;i<5;i++){

    engine.runInference(infBatch,featureVectors);

}

mTitleGridSize = cv::Size(16, 16);

    mClahe = cv::createCLAHE(mClipLimit, mTitleGridSize);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);



    std::cout << "Init Complete... time : " << duration.count() << " ms" << std::endl;

}



cv::Mat Detector::inference(cv::Mat cpuImg) {

	int crop=64;

    cv::Mat rightThresh, leftThresh,forThresh,thresh;

    cv::cvtColor(cpuImg,forThresh,cv::COLOR_BGR2GRAY);

    cv::threshold(forThresh,thresh,230,255,cv::THRESH_BINARY_INV);



    std::vector<std::vector<cv::Point>> contours;

    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);



    cv::Mat forshow;

    cv::cvtColor(thresh,forshow, cv::COLOR_GRAY2BGR);

    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {

        return cv::contourArea(a) > cv::contourArea(b);

    });



    cv::drawContours(forshow, contours, 0, cv::Scalar(0, 0, 255), 20);



    cv::Mat mask = cv::Mat::zeros(forThresh.size(), forThresh.type());

    cv::drawContours(mask, contours, 0, cv::Scalar(255), -1);

    cv::Mat reconstructed=cv::Mat::zeros(mask.size(),cpuImg.type());



    auto start = std::chrono::high_resolution_clock::now();



        // Find the indices of non-zero elements in the row

        cv::Mat nonZeroIndices,rowRes;

        cv::findNonZero(mask, nonZeroIndices);

    int minIdx ;

    int maxIdx;

        if (!nonZeroIndices.empty()) {

            // Get the minimum and maximum non-zero indices in the row

            minIdx = nonZeroIndices.at<cv::Point>(0).x;

            maxIdx = nonZeroIndices.at<cv::Point>(nonZeroIndices.rows - 1).x;

            cv::Mat currentData=cpuImg.colRange(minIdx+crop,maxIdx-crop).clone();



            cpuImg=currentData.clone();

        }



    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);





    std::cout << "masking Process Complete... time : " << duration.count() << " ms" << std::endl;



    ////////////////

    cv::cuda::GpuMat img;

    img.upload(cpuImg);

    cv::cuda::GpuMat padded,heatMap,currPatch;



    cv::cuda::copyMakeBorder(img,padded,0,0,minIdx+padd+crop,crop+reconstructed.cols-maxIdx+padd,cv::BORDER_REFLECT);





    std::vector<cv::cuda::GpuMat> patches;

    std::vector<std::vector<cv::cuda::GpuMat>> batches;

//std::cout<<padded.size()<<std::endl;

    for(int h=0;h<reconstructed.rows-2*padd;h+=patchSize){

        for(int w=0;w<reconstructed.cols;w+=patchSize){



            currPatch=padded(cv::Rect(w,h,patchSize+2*padd,patchSize+2*padd));

            patches.emplace_back(currPatch);



            if(patches.size()==batchSize){

                batches.emplace_back(patches);

                patches.clear();

            }

        }



    }

	if(patches.size()>0){

	batches.emplace_back(patches);

                patches.clear();

	

	}

//    std::cout<<batches.size()<<std::endl;

    end = std::chrono::high_resolution_clock::now();





//    std::cout << "Pre Process Complete... time : " << duration.count() << " ms" << std::endl;



    // Warm up the network before we begin the benchmark

    bool succ;

    std::vector<std::vector<cv::cuda::GpuMat>> sub(batches.begin(),batches.begin()+1);

    std::vector<std::vector<cv::Mat>> featureVectors;

//    std::vector<std::vector<std::vector<float>>> featureVectors;

   // for (int i = 0; i < 10; ++i) {

    //    succ = engine.runInference(sub, featureVectors);

    //    if (!succ) {

    //        throw std::runtime_error("Unable to run inference.");

   //     }

   // }

    start = std::chrono::high_resolution_clock::now();



    cv::Mat wholeRes;

    std::vector<std::vector<cv::cuda::GpuMat>> infBatch;

    std::vector<cv::Mat> results;

    for(const auto &item:batches){

        infBatch.emplace_back(item);

        succ = engine.runInference(infBatch, featureVectors);

        for(const auto &imgRes:featureVectors){

            wholeRes.push_back(imgRes[0]);



        }



        infBatch.clear();

    }

//    cv::Mat wholeRes(128, 128, CV_8UC(results.size()));

//    cv::merge(results, wholeRes);



    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);



    std::cout << "Inference Complete... time : " << duration.count() << " ms" << std::endl;

start=std::chrono::high_resolution_clock::now();

    //std::cout<<wholeRes.size()<<std::endl;



    int stepH=reconstructed.rows/patchSize;

    int stepW=reconstructed.cols/patchSize;

    int hmSize=wholeRes.size[1];



    cv::Mat output=cv::Mat::zeros(cv::Size(hmSize*stepW,hmSize*stepH),CV_32FC1);





    int idx=0;

    for(int h=0;h<output.rows;h+=hmSize){

        for(int w=0;w<output.cols;w+=hmSize){



            cv::Mat curr;

            cv::transpose(wholeRes(cv::Range(idx*hmSize,(idx+1)*(patchSize/8)),cv::Range(0,hmSize)),curr);

            curr.copyTo(output(cv::Range(h,h+hmSize),cv::Range(w,w+hmSize)));

            idx+=1;

//            std::cout<<wholeRes.row(idx).reshape(0,hmSize).size<<std::endl;

            //std::cout<<w<<" "<<h<<" "<<idx<<std::endl;

        }

    }

//    cv::imwrite("mask.jpg",mask);

//    cv::imwrite("raw.jpg",cpuImg);

//    cv::imwrite("thresh.jpg",thresh);

    cv::Mat processBatch(output.size(), output.type());



    cv::resize(mask,mask,output.size(),0,0,cv::INTER_NEAREST);



    cv::Mat subImageMasked;

    processBatch.setTo(cv::Scalar(0));



    output.copyTo(processBatch, mask);

    cv::Mat paddedwrite;

    padded.download(paddedwrite);

    count++;

    auto end2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end);



std::cout << "PostProcess Complete... time : " << duration.count() << " ms" << std::endl;

   

 return processBatch;

}





std::pair<cv::Mat, cv::Mat> Detector::postProcess(cv::Mat input, double ignoreThresh, double areaThresh) {

//    std::cout << "post Started" << std::endl;

    cv::Mat  open;

    bool corrupt = false;

    cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);



    //cv::morphologyEx(thresh,open,cv::MORPH_OPEN,kernel);

    cv::Mat R1, R2, mask;





    std::vector<std::vector<cv::Point> > contours;

    std::vector<cv::Vec4i> hierarchy;

    findContours(input, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > all;

    std::vector<std::vector<cv::Point> > alarm;

    std::cout<<"min area"<< ignoreThresh <<"alarm "<< areaThresh<<std::endl;

    for (const auto &item: contours) {

        double area = cv::contourArea(item);

        if (area > ignoreThresh) {

            all.emplace_back(item);



        }

    }

    if(all.size()>3){ //Number of Defect count
        alarm=all;
    }
    else{
        for (const auto &item: all) {

            double area = cv::contourArea(item);

            if (area > areaThresh) {

                alarm.emplace_back(item);
            }
        }
    }


    cv::Mat zero = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    cv::Mat maskRes = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);





    cv::drawContours(zero, alarm, -1, cv::Scalar(255, 255, 255), cv::FILLED);



    cv::dilate(zero, R1, SK);

    cv::dilate(zero, R2, LK);

    cv::absdiff(R1, R2, mask);





    R1.release();

    R2.release();

    cv::Mat mask2;

    cv::Mat ignoreShow = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    cv::drawContours(ignoreShow, all, -1, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::dilate(ignoreShow, R1, SK);

    cv::dilate(ignoreShow, R2, LK);

    cv::absdiff(R1, R2, mask2);



    maskRes.setTo(cv::Scalar(255, 0, 0), mask2);

    maskRes.setTo(cv::Scalar(0, 0, 255), mask);



    std::pair<cv::Mat, cv::Mat> res;

    res.first = zero;

    res.second = maskRes;





    return res;

}



cv::Mat Detector::enhanced(const cv::Mat &img) {

    cv::Mat equalized_image;

//    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    mClahe->apply(img, equalized_image);

    cv::cvtColor(equalized_image, equalized_image, cv::COLOR_GRAY2BGR);

    return equalized_image;

}



std::map<std::string, int> Detector::report(cv::Mat mask) {

//    cv::imwrite("check0.jpg",mask);



    cv::cvtColor(mask,mask,cv::COLOR_BGR2GRAY);

    cv::threshold(mask,mask,128,255,cv::THRESH_BINARY);

//    cv::imwrite("check1.jpg",mask);

    std::vector<std::vector<cv::Point> > contours;

    std::vector<cv::Vec4i> hierarchy;

    findContours( mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

    std::map<std::string, int> report;

    report["count"]=(int)contours.size();

    report["area"]=cv::countNonZero(mask);

    sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){

        return contourArea(c1, false) < contourArea(c2, false);

    });

    report["minArea"]=(int)cv::contourArea(contours[0]);

    report["maxArea"]=(int)cv::contourArea(contours[contours.size()-1]);

    return report;

}



std::pair<float, float> Detector::calcSize(const cv::Mat& input, float scaleX, float scaleY) {

    cv::Mat gray;





    cv::cvtColor(input,gray,cv::COLOR_BGR2GRAY);

    cv::threshold(gray,gray,230,255,cv::THRESH_BINARY_INV);



    std::vector<std::vector<cv::Point>> contours;

    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);



    cv::Mat forshow;

    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {

        return cv::contourArea(a) > cv::contourArea(b);

    });



    cv::Mat mask = cv::Mat::zeros(gray.size(), gray.type());

    cv::drawContours(mask, contours, 0, cv::Scalar(255), -1);



    //cv::imwrite("checkout.jpg",mask);





    int width=cv::countNonZero(mask.row(mask.rows/2));

    std::pair<float, float> result;

    result.first=width / scaleX;

    result.second=input.rows / scaleY;

    return result;

}


