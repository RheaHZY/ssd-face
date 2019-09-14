#ifndef FACE_DETECOR_H
#define FACE_DETECOR_H
#include "opencv2/opencv.hpp"
namespace ssdface{
    class DetectionResult
    {
    public:
        size_t classid;
        double confidence;
        cv::Rect r;
    };

    int loadmodel(std::string modeldir = "../models");

    std::vector<DetectionResult> Detect(const cv::Mat &img);

    cv::Mat drawDetectionResults(const cv::Mat &img, std::vector<DetectionResult>&results);

    int release();
}

#endif