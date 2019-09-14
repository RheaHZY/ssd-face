#include "mrdir.h"
#include "mropencv.h"
#include "FaceDetector.h"
using namespace ssdface;

int testcamera(int cameraindex=0)
{
    cv::VideoCapture capture(cameraindex);
    cv::Mat frame;
    while (true)
    {
        capture >> frame;
        if(!frame.data)
            break;
        auto faces = Detect(frame);
        auto show = drawDetectionResults(frame, faces);
        imshow("SSDFace", show);
        waitKey(1);
    }
    return 0;
}

int testimage(std::string imgpath= "../images/000001.jpg")
{
    cv::Mat img = cv::imread(imgpath);
    auto faces = Detect(img);
    auto show = drawDetectionResults(img, faces);
    cv::imshow("face",show);
    cv::waitKey();
    return 0;
}

int testdir(const std::string imgdir="./")
{
    auto files = getAllFilesinDir(imgdir,"*.jpg");
    for (int i = 0; i < files.size(); i++)
    {
        std::string filepath = imgdir + "/" + files[i];
        cv::Mat frame = cv::imread(filepath);
        cv::namedWindow("SSDFace", 0);
        cv::imshow("SSDFace", frame);
        cv::waitKey();
    }
    return 0;
}

int main(int argc, char** argv)
{
    loadmodel();
    testimage();
//    testcamera();
//    testdir();
//    test_ibm();
    release();
}