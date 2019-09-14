#include "FaceDetector.h"
#include "config.h"
#include "mrutil.h"

namespace ssdface{

const char *proto = "face_deploy.prototxt";
const char *model = "VGG_Face2017_SSD_300x300_iter_120000.caffemodel";
int _inputH = 300;
int _inputW = 300;
#if USE_CAFFE
    #include <caffe/caffe.hpp>
    caffe::Net<float> *g_net;
#endif

#if USE_DNN
    #include <opencv2/dnn.hpp>
    using namespace cv::dnn;
    cv::dnn::Net g_net;
#endif

    int loadmodel(std::string modeldir)
    {
        #if USE_CAFFE
        g_net = new caffe::Net<float>(modeldir + "/" + proto, caffe::TEST);
        g_net->CopyTrainedLayersFrom(modeldir + "/" + model);
        if (g_net)
        {
            auto inputblob = g_net->input_blobs()[0];
            _inputH = inputblob->height();
            _inputW = inputblob->width();
        }
        #endif

        #if USE_DNN
        cv::String _modelTxt = modeldir + "/face_deploy.prototxt";
        cv::String _modelBin = modeldir + "/VGG_Face2017_SSD_300x300_iter_120000.caffemodel";
        g_net = cv::dnn::readNetFromCaffe(_modelTxt, _modelBin);
        #endif
        return false;
    }

    cv::Mat preprocess(const cv::Mat& img)
    {
        cv::Mat preprocessed;
        img.convertTo(preprocessed, CV_32F);
        cv::resize(preprocessed, preprocessed, cv::Size(_inputW, _inputH));
        cv::subtract(preprocessed, cv::Scalar(104,117,123), preprocessed);
        return preprocessed;
    }
    std::vector<DetectionResult> Detect(const cv::Mat &img)
    {
        std::vector<DetectionResult> faces; 
        #if USE_DNN
        cv::Mat preprocessedFrame = preprocess(img);
        cv::Mat inputBlob = blobFromImage(preprocessedFrame);
        g_net.setInput(inputBlob, "data");
        cv::Mat detection = g_net.forward("detection_out");
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.1)
            {
                DetectionResult dr;
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                float xLeftBottom = detectionMat.at<float>(i, 3) * img.cols;
                float yLeftBottom = detectionMat.at<float>(i, 4) * img.rows;
                float xRightTop = detectionMat.at<float>(i, 5) * img.cols;
                float yRightTop = detectionMat.at<float>(i, 6) * img.rows;
                cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));
                    dr.classid = objectClass;
                dr.confidence = confidence;
                dr.r = object;
                faces.push_back(dr);
            }
        }
        #endif
        return faces;
    }

    cv::Mat drawDetectionResults(const cv::Mat &img, std::vector<DetectionResult>&results)
    {
        cv::Mat show = img.clone();
        for (int i = 0; i < results.size(); i++)
        {
            auto dr=results[i];
            cv::rectangle(show, dr.r, cv::Scalar(0, 255, 0));
            std::string title = "Face:" + double2string(dr.confidence);
            cv::putText(show, title, cv::Point(dr.r.x,dr.r.y), 1, 1,CV_RGB(255, 0, 0));
        }
        return show;
    }

    int release()
    {
        return true;
    }
}