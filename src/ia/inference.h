#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

enum VERSION
{
    V5,
    V8,
    V9,
    V10,
};

class Inference
{
public:
    Inference(
        const std::string &onnxModelPath,
        const cv::Size &modelInputShape = {640, 640},
        const VERSION &version = VERSION::V9,
        const std::string &classesTxtFile = "",
        const bool &runWithCuda = false);
    std::vector<Detection> process(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};
    VERSION versionModel{};

    // define the default classes
    std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    cv::Size2f modelShape{};

    // Default parameters
    float modelConfidenceThreshold{0.25}; // is the probability threshold for the detections
    float modelScoreThreshold{0.45};      // is the threshold for the score (confidence * class probability)
    float modelNMSThreshold{0.50};        // is the probability threshold for the non-maximum suppression algorithm

    bool letterBoxForSquare = true; // if true, the image will be letter boxed to fit the model input shape

    cv::dnn::Net net;
};

#endif
