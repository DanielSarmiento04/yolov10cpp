#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection
{
    float confidence;
    cv::Rect bbox;
    int class_id;
    std::string class_name;
};


class InferenceEngine
{
public:
    InferenceEngine(const std::string &model_path);
    ~InferenceEngine();

    std::vector<float> preprocessImage(const cv::Mat &image);
    std::vector<Detection> filterDetections(const std::vector<float> &results, float confidence_threshold, int img_width, int img_height, int orig_width, int orig_height);
    std::vector<float> runInference(const std::vector<float> &input_tensor_values);
    
    cv::Mat draw_labels(const cv::Mat &image, const std::vector<Detection> &detections);

    std::vector<int64_t> input_shape;
    
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

    std::string getInputName();
    std::string getOutputName();

    static const std::vector<std::string> CLASS_NAMES;
};


#endif // INFERENCE_H
