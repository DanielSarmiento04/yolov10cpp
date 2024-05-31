#include "inference.h"
#include <algorithm>
#include <iostream>

const std::vector<std::string> InferenceEngine::CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

InferenceEngine::InferenceEngine(const std::string &model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
      session_options(),
      session(env, model_path.c_str(), session_options),
      input_shape{1, 3, 640, 640}
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

InferenceEngine::~InferenceEngine() {}

/*
 * Function to preprocess the image
 *
 * @param image_path: path to the image
 * @param orig_width: original width of the image
 * @param orig_height: original height of the image
 *
 * @return: vector of floats representing the preprocessed image
 */
std::vector<float> InferenceEngine::preprocessImage(const cv::Mat &image)
{
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image");
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_shape[2], input_shape[3]));

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);

    std::vector<cv::Mat> channels(3);
    cv::split(resized_image, channels);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float *)channels[c].data, (float *)channels[c].data + input_shape[2] * input_shape[3]);
    }

    return input_tensor_values;
}

/*
    * Function to filter the detections based on the confidence threshold
    *
    * @param results: vector of floats representing the output tensor
    * @param confidence_threshold: minimum confidence threshold
    * @param img_width: width of the input image
    * @param img_height: height of the input image
    * @param orig_width: original width of the image
    * @param orig_height: original height of the image
    *
    * @return: vector of Detection objects

*/
std::vector<Detection> InferenceEngine::filterDetections(const std::vector<float> &results, float confidence_threshold, int img_width, int img_height, int orig_width, int orig_height)
{
    std::vector<Detection> detections;
    const int num_detections = results.size() / 6;

    for (int i = 0; i < num_detections; ++i)
    {
        float left = results[i * 6 + 0];
        float top = results[i * 6 + 1];
        float right = results[i * 6 + 2];
        float bottom = results[i * 6 + 3];
        float confidence = results[i * 6 + 4];
        int class_id = results[i * 6 + 5];

        if (confidence >= confidence_threshold)
        {
            int x = static_cast<int>(left * orig_width / img_width);
            int y = static_cast<int>(top * orig_height / img_height);
            int width = static_cast<int>((right - left) * orig_width / img_width);
            int height = static_cast<int>((bottom - top) * orig_height / img_height);

            detections.push_back(
                {confidence,
                 cv::Rect(x, y, width, height),
                 class_id,
                 CLASS_NAMES[class_id]});
        }
    }

    return detections;
}


/*
    * Function to run inference
    *
    * @param input_tensor_values: vector of floats representing the input tensor
    *
    * @return: vector of floats representing the output tensor
*/
std::vector<float> InferenceEngine::runInference(const std::vector<float> &input_tensor_values)
{
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name = getInputName();
    std::string output_name = getOutputName();

    const char *input_name_ptr = input_name.c_str();
    const char *output_name_ptr = output_name.c_str();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input_tensor_values.data()), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name_ptr, &input_tensor, 1, &output_name_ptr, 1);

    float *floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(floatarr, floatarr + output_tensor_size);
}

/*
    * Function to draw the labels on the image
    *
    * @param image: input image
    * @param detections: vector of Detection objects
    *
    * @return: image with labels drawn

*/
cv::Mat InferenceEngine::draw_labels(const cv::Mat &image, const std::vector<Detection> &detections)
{
    cv::Mat result = image.clone();

    for (const auto &detection : detections)
    {
        cv::rectangle(result, detection.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = detection.class_name + ": " + std::to_string(detection.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(
            result,
            cv::Point(detection.bbox.x, detection.bbox.y - labelSize.height),
            cv::Point(detection.bbox.x + labelSize.width, detection.bbox.y + baseLine),
            cv::Scalar(255, 255, 255),
            cv::FILLED);

        cv::putText(
            result,
            label,
            cv::Point(
                detection.bbox.x,
                detection.bbox.y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
    }

    return result;
}

/*
    * Function to get the input name
    *
    * @return: name of the input tensor
*/
std::string InferenceEngine::getInputName()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

/*
    * Function to get the output name
    *
    * @return: name of the output tensor
*/
std::string InferenceEngine::getOutputName()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetOutputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}
