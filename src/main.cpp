#include "./ia/inference.h"
#include <iostream>
#include <opencv2/opencv.hpp>



int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try
    {
        InferenceEngine engine(model_path);
    
        cv::Mat image = cv::imread(image_path);
        int orig_width = image.cols;
        int orig_height = image.rows;
        std::vector<float> input_tensor_values = engine.preprocessImage(image );

        std::vector<float> results = engine.runInference(input_tensor_values);

        float confidence_threshold = 0.3;

        std::vector<Detection> detections = engine.filterDetections(results, confidence_threshold, engine.input_shape[2], engine.input_shape[3], orig_width, orig_height);

        cv::Mat output = engine.draw_labels(image, detections);

        cv::imwrite("result.jpg", output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
