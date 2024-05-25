#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // Print OpenCV build information
    std::cout << cv::getBuildInformation() << std::endl;

    // Path to your ONNX model
    const std::string model_path = "/Users/josedanielsarmientoblanco/Desktop/hobby/yolov10cpp/yolov10n.onnx";

    try {
        // Load the ONNX model
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        std::cout << "ONNX model loaded successfully." << std::endl;

        // Perform additional operations with the model if necessary

    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
    }

    return 0;
}
