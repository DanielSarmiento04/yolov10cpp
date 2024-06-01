#include "./ia/inference.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <source>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];

    auto source = atoi(argv[1]); // 0 for webcam, 1 for video file
    int apiID = cv::CAP_ANY;     // 0 = autodetect default API

    cv::namedWindow("yolov10", cv::WINDOW_AUTOSIZE);

    InferenceEngine engine(model_path);

    cv::VideoCapture cap;

    cap.open(source, apiID);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cv::Mat frame;

    std::cout << "Start grabbing" << std::endl
              << "Press any key to terminate" << std::endl;

    for (;;)
    {
        cap.read(frame);

        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        int orig_width = frame.cols;
        int orig_height = frame.rows;
        auto timer = cv::getTickCount();

        std::vector<float> input_tensor_values = engine.preprocessImage(frame);

        std::vector<float> results = engine.runInference(input_tensor_values);

        float confidence_threshold = 0.3;

        std::vector<Detection> detections = engine.filterDetections(results, confidence_threshold, engine.input_shape[2], engine.input_shape[3], orig_width, orig_height);

        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8);

        cv::Mat output = engine.draw_labels(frame, detections);

        cv::imshow("test", output);

        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}
