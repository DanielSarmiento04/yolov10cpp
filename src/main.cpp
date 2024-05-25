#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat)
#include <opencv2/videoio.hpp> // Video write
#include <string>

// Custom extensions
#include "ia/inference.h"

// Using standard namespace
using namespace cv;
using namespace std;


const string video_path = "";
const string window_name = "Camera";


int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <Source> <Model Path> \n";
        exit(1);
    }

    int SOURCE        = atoi(argv[1]);
    string MODEL_PATH = argv[2];


    // Load model
    Inference inference_management(MODEL_PATH, cv::Size(640, 640));

    // open the camera selected
    VideoCapture video_capture(SOURCE); 
    if (!video_capture.isOpened())
    {
        cout  << "Could not open the input video: " << SOURCE << endl;
        return -1;
    }

    for(;;)
    {
        Mat frame;
        bool success = video_capture.read(frame);
        auto keyboard_pressed = waitKey(1);

        if (!success)
        {
            cout << "Cannot read  frame " << endl;
            break;
        }

        // Process the frame
        std::vector<Detection> results = inference_management.process(frame);

        if (keyboard_pressed == 'q')
        {
            break;
        } 
        imshow(window_name, frame);
    }

    video_capture.release();
    destroyAllWindows();
    return 0;
}
