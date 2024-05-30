#include "inference.h"

/*
    * This class is responsible for loading the ONNX model and running the inference
    * 
    * The model is loaded in the constructor and the inference is run in the proccess_image method
    * 
*/
Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    loadOnnxNetwork();
    loadClassesFromFile(); // The classes are hard-coded for this example
}

std::vector<Detection> Inference::process(const cv::Mat &input)
{
    
}




/*
    * This method is responsible for:
    * 
    * override the default classes with the classes from the file
    * 
    * The classes are hard-coded for default 
    * please go to `/inference.h`
*/
void Inference::loadClassesFromFile()
{

    // verify if fie is overridden
    if (classesPath != "")
    {
        std::ifstream inputFile(classesPath);
        if (inputFile.is_open())
        {
            std::string classLine;
            while (std::getline(inputFile, classLine))
                classes.push_back(classLine);
            inputFile.close();
        }
    }
}
