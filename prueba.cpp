#include <onnxruntime_cxx_api.h>
#include <iostream>


int main()
{
    // check providers
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers)
    {
        std::cout << provider << std::endl;
    }
    return 0;
}