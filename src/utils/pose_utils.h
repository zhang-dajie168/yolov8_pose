#include <opencv2/opencv.hpp>
#include <postprocess.h>
#include <cstdlib> 


cv::Point getKeyPoint(const object_detect_result& det_result, int idx);

bool isHandsOnHips(const object_detect_result& det_result);

bool isHandsUp(const object_detect_result& det_result);