#include <opencv2/opencv.hpp>
#include <postprocess.h>
#include <cstdlib> 

const int NOSE = 0;
const int LEFT_SHOULDER = 5;
const int RIGHT_SHOULDER = 6;
const int LEFT_ELBOW = 7;
const int RIGHT_ELBOW = 8;
const int LEFT_WRIST = 9;
const int RIGHT_WRIST = 10;
const int LEFT_HIP = 11;
const int RIGHT_HIP = 12;
// 其他关键点按需定义...

cv::Point getKeyPoint(const object_detect_result& det_result, int idx) {
    return cv::Point(
        static_cast<int>(det_result.keypoints[idx][0]),
        static_cast<int>(det_result.keypoints[idx][1])
    );
}

bool isHandsOnHips(const object_detect_result& det_result) {
    // 获取关键点坐标
    cv::Point left_wrist = getKeyPoint(det_result, LEFT_WRIST);
    cv::Point right_wrist = getKeyPoint(det_result, RIGHT_WRIST);
    cv::Point left_hip = getKeyPoint(det_result, LEFT_HIP);
    cv::Point right_hip = getKeyPoint(det_result, RIGHT_HIP);

    // 检查置信度（假设关键点数据为[x, y, confidence]）
    if (det_result.keypoints[LEFT_WRIST][2] < 0.3 || 
        det_result.keypoints[RIGHT_WRIST][2] < 0.3) {
        return false;
    }

    // 判断手腕在髋部附近
    bool left_hand_on_hip = 
        (abs(left_wrist.y - left_hip.y) < 20) &&  // y坐标相近
        (left_wrist.x > left_hip.x);              // 手腕在髋部外侧

    bool right_hand_on_hip = 
        (abs(right_wrist.y - right_hip.y) < 20) && 
        (right_wrist.x < right_hip.x);

    return left_hand_on_hip && right_hand_on_hip;
}



bool isHandsUp(const object_detect_result& det_result) {
    cv::Point left_wrist = getKeyPoint(det_result, LEFT_WRIST);
    cv::Point right_wrist = getKeyPoint(det_result, RIGHT_WRIST);
    cv::Point left_shoulder = getKeyPoint(det_result, LEFT_SHOULDER);
    cv::Point right_shoulder = getKeyPoint(det_result, RIGHT_SHOULDER);

    // 检查置信度
    if (det_result.keypoints[LEFT_WRIST][2] < 0.3 || 
        det_result.keypoints[RIGHT_WRIST][2] < 0.3) {
        return false;
    }

    // 判断手腕高于肩膀一定距离
    bool left_hand_up = 
        (left_shoulder.y - left_wrist.y) > 50;  // 手腕比肩膀高50像素

    bool right_hand_up = 
        (right_shoulder.y - right_wrist.y) > 50;

    return left_hand_up && right_hand_up;
}