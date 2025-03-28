/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <postprocess.h>

#include <opencv2/opencv.hpp>
#include <common.h>
#include <iostream>
#include <turbojpeg.h>
#include <vector>
#include <encode_frame.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/polygon_stamped.hpp"
#include <image_transport/image_transport.hpp>

#include <chrono>
#include <iostream>
#include <sys/time.h>

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
class YoloPoseNode : public rclcpp::Node {
public:
    YoloPoseNode() : Node("yolov8_pose_node") {
        // 参数声明
        declare_parameter<std::string>("model_path", "./yolov8_pose.rknn");

        // 初始化模型
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

        init_post_process();

        std::string model_path_ = this->get_parameter("model_path").as_string();
        if(init_yolov8_pose_model(model_path_.c_str(), &rknn_app_ctx) != 0) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize model");
            rclcpp::shutdown();
        }

        // 初始化订阅器，订阅图像话题
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>
        (
            "/camera/color/image_raw", 
            rclcpp::SensorDataQoS().keep_last(30),
            [this, &rknn_app_ctx](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                this->image_callback(msg, &rknn_app_ctx);}  
         );


         // 初始化发布图像话题
        image_pub_=this->create_publisher<sensor_msgs::msg::Image>("output_post_image",30);

        // 正确初始化成员变量 skeleton
        const int init_skeleton[38] = {
            16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7
        };
        memcpy(skeleton, init_skeleton, sizeof(init_skeleton));

    }

private:

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg,rknn_app_context_t* rknn_app_ctx) {
        try {
           
            // 转换ROS图像到OpenCV格式
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            
            // 初始化准备输入图像
            image_buffer_t src_image;
            memset(&src_image, 0, sizeof(image_buffer_t));


            std::vector<uchar> jpegData;
            unsigned char* jpegBuf = nullptr;
            unsigned long jpegSize = 0;

            // 转换到图像缓冲区
            jpegData.clear();    

            jpegData = tjpeg_encode(frame, 95);// 使用 TurboJPEG 编码为 JPEG
            jpegBuf = jpegData.data(); // 获取JPEG数据的指针
            jpegSize = jpegData.size();  // 获取JPEG数据的大小

            // 检查 JPEG 数据是否有效
            if (jpegBuf == nullptr || jpegSize == 0) {
                std::cerr << "JPEG encoding failed or returned empty data!" << std::endl;
                return;
            }

            read_frame_jpeg(jpegBuf,jpegSize,&src_image) ;  //Mat::frame->image_buffer_t src_image

            // 执行推理
            object_detect_result_list od_results;
            if(inference_yolov8_pose_model(rknn_app_ctx, &src_image, &od_results) == 0) {
                // 绘制检测结果
                
                cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
                   // 画框和概率
                char text[256];
                
                for (int i = 0; i < od_results.count; i++)
                {
                    object_detect_result *det_result = &(od_results.results[i]);
                    printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                        det_result->box.left, det_result->box.top,
                        det_result->box.right, det_result->box.bottom,
                        det_result->prop);
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;


                    // // 颜色常量定义（OpenCV使用BGR格式）
                    // const cv::Scalar COLOR_BLUE(255, 0, 0);
                    // const cv::Scalar COLOR_RED(0, 0, 255);
                    // const cv::Scalar COLOR_ORANGE(0, 165, 255);
                    // const cv::Scalar COLOR_YELLOW(0, 255, 255);

                    // 动作检测 + 时序平滑
                    bool current_hips = isHandsOnHips(*det_result);
                    hands_on_hips_history_.push_back(current_hips);
                    if (hands_on_hips_history_.size() > 5) hands_on_hips_history_.pop_front();
                    bool stable_hips = std::all_of(hands_on_hips_history_.begin(), hands_on_hips_history_.end(), [](bool v) { return v; });

                    bool current_hands_up = isHandsUp(*det_result);
                    hands_up_history_.push_back(current_hands_up);
                    if (hands_up_history_.size() > 5) hands_up_history_.pop_front();
                    bool stable_hands_up = std::all_of(hands_up_history_.begin(), hands_up_history_.end(), [](bool v) { return v; });

                    // 根据稳定结果显示文本
                    if (stable_hips) {
                        cv::putText(frame, "Hands on Hips (Stable)", cv::Point(x1, y1 - 30), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                    }
                    if (stable_hands_up) {
                        cv::putText(frame, "Hands Up (Stable)", cv::Point(x1, y1 - 40), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                    }

                    // 画矩形
                    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),cv::Scalar(255, 0, 0),3);

                    sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

                    cv::putText(frame,text,cv::Point(x1, y1 - 20),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0, 0, 255),2); 

                    //画线
                    for (int j = 0; j < 38/2; ++j)
                    {

                    int idx1 = skeleton[2*j] - 1;
                    int idx2 = skeleton[2*j+1] - 1;

                    cv::Point pt1(static_cast<int>(det_result->keypoints[idx1][0]),
                                    static_cast<int>(det_result->keypoints[idx1][1]));
                    cv::Point pt2(static_cast<int>(det_result->keypoints[idx2][0]),
                                    static_cast<int>(det_result->keypoints[idx2][1]));
                    cv::line(frame,pt1,pt2,cv::Scalar(0, 165, 255),3); 
                    }

                    for (int j = 0; j < 17; ++j)
                    {
                        cv::Point center(static_cast<int>(det_result->keypoints[j][0]),
                                         static_cast<int>(det_result->keypoints[j][1]));
                        cv::circle(frame,center,1, cv::Scalar(0, 255, 255),3);      
                    }
                }

                // 转换回ROS Image并发布

                sensor_msgs::msg::Image::SharedPtr output_msg=cv_bridge::CvImage(msg->header,"bgr8",frame).toImageMsg();
                image_pub_->publish(*output_msg);
                
      
            }

            free_image_buffer(&src_image);
        } 
        catch (const cv_bridge::Exception& e) {

            deinit_post_process();
            int ret = release_yolov8_pose_model(rknn_app_ctx);
            if (ret != 0){
                printf("release_yolov5_model fail! ret=%d\n", ret);
                }

            RCLCPP_ERROR(get_logger(), "CV bridge error: %s", e.what());
        }

    }
    
    // rknn_app_context_t rknn_app_ctx;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    int skeleton[38]={0};

        // 添加动作检测成员函数声明
    bool isHandsOnHips(object_detect_result& det_result);
    bool isHandsUp(object_detect_result& det_result);

    std::deque<bool> hands_up_history_;
    std::deque<bool> hands_on_hips_history_;

};
        // 叉腰动作检测实现
    bool YoloPoseNode::isHandsOnHips(object_detect_result& det_result) {
        // 关键点索引定义（根据COCO规范）
        const int LEFT_WRIST = 9;
        const int RIGHT_WRIST = 10;
        const int LEFT_HIP = 11;
        const int RIGHT_HIP = 12;

        // 获取坐标（假设关键点格式为[x, y, confidence]）
        cv::Point left_wrist(det_result.keypoints[LEFT_WRIST][0], det_result.keypoints[LEFT_WRIST][1]);
        cv::Point right_wrist(det_result.keypoints[RIGHT_WRIST][0], det_result.keypoints[RIGHT_WRIST][1]);
        cv::Point left_hip(det_result.keypoints[LEFT_HIP][0], det_result.keypoints[LEFT_HIP][1]);
        cv::Point right_hip(det_result.keypoints[RIGHT_HIP][0], det_result.keypoints[RIGHT_HIP][1]);

        // 置信度过滤
        if (det_result.keypoints[LEFT_WRIST][2] < 0.3 || 
            det_result.keypoints[RIGHT_WRIST][2] < 0.3) {
            return false;
        }

        // 判断逻辑
        bool left_hand_on_hip = 
            (abs(left_wrist.y - left_hip.y) < 25 && 
            (left_wrist.x > left_hip.x));
        bool right_hand_on_hip = 
            (abs(right_wrist.y - right_hip.y) < 25 && 
            (right_wrist.x < right_hip.x));

        return left_hand_on_hip && right_hand_on_hip;
        }

        // 举双手动作检测实现
    bool YoloPoseNode::isHandsUp(object_detect_result& det_result) {
        const int LEFT_WRIST = 9;
        const int RIGHT_WRIST = 10;
        const int LEFT_SHOULDER = 5;
        const int RIGHT_SHOULDER = 6;

        cv::Point left_wrist(det_result.keypoints[LEFT_WRIST][0], det_result.keypoints[LEFT_WRIST][1]);
        cv::Point right_wrist(det_result.keypoints[RIGHT_WRIST][0], det_result.keypoints[RIGHT_WRIST][1]);
        cv::Point left_shoulder(det_result.keypoints[LEFT_SHOULDER][0], det_result.keypoints[LEFT_SHOULDER][1]);
        cv::Point right_shoulder(det_result.keypoints[RIGHT_SHOULDER][0], det_result.keypoints[RIGHT_SHOULDER][1]);

        if (det_result.keypoints[LEFT_WRIST][2] < 0.3 || 
            det_result.keypoints[RIGHT_WRIST][2] < 0.3) {
            return false;
        }

        bool left_hand_up = (left_shoulder.y - left_wrist.y) > 50;
        bool right_hand_up = (right_shoulder.y - right_wrist.y) > 50;

        return left_hand_up && right_hand_up;
        }


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloPoseNode>());
    rclcpp::shutdown();
    return 0;
}
