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
#include "bytetrack/BYTETracker.h"

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
class YoloPoseNode : public rclcpp::Node
{
public:
    YoloPoseNode() : Node("yolov8_pose_node")
    {
        // 参数声明
        declare_parameter<std::string>("model_path", "./yolov8_pose.rknn");

        // 初始化模型
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

        init_post_process();

        std::string model_path_ = this->get_parameter("model_path").as_string();
        if (init_yolov8_pose_model(model_path_.c_str(), &rknn_app_ctx) != 0)
        {
            RCLCPP_ERROR(get_logger(), "Failed to initialize model");
            rclcpp::shutdown();
        }

        // 初始化订阅器，订阅图像话题
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw",
            rclcpp::SensorDataQoS().keep_last(30),
            [this, &rknn_app_ctx](sensor_msgs::msg::Image::ConstSharedPtr msg)
            { this->image_callback(msg, &rknn_app_ctx); });

        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw",
            rclcpp::SensorDataQoS().keep_last(30),
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg)
            {
                this->depth_image_callback(std::const_pointer_cast<sensor_msgs::msg::Image>(msg));
            });

        // 初始化发布图像话题
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("output_post_image", 30);

        // 正确初始化成员变量 skeleton
        const int init_skeleton[38] = {
            16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
        memcpy(skeleton, init_skeleton, sizeof(init_skeleton));

        // 初始化发布跟踪话题
        tracked_pub_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/tracked_objects", 30);

        // 初始化 跟踪器
        tracker_ = std::make_unique<BYTETracker>(30, 90);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, rknn_app_context_t *rknn_app_ctx)
    {
        try
        {

            // 转换ROS图像到OpenCV格式
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

            // 初始化准备输入图像
            image_buffer_t src_image;
            memset(&src_image, 0, sizeof(image_buffer_t));

            std::vector<uchar> jpegData;
            unsigned char *jpegBuf = nullptr;
            unsigned long jpegSize = 0;

            // 转换到图像缓冲区
            jpegData.clear();

            jpegData = tjpeg_encode(frame, 95); // 使用 TurboJPEG 编码为 JPEG
            jpegBuf = jpegData.data();          // 获取JPEG数据的指针
            jpegSize = jpegData.size();         // 获取JPEG数据的大小

            // 检查 JPEG 数据是否有效
            if (jpegBuf == nullptr || jpegSize == 0)
            {
                std::cerr << "JPEG encoding failed or returned empty data!" << std::endl;
                return;
            }

            read_frame_jpeg(jpegBuf, jpegSize, &src_image); // Mat::frame->image_buffer_t src_image

            // 执行推理
            object_detect_result_list od_results;
            if (inference_yolov8_pose_model(rknn_app_ctx, &src_image, &od_results) != 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Inference failed!");
                return;
            }

            std::cout << "detect_nuber: " << od_results.count << std::endl;

            // 转换检测结果到跟踪格式
            std::vector<Object> trackobj;
            publish_decobj_to_trackobj(od_results, trackobj);

            // 更新跟踪器
            auto tracks = tracker_->update(trackobj);

            for (const auto &track : tracks)
            {
                int track_id = track.track_id;

                // 获取检测结果
                auto det_result = find_detection_by_bbox(track.tlbr, od_results);
                if (!det_result)
                    continue;

                // 初始化或更新跟踪对象
                if (tracked_persons_.find(track_id) == tracked_persons_.end())
                {
                    // tracked_persons_[track_id] = {track_id, false, cv::Mat(), {}, this->now(), rclcpp::Time()};
                    tracked_persons_[track_id] = {track_id, false, {}, this->now(), rclcpp::Time()};
                }
                auto &person = tracked_persons_[track_id];

                // 更新举手历史
                bool hands_up = isHandsUp(*det_result);
                person.hands_up_history.push_back(hands_up);
                if (person.hands_up_history.size() > 60)
                    person.hands_up_history.pop_front();

                // 判断持续举手条件（2秒内60次true）
                bool is_hands_up_long = std::count(person.hands_up_history.begin(),
                                                   person.hands_up_history.end(), true) >= 60;

                // 状态机逻辑
                if (!person.is_tracking)
                {
                    // 检查冷却时间
                    bool in_cooldown = (person.hands_up_stop_time.seconds() != 0.0) &&
                                       ((this->now() - person.hands_up_stop_time).seconds() < 10.0);

                    if (!in_cooldown && is_hands_up_long)
                    {
                        person.is_tracking = true;
                        person.hands_up_start_time = this->now();
                        // // 保存特征
                        // cv::Rect roi(track.tlbr[0], track.tlbr[1],
                        //              track.tlbr[2] - track.tlbr[0], track.tlbr[3] - track.tlbr[1]);
                        // person.feature = extract_feature(frame(roi));
                    }
                }
                else
                {
                    // 检查跟踪超时（5秒）和持续举手条件
                    bool tracking_timeout = (this->now() - person.hands_up_start_time).seconds() >= 5.0;

                    if (tracking_timeout && is_hands_up_long)
                    {
                        person.is_tracking = false;
                        person.hands_up_stop_time = this->now();
                    }
                }
            }

            // 过滤只绘制激活目标
            std::vector<STrack> filtered_tracks;
            for (const auto &track : tracks)
            {
                if (tracked_persons_[track.track_id].is_tracking)
                {
                    filtered_tracks.push_back(track);
                }
            }

            // 绘制结果
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            draw_tracking_results(frame, filtered_tracks, od_results);

            // 转换回ROS Image并发布

            sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
            image_pub_->publish(*output_msg);

            if (src_image.virt_addr != NULL)
            {
                free(src_image.virt_addr);
            }
        }

        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(get_logger(), "CV bridge error: %s", e.what());
            deinit_post_process();
            int ret = release_yolov8_pose_model(rknn_app_ctx);
            if (ret != 0)
            {
                printf("release_yolov8_pose_model fail! ret=%d\n", ret);
            }
        }
    }

    // 根据跟踪框匹配检测结果
    object_detect_result *find_detection_by_bbox(const std::vector<float> &tlbr,
                                                 object_detect_result_list &results)
    {
        const float IOU_THRESHOLD = 0.6f;
        float max_iou = 0.0f;
        object_detect_result *best_match = nullptr;

        for (int i = 0; i < results.count; ++i)
        {
            auto &det = results.results[i];
            float iou = calculate_iou(det, tlbr);
            if (iou > max_iou)
            {
                max_iou = iou;
                best_match = &det;
            }
        }

        if (best_match == nullptr)
        {
            RCLCPP_DEBUG(this->get_logger(), "Track box [%.1f,%.1f,%.1f,%.1f] has no matched detection",
                         tlbr[0], tlbr[1], tlbr[2], tlbr[3]);
        }
        // return best_match;

        return (max_iou >= IOU_THRESHOLD) ? best_match : nullptr;
    }

    // 计算两个矩形的IOU
    static float calculate_iou(const object_detect_result &det, const std::vector<float> &tlbr)
    {
        float det_x1 = det.box.left;
        float det_y1 = det.box.top;
        float det_x2 = det.box.right;
        float det_y2 = det.box.bottom;

        float track_x1 = tlbr[0];
        float track_y1 = tlbr[1];
        float track_x2 = tlbr[2];
        float track_y2 = tlbr[3];

        float xx1 = std::max(det_x1, track_x1);
        float yy1 = std::max(det_y1, track_y1);
        float xx2 = std::min(det_x2, track_x2);
        float yy2 = std::min(det_y2, track_y2);

        float w = std::max(0.0f, xx2 - xx1);
        float h = std::max(0.0f, yy2 - yy1);
        float inter_area = w * h;

        float det_area = (det_x2 - det_x1) * (det_y2 - det_y1);
        float track_area = (track_x2 - track_x1) * (track_y2 - track_y1);
        float union_area = det_area + track_area - inter_area;

        return (union_area == 0) ? 0.0f : (inter_area / union_area);
    }

    // // 特征提取函数（示例）
    // cv::Mat extract_feature(const cv::Mat &roi)
    // {
    //     cv::Mat feature;
    //     cv::resize(roi, feature, cv::Size(64, 128)); // 简单resize作为特征
    //     return feature;
    // }

    // 深度图像回调
    void depth_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        try
        {
            // 使用 cv_bridge 将 ROS 图像消息转化为 OpenCV 图像
            cv::Mat depth_image = cv_bridge::toCvShare(msg, "32FC1")->image;
            depth_image_ = depth_image.clone();
            // 你可以在这里对深度图像进行处理，如果需要做其他操作
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
    }

    void publish_decobj_to_trackobj(object_detect_result_list &results, std::vector<Object> &trackobj)
    {

        if (results.count > 0)
        {
            trackobj.clear();
        }
        for (int i = 0; i < results.count; i++)
        {
            Object obj;
            if (results.results[i].cls_id != 0)
                continue;
            obj.classId = results.results[i].cls_id;
            obj.score = results.results[i].prop;
            obj.box = cv::Rect(
                results.results[i].box.left,
                results.results[i].box.top,
                results.results[i].box.right - results.results[i].box.left,
                results.results[i].box.bottom - results.results[i].box.top);

            trackobj.push_back(obj);
        }
    }

    void draw_line_point(cv::Mat &img, const object_detect_result &det_result)
    {
        // 画线
        for (int j = 0; j < 38 / 2; ++j)
        {

            int idx1 = skeleton[2 * j] - 1;
            int idx2 = skeleton[2 * j + 1] - 1;

            cv::Point pt1(static_cast<int>(det_result.keypoints[idx1][0]),
                          static_cast<int>(det_result.keypoints[idx1][1]));
            cv::Point pt2(static_cast<int>(det_result.keypoints[idx2][0]),
                          static_cast<int>(det_result.keypoints[idx2][1]));
            cv::line(img, pt1, pt2, cv::Scalar(0, 165, 255), 3);
        }

        for (int j = 0; j < 17; ++j)
        {
            cv::Point center(static_cast<int>(det_result.keypoints[j][0]),
                             static_cast<int>(det_result.keypoints[j][1]));
            cv::circle(img, center, 1, cv::Scalar(0, 255, 255), 3);
        }
    }

    void draw_tracking_results(cv::Mat &img, const std::vector<STrack> &tracks, object_detect_result_list &results)
    {

        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp = this->get_clock()->now();
        polygon_msg.header.frame_id = "camera_link";

        std::cout << "Tracking result (output_stracks):" << std::endl;
        if (tracks.empty())
        {
            cv::putText(img, "No active tracking targets", cv::Point(20, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }
        for (const auto &track : tracks)
        {
            if (tracked_persons_[track.track_id].is_tracking)
            {
                // 获取对应检测结果
                auto det_result = find_detection_by_bbox(track.tlbr, results);
                if (!det_result)
                {
                    RCLCPP_DEBUG(this->get_logger(), "No valid detection for track %d", track.track_id);
                    continue; // 跳过无效检测
                }

                draw_line_point(img, *det_result);

                // 画出跟踪框
                std::cout << "track_id: " << track.track_id << ",Track  Bounding Box:[" << track.tlbr[0] << "," << track.tlbr[1] << "," << track.tlbr[2] << "," << track.tlbr[3] << "]" << ",sorce:" << track.score << std::endl;
                int x1 = static_cast<int>(track.tlbr[0]);
                int y1 = static_cast<int>(track.tlbr[1]);
                int x2 = static_cast<int>(track.tlbr[2]);
                int y2 = static_cast<int>(track.tlbr[3]);
                cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

                // 在中心点绘制红色圆点
                cv::circle(img, cv::Point((x1 + x2) / 2, (y1 + y2) / 2), 5, cv::Scalar(0, 0, 255), -1); // 红色实心圆
                float center_x = (x1 + x2) / 2;
                float center_y = (y1 + y2) / 2;

                float depth = 0.0f;
                // 从深度图像中获取中心点的深度值
                if (det_result != nullptr)
                {
                    depth = compute_body_depth(*det_result); // 获取关键关键节点的平均深度（距离）
                }


                // 将像素坐标转换为相机坐标系下的三维坐标
                geometry_msgs::msg::Point32 point;
                point.x = center_x; // X in camera frame
                point.y =center_y; // Y in camera frame
                point.z = depth;                        // Z (depth) in camera frame

                // 添加点到消息
                polygon_msg.polygon.points.push_back(point);

                std::cout << "(X,Y,Z):(" << center_x << "," << center_y << "," << depth << ")" << std::endl;
                // 打印目标信息
                // RCLCPP_INFO(this->get_logger(), "Detected object center [%f, %f], Depth: %.2f meters",
                //                 center_x, center_y, depth);

                // 显示跟踪 ID
                int text_x = x1;
                int text_y = y1 - 5;

                // 添加状态提示
                std::string state_text = "Tracking ID: " + std::to_string(track.track_id);
                cv::putText(img, state_text, cv::Point(x1, y1 - 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                // cv::putText(img, std::to_string(track.track_id), cv::Point(text_x,text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

                cv::putText(img, std::to_string(depth), cv::Point((x1 + x2) / 2, (y1 + y2) / 2 - 5), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
            }
        }
        tracked_pub_->publish(polygon_msg);

    }

    // 变量声明

    // 数据结构：跟踪目标状态
    struct TrackedPerson
    {
        int track_id;
        bool is_tracking;                  // 是否正在跟踪
        //cv::Mat feature;                   // 保存的特征
        std::deque<bool> hands_up_history; // 举手状态历史
        rclcpp::Time hands_up_start_time;  // 举手开始时间
        rclcpp::Time hands_up_stop_time;   // 举手结束时间
    };

    std::map<int, TrackedPerson> tracked_persons_; // 所有检测到的人
    int active_track_id_ = -1;                     // 当前激活的跟踪ID

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    cv::Mat depth_image_;
    std::mutex depth_mutex_;

    int skeleton[38] = {0};

    // 动作检测成员函数声明
    bool isHandsUp(object_detect_result &det_result);
    float compute_body_depth(object_detect_result &det_result);

    std::deque<bool> hands_up_history_;
    // std::deque<bool> hands_on_hips_history_;

    std::unique_ptr<BYTETracker> tracker_;
    std::vector<Object> track_objs;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr tracked_pub_;
};

// 举双手动作检测实现
bool YoloPoseNode::isHandsUp(object_detect_result &det_result)
{
    const int LEFT_WRIST = 9;
    const int RIGHT_WRIST = 10;
    const int LEFT_SHOULDER = 5;
    const int RIGHT_SHOULDER = 6;

    cv::Point left_wrist(det_result.keypoints[LEFT_WRIST][0], det_result.keypoints[LEFT_WRIST][1]);
    cv::Point right_wrist(det_result.keypoints[RIGHT_WRIST][0], det_result.keypoints[RIGHT_WRIST][1]);
    cv::Point left_shoulder(det_result.keypoints[LEFT_SHOULDER][0], det_result.keypoints[LEFT_SHOULDER][1]);
    cv::Point right_shoulder(det_result.keypoints[RIGHT_SHOULDER][0], det_result.keypoints[RIGHT_SHOULDER][1]);

    if (det_result.keypoints[LEFT_WRIST][2] < 0.3 ||
        det_result.keypoints[RIGHT_WRIST][2] < 0.3)
    {
        return false;
    }

    bool left_hand_up = (left_shoulder.y - left_wrist.y) > 50;
    bool right_hand_up = (right_shoulder.y - right_wrist.y) > 50;

    return left_hand_up || right_hand_up;
}

// 计算有效关键点的平均距离
float YoloPoseNode::compute_body_depth(object_detect_result &det_result)
{
    std::lock_guard<std::mutex> lock(depth_mutex_);
    std::vector<float> valid_depths;

    // 遍历所有关键点（COCO 17个关键点）
    for (int j = 0; j < 17; ++j)
    {
        // 获取关键点坐标和置信度
        float x = det_result.keypoints[j][0];
        float y = det_result.keypoints[j][1];
        float confidence = det_result.keypoints[j][2];

        // 过滤低置信度关键点
        if (confidence < 0.3f)
            continue;

        // 检查坐标是否在深度图像范围内
        if (x < 0 || y < 0 || x >= depth_image_.cols || y >= depth_image_.rows)
        {
            continue;
        }

        // 获取深度值（单位：米）
        float depth = depth_image_.at<float>(y, x) / 1000.0f;

        // 剔除无效深度（0或异常值）
        if (depth <= 0.1f || depth > 10.0f)
            continue;

        valid_depths.push_back(depth);
    }

    // 无有效数据时返回0
    if (valid_depths.empty())
        return 0.0f;

    // 计算平均深度
    float sum = std::accumulate(valid_depths.begin(), valid_depths.end(), 0.0f);
    float mean_depth = sum / valid_depths.size();

    // 剔除离群点（与均值差异超过1米）
    std::vector<float> filtered_depths;
    for (float d : valid_depths)
    {
        if (std::abs(d - mean_depth) <= 1.0f)
        {
            filtered_depths.push_back(d);
        }
    }

    if (filtered_depths.empty())
        return 0.0f;

    // 返回最终平均深度
    sum = std::accumulate(filtered_depths.begin(), filtered_depths.end(), 0.0f);
    return sum / filtered_depths.size();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloPoseNode>());
    rclcpp::shutdown();
    return 0;
}
