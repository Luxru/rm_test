#include <stdio.h>

#include <opencv2/opencv.hpp>

using namespace cv;
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: week_01 <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread(argv[1], IMREAD_COLOR);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // 定义红色的HSV范围
    cv::Scalar redLower1(0, 100, 50);
    cv::Scalar redUpper1(5, 255, 255);
    cv::Scalar redLower2(175, 100, 50);
    cv::Scalar redUpper2(180, 255, 255);

    // 定义蓝色的HSV范围
    cv::Scalar blueLower(100, 100, 0);
    cv::Scalar blueUpper(135, 255, 255);

    // 创建红色掩码
    cv::Mat redMask,redMask1,redMask2;
    cv::inRange(hsvImage, redLower1, redUpper1, redMask1);
    cv::inRange(hsvImage, redLower2, redUpper2, redMask2);
    cv::bitwise_or(redMask1,redMask2,redMask);
    // 创建蓝色掩码
    cv::Mat blueMask;
    cv::inRange(hsvImage, blueLower, blueUpper, blueMask);

    // 对红色和蓝色掩码进行形态学操作，以去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(redMask, redMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(blueMask, blueMask, cv::MORPH_OPEN, kernel);

    // 寻找红色和蓝色色块的轮廓
    std::vector<std::vector<cv::Point>> redContours, blueContours;
    cv::findContours(redMask, redContours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(blueMask, blueContours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // 遍历红色色块的轮廓并标记中心坐标
    for (const auto& redContour : redContours) {
        cv::Moments moments = cv::moments(redContour);
        if (moments.m00 != 0) {
            int cX = static_cast<int>(moments.m10 / moments.m00);
            int cY = static_cast<int>(moments.m01 / moments.m00);
            cv::circle(image, cv::Point(cX, cY), 5, cv::Scalar(0, 0, 255), -1);
        }
    }

    // 遍历蓝色色块的轮廓并标记中心坐标
    for (const auto& blueContour : blueContours) {
        cv::Moments moments = cv::moments(blueContour);
        if (moments.m00 != 0) {
            int cX = static_cast<int>(moments.m10 / moments.m00);
            int cY = static_cast<int>(moments.m01 / moments.m00);
            cv::circle(image, cv::Point(cX, cY), 5, cv::Scalar(255, 0, 0), -1);
        }
    }

    // 显示结果图像
    cv::imshow("Result", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}