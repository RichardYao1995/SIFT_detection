#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
using namespace std;
using namespace chrono;

int main()
{
    cv::Mat image_src = cv::imread("../image/11.png");
    cv::Mat image_gray, image_hist;
    cv::cvtColor(image_src, image_gray, CV_RGB2GRAY);
    
    auto start_sift = std::chrono::system_clock::now();
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift_detector = cv::xfeatures2d::SiftFeatureDetector::create(200);    
    std::vector<cv::KeyPoint> key_points_sift;
    sift_detector->detect(image_gray, key_points_sift);
    cv::Mat output_sift;
    cv::drawKeypoints(image_gray, key_points_sift, output_sift);
    cv::imwrite("result_sift.jpg", output_sift);
    std::cout << key_points_sift.size() << " time: " << chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start_sift).count() / 1000 << "ms" << std::endl;

    auto start_surf = std::chrono::system_clock::now();
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surf_detector = cv::xfeatures2d::SurfFeatureDetector::create(2357);
    std::vector<cv::KeyPoint> key_points_surf;
    surf_detector->detect(image_gray, key_points_surf);
    cv::Mat output_surf;
    cv::drawKeypoints(image_gray, key_points_surf, output_surf);
    cv::imwrite("result_surf.jpg", output_surf);
    std::cout << key_points_surf.size() << " time: " << chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start_surf).count() / 1000 << "ms" << std::endl;
    
    return 0;
}
