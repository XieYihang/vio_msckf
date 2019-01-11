#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include "ros/ros.h"

using namespace std;
using namespace Eigen;

extern cv_bridge::CvImageConstPtr cam0_cur_img_ptr;

extern cv::Ptr<cv::Feature2D> detector_ptr;
extern vector<cv::Point2f> prev_features, cur_features,track_features;
extern vector<cv::Point2f> cur_un_pts;
extern vector<int> track_cnt;
extern vector<int> features_id;

extern int image_cols;
extern int image_rows;
extern double disparity;

void init_first_frame();

void track_feature();

void add_new_feature();

void undistortedPoints(const vector<cv::Point2f>& pts_in, vector<cv::Point2f>& pts_out);

