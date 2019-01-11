#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <message_filters/subscriber.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>

#include <fstream>
#include <eigen3/Eigen/Dense>

#include <image_processor/image_processor.h>

using namespace std;
using namespace Eigen;

#define WINDOW_SIZE 20

struct Data
{
    Data(FILE *f)
    {
        fscanf(f, " %lf,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &t,
               &px, &py, &pz,
               &qw, &qx, &qy, &qz,
               &vx, &vy, &vz,
               &wx, &wy, &wz,
               &ax, &ay, &az);
        t /= 1e9;
    }
    double t;
    float px,py,pz;
    float qw,qx,qy,qz;
    float vx,vy,vz;
    float wx,wy,wz;
    float ax,ay,az;
};
int idx = 1;
vector<Data> benchmark;

ros::Publisher pub_odom;
ros::Publisher pub_path;
ros::Publisher pub_feature,pub_match;

nav_msgs::Path path;
geometry_msgs::PoseStamped pose_stamped;

Quaterniond baseRgt = {1,0,0,0};
Vector3d baseTgt = {0,0,0};

long temptime = 0;
char base_name[256];
string str;



int first_img = 0;
bool original_pos_flag = false;
bool pub_flag = false;
Vector3d original_pos = Vector3d::Zero();

void imageCallback(const sensor_msgs::ImageConstPtr& img_msg)
{
    //cout<<"callback ok"<<endl;
    cam0_cur_img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    if(!first_img)
    {
        init_first_frame();
        first_img++;
    }
    else
    {
         track_feature();

         add_new_feature();        
    }

    undistortedPoints(cur_features, cur_un_pts);

    if( disparity < 20 && pub_flag==false )
    {
            ROS_INFO("disparity: %f pos: %f %f %f",disparity,pose_stamped.pose.position.x
    ,pose_stamped.pose.position.y,pose_stamped.pose.position.z);
    }


    if( disparity > 5&& pub_flag==false )
    {
        pub_flag = true;
    }


    if( pub_flag == true)
    {
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        for(unsigned int i= 0;i<features_id.size();i++)
        {
            if (track_cnt[i] > 1)
            {
                
                int p_id = features_id[i];
                geometry_msgs::Point32 p;
                p.x = cur_un_pts[i].x;
                p.y = cur_un_pts[i].y;
                p.z = 1;   
                //ROS_INFO("p_id: %d",p_id);
                feature_points->points.push_back(p);
                id_of_point.values.push_back(p_id);
                u_of_point.values.push_back(cur_features[i].x);
                v_of_point.values.push_back(cur_features[i].y); 
            }    
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        //ROS_INFO("feature_size: %d",feature_points->points.size());
        pub_feature.publish(feature_points);

    }

    //show
    cv_bridge::CvImageConstPtr ptr;
    ptr = cv_bridge::toCvCopy(img_msg,sensor_msgs::image_encodings::MONO8);
    ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
    cv::Mat tmp_img = ptr->image.rowRange(0,image_rows);

    
    for(unsigned int i = 0;i<cur_features.size();i++)
    {
        double len = std::min(1.0,1.0*track_cnt[i]/WINDOW_SIZE);
        cv::circle(tmp_img, cur_features[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    pub_match.publish(ptr->toImageMsg());

    // if( idx < static_cast<int>(benchmark.size()) )
    // {idx++;}
    // else
    // {idx = 1;}

    // for(;benchmark[idx].t < img_msg->header.stamp.toSec();idx++)
    // {
    //     nav_msgs::Odometry odometry;
    //     odometry.header.stamp = ros::Time();
    //     odometry.header.frame_id = "world";
    //     odometry.child_frame_id = "world";

    //     Vector3d tmp_T = baseTgt + baseRgt*Vector3d{benchmark[idx -1].px,benchmark[idx -1].py,benchmark[idx-1].pz};
    
    //     if(original_pos_flag==false)
    //     {
    //         original_pos_flag=true;
    //         original_pos = tmp_T;
    //     }
    //     odometry.pose.pose.position.x = tmp_T.x()-original_pos.x();
    //     odometry.pose.pose.position.y = tmp_T.y()-original_pos.y();
    //     odometry.pose.pose.position.z = tmp_T.z()-original_pos.z();

    //     Quaterniond tmp_R = baseRgt * Quaterniond{benchmark[idx - 1].qw,
    //                                             benchmark[idx - 1].qx,
    //                                             benchmark[idx - 1].qy,
    //                                             benchmark[idx - 1].qz};

    //     odometry.pose.pose.orientation.w = tmp_R.w();
    //     odometry.pose.pose.orientation.x = tmp_R.x();
    //     odometry.pose.pose.orientation.y = tmp_R.y();
    //     odometry.pose.pose.orientation.z = tmp_R.z();

    //     Vector3d tmp_V = baseRgt * Vector3d{benchmark[idx -1].vx,
    //                                         benchmark[idx -1].vy,
    //                                         benchmark[idx -1].vz};

    //     odometry.twist.twist.linear.x = tmp_V.x();
    //     odometry.twist.twist.linear.y = tmp_V.y();
    //     odometry.twist.twist.linear.z = tmp_V.z();
    //     pub_odom.publish(odometry);

    //     geometry_msgs::PoseStamped pose_stamped;
    //     pose_stamped.header = odometry.header;
    //     pose_stamped.pose = odometry.pose.pose;
    //     path.header = odometry.header;
    //     path.poses.push_back(pose_stamped);
    //     pub_path.publish(path);

    // }
    
}

void poseCallback(const geometry_msgs::PointStamped& pos_msg)
{
//     nav_msgs::Odometry odometry;
//     odometry.header.stamp = ros::Time();
//     odometry.header.frame_id = "world";
//     odometry.child_frame_id = "world";

        if(original_pos_flag==false)
        {
            original_pos_flag=true;
            original_pos(0) = pos_msg.point.x;
            original_pos(1) = pos_msg.point.y;
            original_pos(2) = pos_msg.point.z;
        }

     pose_stamped.header = pos_msg.header;
     pose_stamped.header.frame_id = "world";
     pose_stamped.pose.position.x = pos_msg.point.x - original_pos(0);
     pose_stamped.pose.position.y = pos_msg.point.y - original_pos(1);
     pose_stamped.pose.position.z = pos_msg.point.z - original_pos(2);
     path.header = pose_stamped.header;
     path.poses.push_back(pose_stamped);
     pub_path.publish(path);
}

int main(int argc, char *argv[])
{
    /* code for main function */
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;

    //cv::destroyWindow("view");
    // ros::Rate loop_rate(10);

    //string csv_file = readParam<string>(n, "data_name");
    //string csv_file = "/home/levy/myslam/mono_vio_1.0/src/config/MH_05_difficult/data.csv";
    string csv_file = "/home/levy/myslam/mono_vio_1.0/src/config/MH_01_easy/data.csv";
    std::cout<<"load ground truth"<< csv_file<<std::endl;
    FILE *f = fopen(csv_file.c_str(),"r");
    if(f==NULL)
    {
        std::cout<<"can't load ground truth;wrong path"<<std::endl;
        return 0;
    }
    ROS_INFO("===========================================");
    char tmp[10000];
    fgets(tmp,10000,f); //读入字符串
    while(!feof(f))     //判断指针是否到达文件尾部
        benchmark.emplace_back(f);  //类似push_back 效率更高
    fclose(f);
    benchmark.pop_back();   //删除容器尾端的数据

    ROS_INFO("Data loaded: %d",(int)benchmark.size());
    pub_odom = nh.advertise<nav_msgs::Odometry>("odometry",1000);
    pub_path = nh.advertise<nav_msgs::Path>("path",1000);
    pub_match = nh.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_feature = nh.advertise<sensor_msgs::PointCloud>("feature",100);

    ros::Subscriber  sub = nh.subscribe("/cam0/image_raw",1,imageCallback);
    ros::Subscriber  gt_sub = nh.subscribe("/leica/position",100,poseCallback);
 
    ros::spin();

    // while (ros::ok())
    // {
    //     /* code for loop body */
    //     std_msgs::String msg;
    //     std::stringstream ss;
    //     ss<<"Hello world";
    //     msg.data = ss.str();
    //     chatter_pub.publish(msg);
    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }
    
    
    return 0;
}
