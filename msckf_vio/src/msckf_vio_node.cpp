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

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>


// #include <tf/transform_broadcaster.h>
// #include <eigen_conversions/eigen_msg.h>
// #include <tf_conversions/tf_eigen.h>

#include <fstream>
#include <eigen3/Eigen/Dense>

#include <msckf_vio/msckf_vio.h>

using namespace std;
using namespace Eigen;

ros::Publisher odom_pub;
ros::Publisher feature_pub;
ros::Publisher pub_est_path;
nav_msgs::Path est_path;

Msckf msckf_est;

static bool is_gravity_init = false;
static bool is_first_img = true;

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!is_gravity_init) return;;

    if (is_first_img) {
        is_first_img = false;
        msckf_est.imu_msg_buffer.clear();
        msckf_est.imu_state.time = feature_msg->header.stamp.toSec();
    }

    int used_imu_msg_cntr = 0;
    double time_bound = feature_msg->header.stamp.toSec();
    
    ROS_INFO("feature_size: %d",feature_msg->points.size());
    //ROS_INFO("time_bound: %d",time_bound);    
    for( const auto& imu_msg : msckf_est.imu_msg_buffer )
    {
        double imu_time = imu_msg.header.stamp.toSec();
        if (imu_time < msckf_est.imu_state.time) {
            ++used_imu_msg_cntr;
            continue;
        }
        if (imu_time > time_bound) break;

        Vector3d m_gyro, m_acc;
        m_gyro(0) = imu_msg.angular_velocity.x;
        m_gyro(1) = imu_msg.angular_velocity.y;
        m_gyro(2) = imu_msg.angular_velocity.z;

        m_acc(0) = imu_msg.linear_acceleration.x;
        m_acc(1) = imu_msg.linear_acceleration.y;
        m_acc(2) = imu_msg.linear_acceleration.z;
                     
        //IMU 捷联递推
        msckf_est.PropagateMsckfState( imu_time, m_gyro, m_acc );

        ++used_imu_msg_cntr;
    }

    // ROS_INFO("pos: %f %f %f",msckf_est.imu_state.position.x(),
    //             msckf_est.imu_state.position.y(),
    //             msckf_est.imu_state.position.z());

    msckf_est.imu_msg_buffer.erase(
        msckf_est.imu_msg_buffer.begin(),msckf_est.imu_msg_buffer.begin()+used_imu_msg_cntr );

    msckf_est.imu_state.imu_state_id = msckf_est.imu_state.next_id++;
    
    //augment the states
    double cur_time = feature_msg->header.stamp.toSec();
    msckf_est.AugmentState(cur_time);

    //添加新的观测量
    msckf_est.AddFeatureObservations(feature_msg);

    //利用丢失的特征点更新 marginalized
    msckf_est.Marginalize();

    //更新cam-state buffer
    msckf_est.PruneCamStateBuffer();

    //Reset the system if necessary.
    msckf_est.OnlineReset();

    //发布信息
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = feature_msg->header.stamp;
    odom_msg.header.frame_id = "world";
    odom_msg.child_frame_id = "robot";

    odom_msg.pose.pose.position.x = msckf_est.imu_state.position.x();
    odom_msg.pose.pose.position.y = msckf_est.imu_state.position.y();
    odom_msg.pose.pose.position.z = msckf_est.imu_state.position.z();
    odom_msg.pose.pose.orientation.x = msckf_est.imu_state.quaternion.x();
    odom_msg.pose.pose.orientation.y = msckf_est.imu_state.quaternion.y();
    odom_msg.pose.pose.orientation.z = msckf_est.imu_state.quaternion.z();
    odom_msg.pose.pose.orientation.w = msckf_est.imu_state.quaternion.w();
    odom_msg.twist.twist.linear.x = msckf_est.imu_state.velocity.x();
    odom_msg.twist.twist.linear.y = msckf_est.imu_state.velocity.y();
    odom_msg.twist.twist.linear.z = msckf_est.imu_state.velocity.z();

    odom_pub.publish(odom_msg);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = feature_msg->header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odom_msg.pose.pose;
    est_path.header = feature_msg->header;
    est_path.header.frame_id = "world";
    est_path.poses.push_back(pose_stamped);
    pub_est_path.publish(est_path);

}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    
    msckf_est.imu_msg_buffer.push_back(*imu_msg);

    if(!is_gravity_init)
    {
        if(msckf_est.imu_msg_buffer.size()<200) return;
        msckf_est.InitGravityBias();
        is_gravity_init = true;
        //
    }

    //ROS_INFO("imu_buffer_len: %d",msckf_est.imu_msg_buffer.size());
    return;
}

int main(int argc, char *argv[])
{
    ros::init(argc,argv,"vio_estimator");
    ros::NodeHandle est_node("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    ros::Subscriber imu_sub = est_node.subscribe("/imu0",50,imu_callback);
    ros::Subscriber feature_sub = est_node.subscribe("/feature",50,feature_callback);

    pub_est_path = est_node.advertise<nav_msgs::Path>("est_path",1000);
    odom_pub = est_node.advertise<nav_msgs::Odometry>("est_odom", 100);
    feature_pub = est_node.advertise<sensor_msgs::PointCloud>("feature_point_cloud", 100);

    ros::spin();

    return 0;

}