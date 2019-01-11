#ifndef MSCKF_H
#define MSCKF_H

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>


using namespace std;
using namespace Eigen;

typedef long long int StateIDType;

struct CAM_state{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double time;
    StateIDType cam_state_id;
    

    Eigen::Quaterniond quaternion;  //q_CG
    Eigen::Vector3d position;
};

typedef std::map<StateIDType, CAM_state, std::less<int>,
        Eigen::aligned_allocator< std::pair<const StateIDType, CAM_state> > > CamStateServer;

typedef long long int FeatureIDType;

struct IMU_state{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    //数据结构内存对齐

    double time;
    StateIDType imu_state_id;
    static StateIDType next_id;

    Eigen::Quaterniond quaternion;    //world to body
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    Eigen::Vector3d T_cam0_imu;
    Eigen::Matrix3d R_imu_cam0;

    // static double gyro_noise;
    // static double acc_noise;
    // static double gyro_bias_noise;
    // static double acc_bias_noise;
    Eigen::Quaterniond quaternion_null;
    Eigen::Vector3d position_null;
    Eigen::Vector3d velocity_null;
    
    double g;
    //static Eigen::Isometry3d T_imu_body;
};

typedef long long int FeatureIDType;
struct Feature{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

  FeatureIDType id;

  // id for next feature
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector2d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector2d> > > observations;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

    // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}
};

typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<std::pair<const FeatureIDType, Feature> > > MapServer;


//OptimizationConfig Configuration parameters for 3d feature position optimization.
struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

      OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {return;}
      };

class Msckf {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Msckf();

        ~Msckf(); 

        void Initialize(void);

        void InitGravityBias(void);

        void PropagateMsckfState(const double& time,const Vector3d& gyro, const Vector3d& acc);

        void AugmentState(const double& time);

        void AddFeatureObservations(const sensor_msgs::PointCloudConstPtr &feature_msg);

        void Marginalize(void);

        void PruneCamStateBuffer(void);

        void MeasurementUpdate(const MatrixXd H,const VectorXd r);

        void OnlineReset(void);

        std::vector<sensor_msgs::Imu> imu_msg_buffer;

        IMU_state imu_state;
        CamStateServer cam_states;

        Isometry3d T_imu_body;

    private:
        
    struct StateServer {
      IMU_state imu_state;
      CamStateServer cam_states;

      // State covariance matrix
      Eigen::MatrixXd state_cov;
      Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };

        void PredictImuState(const double& dt,const Vector3d& gyro,const Vector3d& acc);        

        bool MahaGatingTest(const MatrixXd& H, const VectorXd& r, const int& dof);

        bool CheckMotion( const Feature feature,
                          const CamStateServer& cam_states );

        bool FeatureInitPosition( Feature& feature,const CamStateServer& cam_states);                          

        void GenerateInitGuess( const Isometry3d& T_c1_c2, const Vector2d& z1,
                                const Vector2d& z2, Vector3d& p );

        void FeatureCost(const Isometry3d& T_c0_ci,const Vector3d& x, 
                         const Vector2d& z, double& e);
                         
        void FindRedundantCamStates( vector<StateIDType>& rm_cam_state_ids );

        void Jacobian(const Isometry3d& T_c0_ci,const Vector3d& x,const Vector2d& z,
                      Matrix<double, 2, 3>& J, Vector2d& r, double& w);  

        void FeatureJacobian( const FeatureIDType feature_id, 
                      const vector<StateIDType>& cam_state_ids, MatrixXd& H_x, VectorXd& r);  

        void MeasurementJacobian( const StateIDType& cam_state_id, const FeatureIDType& feature_id,
                      Matrix<double, 2, 6>& H_x, Matrix<double, 2, 3>& H_f, Vector2d& r);

        //double observation_noise;

        // chi squared test table.
        static std::map<int, double> chi_squared_test_table;

        MapServer map_server;

        Eigen::Vector3d Gravity;

        double tracking_rate;
        //double tracked_feature_num;
        unsigned int max_cam_state_size;

        OptimizationConfig optimization_config;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;

};

#endif