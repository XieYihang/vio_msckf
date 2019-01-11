#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/vio_math.h>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
//#include <Eigen/SPQRSupport>
 #include <Eigen/SparseQR>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#define GRAVITY_ACCELERATION 9.81

static double gyro_noise = 0.001;
static double acc_noise = 0.01;
static double gyro_bias_noise = 0.001;
static double acc_bias_noise = 0.01;

static double pos_std_threshold = 8.0;
static double observation_noise = 0.01;

StateIDType IMU_state::next_id = 0;

//Isometry3d IMU_state::T_imu_body = Isometry3d::Identity();

FeatureIDType Feature::next_id = 0;

map<int, double> Msckf::chi_squared_test_table;

Msckf::Msckf()
{
    Initialize();
}

Msckf::~Msckf()
{}

void Msckf::Initialize(void)
{

    optimization_config.translation_threshold = 0.2;

    ROS_INFO("huber_epsilon: %f", optimization_config.huber_epsilon);

    gyro_noise *= gyro_noise;
    gyro_bias_noise *= gyro_bias_noise;
    acc_noise *= acc_noise;
    acc_bias_noise *= acc_bias_noise;

    observation_noise *=observation_noise;

    // Initialize state server
    continuous_noise_cov = Matrix<double, 12, 12>::Zero();
    continuous_noise_cov.block<3, 3>(0, 0) = Matrix3d::Identity()*gyro_noise;
    continuous_noise_cov.block<3, 3>(3, 3) = Matrix3d::Identity()*gyro_bias_noise;
    continuous_noise_cov.block<3, 3>(6, 6) = Matrix3d::Identity()*acc_noise;
    continuous_noise_cov.block<3, 3>(9, 9) = Matrix3d::Identity()*acc_bias_noise;

    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    velocity_cov = 0.25;
    gyro_bias_cov = 1e-4;
    acc_bias_cov = 1e-2;

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    extrinsic_rotation_cov = 3.0462e-4;
    extrinsic_translation_cov = 1e-4;
    
    state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_cov(i, i) = extrinsic_translation_cov;

    Gravity = Vector3d(0, 0, -9.8);

    T_imu_body = Isometry3d::Identity();

    imu_state.next_id = 0;
    imu_state.imu_state_id = 0;
    imu_state.time = 0.0;
    imu_state.quaternion = Quaterniond(0.0, 0.0, 0.0, 1.0);
    imu_state.position = Vector3d::Zero();
    imu_state.velocity = Vector3d::Zero();
    imu_state.gyro_bias = Vector3d::Zero();
    imu_state.acc_bias = Vector3d::Zero();

    imu_state.quaternion_null = imu_state.quaternion;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    Eigen::Matrix<double, 4, 4> T_cam_imu;
    T_cam_imu << 0.0148655429818, -0.999880929698,   0.00414029679422, -0.021640145497,
                 0.999557249008,   0.0149672133247,  0.025715529948,   -0.064676986768,
                -0.0257744366974,  0.00375618835797, 0.999660727178,    0.009810730590,
                 0.0,              0.0,              0.0,               1.000000000000;

    imu_state.T_cam0_imu = T_cam_imu.block<3,1>(0,3);
    imu_state.R_imu_cam0 = T_cam_imu.block<3,3>(0,0).transpose();

    cam_states.clear();
    map_server.clear();
    imu_msg_buffer.clear();

    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_test_table[i] = boost::math::quantile(chi_squared_dist, 0.05);
    }

    max_cam_state_size = 30;

    ROS_INFO("msckf init ok");
    // return true;
}

void Msckf::InitGravityBias(void)
{
    Vector3d sum_gyro = Vector3d::Zero();
    Vector3d sum_acc = Vector3d::Zero();

    Vector3d gyro = Vector3d::Zero();
    Vector3d acc = Vector3d::Zero();

    for(auto& imu_msg : imu_msg_buffer)
    {
        gyro(0) = imu_msg.angular_velocity.x;
        gyro(1) = imu_msg.angular_velocity.y;
        gyro(2) = imu_msg.angular_velocity.z;

        acc(0) = imu_msg.linear_acceleration.x;
        acc(1) = imu_msg.linear_acceleration.y;
        acc(2) = imu_msg.linear_acceleration.z;

        sum_gyro +=gyro;
        sum_acc +=acc;
    }

    imu_state.gyro_bias = sum_gyro / imu_msg_buffer.size();

    //起飞状态必须为水平
    Vector3d gravity_imu = sum_acc / imu_msg_buffer.size();
    double gravity_norm = gravity_imu.norm();
    Gravity = Vector3d(0.0, 0.0, -gravity_norm);
    
    Quaterniond q0_i_w = Quaterniond::FromTwoVectors(gravity_imu , -Gravity);
   
    imu_state.quaternion = q0_i_w.toRotationMatrix().transpose();   //w-to-b
    ROS_INFO("quat: %f %f %f %f",imu_state.quaternion.w(),
                            imu_state.quaternion.x(),
                            imu_state.quaternion.y(),
                            imu_state.quaternion.z());

     Vector3d w;                       
     w = imu_state.quaternion.toRotationMatrix().transpose()*gravity_imu;    
         ROS_INFO("w: %f %f %f",w(0),w(1),w(2)); 
         ROS_INFO("G: %f %f %f",Gravity(0),Gravity(1),Gravity(2));                       
    return;

}

void Msckf::PropagateMsckfState( const double& time,
    const Vector3d& gyro, const Vector3d& acc )
{
    Vector3d gyro_t = gyro - imu_state.gyro_bias;
    Vector3d acc_t = acc - imu_state.acc_bias;
    double dt = time - imu_state.time;
    
    Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
    Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();
    Matrix<double, 3, 3> R_wi = imu_state.quaternion.toRotationMatrix();

    //计算 误差状态矩阵F
    //matrix.block<i,j>(a,b): 矩阵（a,b）位置起始i×j矩阵
    F.block<3,3>(0,0) = -crossMat(gyro_t);
    F.block<3,3>(0,3) = -Matrix3d::Identity();
    F.block<3,3>(6,0) = -R_wi.transpose()*crossMat(acc_t);
    F.block<3,3>(6,9) = -R_wi.transpose();
    F.block<3,3>(12,6) = Matrix3d::Identity();

    //噪声矩阵计算
    G.block<3,3>(0,0) = -Matrix3d::Identity();
    G.block<3,3>(3,3) =  Matrix3d::Identity();
    G.block<3,3>(6,6) = -R_wi.transpose();
    G.block<3,3>(9,9) =  Matrix3d::Identity();

    // F = F*dt;
    // Matrix<double, 21, 21> Phi = F.exp();
    //Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() + F*dt;
    
    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s.
    Matrix<double, 21, 21> Fdt = F * dt;
    Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
    Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;


    // Propogate the state using 4th order Runge-Kutta
    PredictImuState(dt, gyro_t, acc_t);

    // Apply observability constraints - enforce nullspace of Phi
    // Ref: Observability-constrained Vision-aided Inertial Navigation, Hesch J.et al. Feb, 2012
    Matrix<double, 3, 3> R_kk_1(imu_state.quaternion_null);
    Phi.block<3,3>(0,0) = imu_state.quaternion.toRotationMatrix() * R_kk_1.transpose();

    Vector3d u = R_kk_1 * Gravity;
    RowVector3d s = (u.transpose() * u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3,3>(6,0);
    Vector3d tmp = imu_state.velocity_null-imu_state.velocity;
    Vector3d w1 = crossMat(tmp) * Gravity;
    Phi.block<3,3>(6,0) = A1 -(A1*u - w1)*s;

    Matrix3d A2 = Phi.block<3,3>(12,0);
    tmp = dt * imu_state.velocity_null + imu_state.position_null - imu_state.position;
    Vector3d w2 = crossMat(tmp)*Gravity;
    Phi.block<3,3>(12,0) = A2 - (A2*u - w2)*s;

    // Propogate the state covariance matrix.
    Matrix<double, 21, 21> Q = Phi*G*continuous_noise_cov*G.transpose()*Phi.transpose()*dt;
    state_cov.block<21, 21>(0, 0) = Phi*state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;

    // 更新imu部分协方差矩阵
    // Matrix<double, 21, 21> Q = G*continuous_noise_cov*G.transpose()*dt;
    // state_cov.block<21, 21>(0, 0) = Phi*state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;

    //更新cam-cam协方差 

    //更新imu-cam协方差 
    if( cam_states.size() > 0 )
    {
        state_cov.block( 0, 21, 21, state_cov.cols()-21) = 
        Phi * state_cov.block(0, 21, 21, state_cov.cols()-21);

        state_cov.block( 21, 0, state_cov.rows()-21, 21) =
        state_cov.block( 21, 0, state_cov.rows()-21, 21) * Phi.transpose();
    }
    
    //cov矩阵对称化 ，主对角线取绝对值，非对角线取平均值
    MatrixXd state_cov_fixed = (state_cov + state_cov.transpose()) / 2.0;
    state_cov = state_cov_fixed;

    imu_state.quaternion_null = imu_state.quaternion;
    imu_state.velocity_null = imu_state.velocity;
    imu_state.position_null = imu_state.position;
    imu_state.time = time;

    return;
}

void Msckf::PredictImuState(const double& dt,
    const Vector3d& gyro,const Vector3d& acc)
{
  
    Matrix4d omega = Matrix4d::Zero();
    omega.block<3,3>(0,0) = -crossMat(gyro);
    omega.block<3,1>(0,3) =  gyro;
    omega.block<1,3>(3,0) = -gyro;

    Matrix4d omega_psi = 0.5 * omega;

    Vector4d y0, k0, k1, k2, k3, k4, k5, y_t;
    y0(0) = -imu_state.quaternion.x();
    y0(1) = -imu_state.quaternion.y();
    y0(2) = -imu_state.quaternion.z();
    y0(3) = imu_state.quaternion.w();

    k0 = omega_psi * (y0);
    k1 = omega_psi * (y0 + (k0 / 4.) * dt);
    k2 = omega_psi * (y0 + (k0 / 8. + k1 / 8.) * dt);
    k3 = omega_psi * (y0 + (-k1 / 2. + k2) * dt);
    k4 = omega_psi * (y0 + (k0 * 3. / 16. + k3 * 9. / 16.) * dt);
    k5 = omega_psi *
        (y0 +
        (-k0 * 3. / 7. + k1 * 2. / 7. + k2 * 12. / 7. - k3 * 12. / 7. + k4 * 8. / 7.) *
        dt);

    y_t = y0 + (7. * k0 + 32. * k2 + 12. * k3 + 32. * k4 + 7. * k5) * dt / 90.;

    Quaternion<double> q(y_t(3), -y_t(0), -y_t(1), -y_t(2));
    q.normalize();
 
    Vector3d delta_v_I_G = (((imu_state.quaternion.toRotationMatrix()).transpose()) * acc + Gravity) * dt;
    
    Vector3d pre_vel = imu_state.velocity;

    imu_state.quaternion = q;
    imu_state.velocity = imu_state.velocity + delta_v_I_G;
    imu_state.position = imu_state.position + pre_vel * dt;
    
    // double gyro_norm = gyro.norm();

    // Vector4d q;

    // q(0) = imu_state.quaternion.x();
    // q(1) = imu_state.quaternion.y();
    // q(2) = imu_state.quaternion.z();
    // q(3) = imu_state.quaternion.w();

    // Vector4d dq_dt, dq_dt2;
    // if (gyro_norm > 1e-5) {
    //     dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
    //     1/gyro_norm*sin(gyro_norm*dt*0.5)*omega) * q;
    //     dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
    //     1/gyro_norm*sin(gyro_norm*dt*0.25)*omega) * q;
    // }
    // else {
    //     dq_dt = (Matrix4d::Identity()+0.5*dt*omega) * cos(gyro_norm*dt*0.5) * q;
    //     dq_dt2 = (Matrix4d::Identity()+0.25*dt*omega) * cos(gyro_norm*dt*0.25) * q;
    // }
    // Eigen::Quaterniond q1,q2;
    // q1.w() = dq_dt(3);q1.x()=dq_dt(0);q1.y()=dq_dt(1);q1.z()=dq_dt(2);
    // q2.w() = dq_dt2(3);q2.x()=dq_dt2(0);q2.y()=dq_dt2(1);q2.z()=dq_dt2(2);
    
    // Matrix3d dR_dt_transpose = q1.toRotationMatrix().transpose();
    // Matrix3d dR_dt2_transpose = q2.toRotationMatrix().transpose();

    // Quaterniond& imu_q = imu_state.quaternion;
    // Vector3d& v = imu_state.velocity;
    // Vector3d& p = imu_state.position;

    // // k1 = f(tn, yn)
    // Vector3d k1_v_dot = R_iw*acc + Gravity;
    // Vector3d k1_p_dot = v;

    // Vector3d k1_v = v + k1_v_dot*dt/2;
    // Vector3d k2_v_dot = dR_dt2_transpose*acc + Gravity;
    // Vector3d k2_p_dot = k1_v;

    // Vector3d k2_v = v + k2_v_dot*dt/2;
    // Vector3d k3_v_dot = dR_dt2_transpose*acc + Gravity;
    // Vector3d k3_p_dot = k2_v;

    // Vector3d k3_v = v + k3_v_dot*dt;
    // Vector3d k4_v_dot = dR_dt_transpose*acc + Gravity;
    // Vector3d k4_p_dot = k3_v;

    // imu_q = q1.normalized();
    // //v = v + (R_iw*acc + Gravity)*dt/2;

    // // Vector3d w;                       
    // //  w = (R_iw*acc + Gravity)*dt/2;    
    // //      ROS_INFO("w: %f %f %f dt: %f",w(0),w(1),w(2),dt); 

    // v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
    // p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

    return;

}


void Msckf::AugmentState(const double& time)
{
    Matrix3d R_i_c = imu_state.R_imu_cam0;
    Vector3d T_c_i = imu_state.T_cam0_imu;

    // //由imu与camera的固连关系得到新相机位置和姿态,对应imu状态
    Matrix3d R_w_i = imu_state.quaternion.toRotationMatrix();
    Matrix3d R_w_c = R_i_c*R_w_i;
    Vector3d T_c_w = imu_state.position + R_w_i.inverse()*T_c_i;

    //cam_states[imu_state.id] = 
    CAM_state new_cam_state;
    new_cam_state.time = time;
    new_cam_state.quaternion = Quaterniond(R_w_c);
    new_cam_state.position = T_c_w;

    cam_states[imu_state.imu_state_id] = new_cam_state;

    //计算增广针对msckf状态的jacobin
    Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
    J.block<3, 3>(0, 0) = R_i_c;
    J.block<3, 3>(0, 15) = Matrix3d::Identity();
    J.block<3, 3>(3, 0) = crossMat(R_w_i.transpose()*T_c_i);
    J.block<3, 3>(3, 12) = Matrix3d::Identity();
    J.block<3, 3>(3, 18) = Matrix3d::Identity();

    //ROS_INFO("cam_state: %d",cam_states.size());
    
    // //构造带cam增广协方差矩阵
    size_t old_rows = state_cov.rows();
    size_t old_cols = state_cov.cols();
    state_cov.conservativeResize(old_rows+6,old_cols+6);
    //ROS_INFO("state_cov: %d %d",state_cov.rows(),state_cov.cols());

    const Matrix<double,21,21>& P11 = state_cov.block<21, 21>(0, 0);
    const MatrixXd& P12 = state_cov.block(0, 21, 21, old_cols-21);

    state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
    state_cov.block(0, old_cols, old_rows, 6) = state_cov.block(old_rows, 0, 6, old_cols).transpose();
    state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose();

    //cov矩阵对称化 ，主对角线取绝对值，非对角线取平均值
    MatrixXd state_cov_fixed = (state_cov + state_cov.transpose()) / 2.0;
    state_cov = state_cov_fixed;

    return;
}

void Msckf::MeasurementUpdate(const MatrixXd H,const VectorXd r)
{
    if(H.rows()==0||r.rows()==0) return;

    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    //SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    // SPQR<SparseMatrix<double> > spqr_helper;
    // spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    // spqr_helper.compute(H_sparse);

    // MatrixXd H_temp;
    // VectorXd r_temp;
    // (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    // (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    // H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    // r_thin = r_temp.head(21+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;

    // Put residuals in update-worthy form
    // Calculates T_H matrix according to Mourikis 2007

    // HouseholderQR<MatrixXd> qr(H_o);
    // MatrixXd Q = qr.householderQ();
    // MatrixXd R = qr.matrixQR().template triangularView<Upper>();

    // VectorXd nonZeroRows = R.rowwise().any();
    // int numNonZeroRows = nonZeroRows.sum();

    // T_H = MatrixXd::Zero(numNonZeroRows, R.cols());
    // Q_1 = MatrixXd::Zero(Q.rows(), numNonZeroRows);
          

        H_thin = H;
        r_thin = r;
    } else {
        H_thin = H;
        r_thin = r;
    }

    //计算 kalman增益
    const MatrixXd& P = state_cov;
    MatrixXd S = H_thin*P*H_thin.transpose() + observation_noise*MatrixXd::Identity(H_thin.rows(),H_thin.rows());
    MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
    MatrixXd K = K_transpose.transpose();

    //计算error
    VectorXd delta_x = K*r_thin;

    const VectorXd& delta_x_imu = delta_x.head<21>();

    if ( delta_x_imu.segment<3>(6).norm() > 0.5 || delta_x_imu.segment<3>(12).norm() > 1.0)
    {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        ROS_WARN("Update change is too large.");
        //return;
    }

    //!!!!
    const Vector4d dq_imu = smallAngleQuaternion(delta_x_imu.head<3>());
    Quaterniond dq_q;
    dq_q.w() = dq_imu(3);
    dq_q.x() = dq_imu(0);
    dq_q.y() = dq_imu(1);
    dq_q.z() = dq_imu(2);
    imu_state.quaternion = dq_q*imu_state.quaternion;
    imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    imu_state.velocity += delta_x_imu.segment<3>(6);
    imu_state.acc_bias += delta_x_imu.segment<3>(9);
    imu_state.position += delta_x_imu.segment<3>(12);

    const Vector4d dq_extrinsic = smallAngleQuaternion(delta_x_imu.segment<3>(15));
    Quaterniond dq_e;
    dq_e.w() = dq_extrinsic(3);
    dq_e.x() = dq_extrinsic(0);
    dq_e.y() = dq_extrinsic(1);
    dq_e.z() = dq_extrinsic(2);
    imu_state.R_imu_cam0 = dq_e.toRotationMatrix()*imu_state.R_imu_cam0;
    imu_state.T_cam0_imu += delta_x_imu.segment<3>(18);

    //update camera states
    auto cam_state_iter = cam_states.begin();
    for(unsigned int i = 0; i<cam_states.size(); i++,cam_state_iter++)
    {
        const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        Quaterniond dq_c;
        dq_c.w() = dq_cam(3);
        dq_c.x() = dq_cam(0);
        dq_c.y() = dq_cam(1);
        dq_c.z() = dq_cam(2);
        cam_state_iter->second.quaternion = dq_c*cam_state_iter->second.quaternion;
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    //update cov
    MatrixXd I_KH = MatrixXd::Identity(K.rows(),H_thin.cols()) - K*H_thin;
    state_cov = I_KH*state_cov;

    MatrixXd state_cov_fixed = (state_cov + state_cov.transpose()) / 2.0;
    state_cov = state_cov_fixed;

    return;

}

bool Msckf::MahaGatingTest(const MatrixXd& H, const VectorXd& r, const int& dof)
{
    //Mahalanobis Gating 

    MatrixXd P1 = H*state_cov*H.transpose();
    MatrixXd P2 = observation_noise*MatrixXd::Identity(H.rows(),H.rows());

    double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

    if(gamma < chi_squared_test_table[dof])
    {
        return true;
    }else{
        return false;
    }
}

void Msckf::AddFeatureObservations(const sensor_msgs::PointCloudConstPtr &feature_msg)
{

    StateIDType state_id = imu_state.imu_state_id;
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    for(unsigned int i=0;i<feature_msg->points.size();i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        double p_u = feature_msg->points[i].x;
        double p_v = feature_msg->points[i].y;

        //ROS_INFO_STREAM_ONCE("u v: "<<p_u);
        if(map_server.find(feature_id) == map_server.end())
        {
            map_server[feature_id] = Feature(feature_id);
            map_server[feature_id].observations[state_id] = Vector2d(p_u, p_v);
        }
        else
        {
            map_server[feature_id].observations[state_id] = Vector2d(p_u, p_v);
            tracked_feature_num++;
        }
    }

    tracking_rate = static_cast<double>(tracked_feature_num)/static_cast<double>(curr_feature_num);
    //ROS_INFO("tracking_rate: %f %d %d",tracking_rate,tracked_feature_num,curr_feature_num);
    return;

}

//remove lost features use them to update the camera states that observed them
void Msckf::Marginalize(void)
{
    int jacobian_row_size = 0;
    vector<FeatureIDType>  invalid_feature_ids(0);
    vector<FeatureIDType>  processed_feature_ids(0);

    for(auto iter = map_server.begin(); iter != map_server.end(); iter++)
    {
        // Rename the feature to be checked.
        auto& feature = iter->second;

        // Pass the features that are still being tracked.
        if ( feature.observations.find(imu_state.imu_state_id) != feature.observations.end() )
            continue;

        if (feature.observations.size() < 3) 
        {
            invalid_feature_ids.push_back(feature.id);
            continue; 
        }

        //如果特征点能被初始化，则初始化
        if( !feature.is_initialized )
        {
            if( !CheckMotion(feature,cam_states) )
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }
            else
            {
                if( !FeatureInitPosition(feature,cam_states) )
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;    
                }
            }
        }  

        jacobian_row_size += 2*feature.observations.size() - 3;
        processed_feature_ids.push_back(feature.id);
    }    

    for(const auto& feature_id : invalid_feature_ids)
        map_server.erase(feature_id);
    ROS_INFO("processed_feature_ids: %d",processed_feature_ids.size());    
    //return if there is no lost feature to be processed
    if(processed_feature_ids.size() == 0) return;

    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21+6*cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // 处理丢失的特征点，作量测更新
    for(const auto& feature_id:processed_feature_ids)
    {
        auto& feature = map_server[feature_id];

        vector<StateIDType> cam_state_ids(0);
        for (const auto& measurement : feature.observations)
            cam_state_ids.push_back(measurement.first);

        MatrixXd H_xj;
        VectorXd r_j;
        FeatureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        if( MahaGatingTest(H_xj,r_j,cam_state_ids.size()-1) )
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();        
        }

        //限制量测矩阵大小
        if (stack_cntr > 1500) break;
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    MeasurementUpdate(H_x, r);

    // Remove all processed features from the map.
    for (const auto& feature_id : processed_feature_ids)
        map_server.erase(feature_id);

    return;
}

bool Msckf::CheckMotion( const Feature feature,
                    const CamStateServer& cam_states)
{
    const StateIDType& first_cam_id = feature.observations.begin()->first;
    const StateIDType& last_cam_id = (--feature.observations.end())->first;

    Eigen::Isometry3d first_cam_pose;
    Eigen::Isometry3d last_cam_pose;

    first_cam_pose.linear() = cam_states.find(first_cam_id)->second.quaternion.toRotationMatrix().transpose();
    first_cam_pose.translation() = cam_states.find(first_cam_id)->second.position;

    last_cam_pose.linear() = cam_states.find(last_cam_id)->second.quaternion.toRotationMatrix().transpose();
    last_cam_pose.translation() = cam_states.find(last_cam_id)->second.position;

    // Get the direction of the feature when it is first observed.
    // This direction is represented in the world frame.
    Eigen::Vector3d feature_direction(
        feature.observations.begin()->second(0),
        feature.observations.begin()->second(1), 1.0);
    feature_direction = feature_direction / feature_direction.norm();
    feature_direction = first_cam_pose.linear()*feature_direction;

    // Compute the translation between the first frame
    // and the last frame. We assume the first frame and
    // the last frame will provide the largest motion to
    // speed up the checking process.
    Eigen::Vector3d translation = last_cam_pose.translation() - first_cam_pose.translation();

    double parallel_translation = translation.transpose()*feature_direction;

    Eigen::Vector3d orthogonal_translation = translation - parallel_translation*feature_direction;

    if (orthogonal_translation.norm() > optimization_config.translation_threshold)
        return true;
    else 
        return false;
}

bool Msckf::FeatureInitPosition( Feature& feature,const CamStateServer& cam_states)
{
    std::vector<Eigen::Isometry3d,
        Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
    std::vector<Eigen::Vector2d,
        Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);   

    for(auto& m : feature.observations)
    {
        auto cam_state_iter = cam_states.find(m.first);
        if(cam_state_iter == cam_states.end() )continue;

        // Add the measurement.
        measurements.push_back(m.second.head<2>());
        measurements.push_back(m.second.tail<2>());

        // This camera pose will take a vector from this camera frame
        // to the world frame.
        Eigen::Isometry3d cam0_pose;
        cam0_pose.linear() = cam_state_iter->second.quaternion.toRotationMatrix().transpose();
        cam0_pose.translation() = cam_state_iter->second.position;

        cam_poses.push_back(cam0_pose);
    }

    // All camera poses should be modified such that it takes a
    // vector from the first camera frame in the buffer to this camera frame.
    Eigen::Isometry3d T_c0_w = cam_poses[0];
    for (auto& pose : cam_poses)
        pose = pose.inverse() * T_c0_w;

    // Generate initial guess
    Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
     GenerateInitGuess(cam_poses[cam_poses.size()-1], measurements[0],
         measurements[measurements.size()-1], initial_position);
    Eigen::Vector3d solution(
        initial_position(0)/initial_position(2),
        initial_position(1)/initial_position(2),
        1.0/initial_position(2));

    //Apply Levenberg-Marquart method to solve for the 3d position.
    double lambda = optimization_config.initial_damping;
    int inner_loop_cntr = 0;
    int outer_loop_cntr = 0;
    bool is_cost_reduced = false;
    double delta_norm = 0;

    //计算初始cost
    double total_cost = 0.0;
    for(unsigned int i = 0;i<cam_poses.size();i++)
    {
        double this_cost = 0.0;
        FeatureCost(cam_poses[i], solution, measurements[i], this_cost);
        total_cost += this_cost;    
    }

    //outer loop
    do{
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();

        for(unsigned int i = 0;i<cam_poses.size();i++)
        {
            Matrix<double, 2, 3> J;
            Vector2d r;
            double w;

            Jacobian(cam_poses[i], solution, measurements[i], J, r, w);

            if (w == 1) {
                A += J.transpose() * J;
                b += J.transpose() * r;
            } else {
                double w_square = w * w;
                A += w_square * J.transpose() * J;
                b += w_square * J.transpose() * r;
            }
        }

        //inner loop solve for the delta that can reduce the total cost
        do
        {
            Matrix3d damper = lambda * Matrix3d::Identity();
            Vector3d delta = (A + damper).ldlt().solve(b);
            Vector3d new_solution = solution - delta;
            delta_norm = delta.norm();

            double new_cost = 0.0;
            for(unsigned int i = 0;i<cam_poses.size();i++)
            {
                double this_cost = 0;
                FeatureCost(cam_poses[i], new_solution, measurements[i], this_cost);
                new_cost += this_cost;
            }

            if(new_cost < total_cost)
            {
                is_cost_reduced = true;
                solution = new_solution;
                total_cost = new_cost;
                lambda = lambda/10 > 1e-10 ? lambda/10:1e-10;
            }
            else
            {
                is_cost_reduced = false;
                lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;    
            }

        }while(inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

        inner_loop_cntr = 0;
    }
    while(outer_loop_cntr++ < optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

    ROS_INFO("J: %f",total_cost);   
    
    Vector3d final_position(solution(0)/solution(2),solution(1)/solution(2), 1.0/solution(2));

    bool is_valid_solution = true;
    for(const auto& pose : cam_poses)
    {
        Eigen::Vector3d position = pose.linear()*final_position + pose.translation();
        if (position(2) <= 0) {
            is_valid_solution = false;
            break;
        }
    }

      // Convert the feature position to the world frame.
    feature.position = T_c0_w.linear()*final_position + T_c0_w.translation();

    if (is_valid_solution)
    feature.is_initialized = true;

    return is_valid_solution;

}

void Msckf::GenerateInitGuess( const Isometry3d& T_c1_c2, const Vector2d& z1,
                                const Vector2d& z2, Vector3d& p )
{
  // Construct a least square problem to solve the depth.
  Vector3d m = T_c1_c2.linear() * Vector3d(z1(0), z1(1), 1.0);

  Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;        
}

void Msckf::FeatureCost(const Isometry3d& T_c0_ci,const Vector3d& x, 
                         const Vector2d& z, double& e)
{
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Vector3d h = T_c0_ci.linear()*Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;    
}

void Msckf::Jacobian(const Isometry3d& T_c0_ci,const Vector3d& x,const Vector2d& z,
                      Matrix<double, 2, 3>& J, Vector2d& r, double& w)
{
    const double& alpha = x(0);
    const double& beta = x(1);
    const double& rho = x(2);

    Vector3d h = T_c0_ci.linear()*Vector3d(alpha,beta,1.0) + rho*T_c0_ci.translation();
    double& h1 = h(0);
    double& h2 = h(1);
    double& h3 = h(2);

      // Compute the Jacobian.
    Matrix3d W;
    W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
    W.rightCols<1>() = T_c0_ci.translation();

    J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
    J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

    // Compute the residual.
    Vector2d z_hat(h1/h3, h2/h3);
    r = z_hat - z;

    // Compute the weight based on the residual.
    double e = r.norm();
    if (e <= optimization_config.huber_epsilon)
        w = 1.0;
    else
        w = optimization_config.huber_epsilon / (2*e);

    return;
}

void Msckf::PruneCamStateBuffer(void)
{
    if( cam_states.size() < max_cam_state_size )
        return;

    vector<StateIDType> rm_cam_state_ids(0);
    FindRedundantCamStates( rm_cam_state_ids );

    int jacobian_row_size = 0;
    for(auto& item:map_server)
    {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for(const auto& cam_id : rm_cam_state_ids)
        {
            if( feature.observations.find(cam_id) != feature.observations.end() )
            {
                involved_cam_state_ids.push_back(cam_id);
            }
        }

        if (involved_cam_state_ids.size() == 0) continue;
        if (involved_cam_state_ids.size() == 1) {
        feature.observations.erase(involved_cam_state_ids[0]);
        continue;
        }

        if(!feature.is_initialized)
        {
            if( !CheckMotion(feature,cam_states) )
            {
                for(const auto& cam_id : involved_cam_state_ids)
                    feature.observations.erase(cam_id);
                continue;
            }   
            else
            {
                if( !FeatureInitPosition(feature,cam_states) )
                {
                    for (const auto& cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    continue;    
                }
            } 
        }

        jacobian_row_size += 2*involved_cam_state_ids.size() - 3;
    }
    
    //计算Jacobian Residual
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21+6*cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for(auto& item : map_server)
    {
        auto& feature = item.second;

        // Check how many camera states to be removed are associated
        // with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for(const auto& cam_id : rm_cam_state_ids)
        {
            if( feature.observations.find(cam_id) != feature.observations.end() )
                involved_cam_state_ids.push_back(cam_id);
        }

        if(involved_cam_state_ids.size() == 0) continue;

        MatrixXd H_xj;
        VectorXd r_j;

        FeatureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if( MahaGatingTest(H_xj,r_j,involved_cam_state_ids.size()) )
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();        
        }

        for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
    }
    ROS_INFO("stack_cntr: %d",stack_cntr);    
    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    MeasurementUpdate(H_x, r);

    for( const auto& cam_id : rm_cam_state_ids )
    {
        int cam_sequence = std::distance( cam_states.begin(),cam_states.find(cam_id) );
        int cam_state_start = 21+6*cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state covariance matrix. 
        if( cam_state_end < state_cov.rows() )
        {
            state_cov.block( cam_state_start, 0, state_cov.rows()-cam_state_end, state_cov.cols() )= 
                state_cov.block(cam_state_end, 0, state_cov.rows()-cam_state_end, state_cov.cols());
            state_cov.block( 0, cam_state_start, state_cov.rows(), state_cov.cols()-cam_state_end) =
                state_cov.block(0, cam_state_end, state_cov.rows(), state_cov.cols()-cam_state_end);   

            state_cov.conservativeResize( state_cov.rows()-6, state_cov.cols()-6 );          
        }
        else
        {
            state_cov.conservativeResize( state_cov.rows()-6, state_cov.cols()-6 ); 
        }

        // Remove this camera state in the state vector.
        cam_states.erase(cam_id);
    }

    return;
}

void Msckf::FindRedundantCamStates( vector<StateIDType>& rm_cam_state_ids )
{
    //移除两个cam状态

    if (cam_states.size() < 5) return;

    // Move the iterator to the key position.
    auto key_cam_state_iter = cam_states.end();
    for (int i = 0; i < 4; ++i)
        --key_cam_state_iter;
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;
    auto first_cam_state_iter = cam_states.begin();

    // Pose of the key camera state.
    const Vector3d key_position = key_cam_state_iter->second.position;
    const Matrix3d key_rotation = key_cam_state_iter->second.quaternion.toRotationMatrix();

    // Mark the camera states to be removed based on the
    // motion between states.
    for (int i = 0; i < 2; ++i) {
        const Vector3d position = cam_state_iter->second.position;
        const Matrix3d rotation = cam_state_iter->second.quaternion.toRotationMatrix();

        double distance = (position-key_position).norm();
        //double angle = cam_state_iter->second.quaternion.angularDistance(key_cam_state_iter->second.quaternion);
        double angle = AngleAxisd(rotation*key_rotation.transpose()).angle();

        //if (angle < 0.1745 && distance < 0.2 && tracking_rate > 0.5) {
        if (angle < 0.2618 && distance < 0.4 && tracking_rate > 0.5) {
        rm_cam_state_ids.push_back(cam_state_iter->first);
        ++cam_state_iter;
        } else {
        rm_cam_state_ids.push_back(first_cam_state_iter->first);
        ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

    return;
}

void Msckf::FeatureJacobian( const FeatureIDType feature_id, 
        const vector<StateIDType>& cam_state_ids, MatrixXd& H_x, VectorXd& r)
{
    const auto& feature = map_server[feature_id];

    vector<StateIDType> valid_cam_state_ids(0);

    for(const auto& cam_id : cam_state_ids)
    {
        if(feature.observations.find(cam_id) == feature.observations.end() )
            continue;
        
        valid_cam_state_ids.push_back(cam_id);
    }    

    int jacobian_row_size = 0;
    jacobian_row_size = 2*valid_cam_state_ids.size();

    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,21+cam_states.size()*6);
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (const auto& cam_id : valid_cam_state_ids) {
            Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
            Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
            Vector2d r_i = Vector2d::Zero();

            MeasurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

            auto cam_state_iter = cam_states.find(cam_id);
            int cam_state_cntr = std::distance(cam_states.begin(), cam_state_iter);

            // Stack the Jacobians.
            H_xj.block<2, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
            H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
            r_j.segment<2>(stack_cntr) = r_i;
            stack_cntr += 2;    
    }

     // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(jacobian_row_size - 3);

    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;

    return;

}

void Msckf::MeasurementJacobian(
    const StateIDType& cam_state_id, const FeatureIDType& feature_id,
    Matrix<double, 2, 6>& H_x, Matrix<double, 2, 3>& H_f, Vector2d& r) 
{
    // Prepare all the required data.
    const CAM_state& cur_cam_state = cam_states[cam_state_id];
    const Feature& feature = map_server[feature_id];

    // Cam0 pose.
    Matrix3d R_w_c0 = cur_cam_state.quaternion.toRotationMatrix();
    //quaternionToRotation(cam_state.orientation);
    const Vector3d& t_c0_w = cur_cam_state.position;

    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Vector3d& p_w = feature.position;
    const Vector2d& z = feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to
    // the cam0 and cam1 frame.
    Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);

    // Compute the Jacobians.
    Matrix<double, 2, 3> dz_dpc0 = Matrix<double, 2, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

    Matrix<double, 2, 6> A;
    A << dz_dpc0 * crossMat(p_c0), -dz_dpc0 * R_w_c0; 

    Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) = cur_cam_state.quaternion.toRotationMatrix()*Gravity;
    u.block<3, 1>(3, 0) = crossMat(p_w - cur_cam_state.position)*Gravity;

     H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
     H_f = -H_x.block<2, 3>(0, 3);

    // Compute the residual.
    r = z - Vector2d( p_c0(0)/p_c0(2), p_c0(1)/p_c0(2) );

    return;

}



void Msckf::OnlineReset(void)
{
    if( pos_std_threshold <= 0) return;
    //static long long int online_reset_counter = 0;

    double pos_x_std = std::sqrt( state_cov(12,12) );
    double pos_y_std = std::sqrt( state_cov(13,13) );
    double pos_z_std = std::sqrt( state_cov(14,14) );

    if( pos_x_std < pos_std_threshold && pos_y_std < pos_std_threshold && pos_z_std < pos_std_threshold)
        return;

    cam_states.clear();

    map_server.clear();

    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    velocity_cov = 0.25;
    gyro_bias_cov = 1e-4;
    acc_bias_cov = 1e-2;

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    extrinsic_rotation_cov = 3.0462e-4;
    extrinsic_translation_cov = 1e-4;
    
    state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_cov(i, i) = extrinsic_translation_cov;

    return;
}
