#ifndef VIO_MATH_H
#define VIO_MATH_H

#include <cmath>
#include <Eigen/Dense>

Eigen::Matrix3d crossMat(const Eigen::Vector3d w)
{
    Eigen::Matrix3d w_hat;
    w_hat(0,0) = 0;
    w_hat(0,1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = 0;
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = 0;

    return w_hat;
}

Eigen::Vector4d smallAngleQuaternion(const Eigen::Vector3d& dtheta)
{
    Eigen::Vector3d dq = dtheta/2.0;
    Eigen::Vector4d q;
    double dq_square_norm = dq.squaredNorm();

    if( dq_square_norm <= 1)
    {
        q.head<3>() = dq;
        q(3) = std::sqrt(1-dq_square_norm);
    }
    else
    {
        q.head<3>() = dq;
        q(3) = 1;
        q = q / std::sqrt(1+dq_square_norm);
    }

    return q;
}

// Eigen::Quaterniond RotToQua( const Eigen::Vector3d rotVec)
// {
//     double veclen = rotVec.norm();

//     // if()

// }

#endif