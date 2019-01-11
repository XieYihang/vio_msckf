#include <image_processor/image_processor.h>

#define EQUALIZE 0
#define MIN_DIST 30
#define MAX_CNT 100
#define F_THRESHOLD 1.0


// cam0:
//   T_cam_imu:
//     [0.0, 1.0, 0.0, 0.05,
//     -1.0, 0.0, 0.0, 0.0,
//      0.0, 0.0, 1.0, 0.0,
//      0.0, 0.0, 0.0, 1.0]
//   camera_model: pinhole
//   distortion_coeffs: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
//   distortion_model: radtan
//   intrinsics: [458.654, 457.296, 367.215, 248.375]
//   resolution: [752, 480]
//   timeshift_cam_imu: 0.0
// T_imu_body:
//   [1.0000, 0.0000, 0.0000, 0.0000,
//   0.0000, 1.0000, 0.0000, 0.0000,
//   0.0000, 0.0000, 1.0000, 0.0000,
//   0.0000, 0.0000, 0.0000, 1.0000]
float cam0_intrinsics[4] = {458.654, 457.296, 367.215, 248.375};
float cam0_distortion_coeffs[4] = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};

//const cv::Vec4d cam0_intrinsics = ();

int image_cols = 752;
int image_rows = 480;
double disparity = 0;

typedef unsigned long long int FeatureIDType;
FeatureIDType next_feature_id;

cv_bridge::CvImageConstPtr cam0_cur_img_ptr;

cv::Ptr<cv::Feature2D> detector_ptr;
vector<cv::Point2f> prev_features, cur_features,track_features;
vector<cv::Point2f> cur_un_pts;
cv::Mat prev_image, cur_image;

vector<int> features_id;
vector<int> track_cnt;

void removeVector(vector<cv::Point2f> &v,vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);    
}

void removeVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void init_first_frame()
{
    // detector_ptr = FastFeatureDetector::create(fast_threshold);

    // vector<KeyPoint> cur_pts(0);
    // detector_ptr->detect(cam0_cur_img_ptr, cur_pts);
    // for(unsigned int i = 0;i<cur_pts.size();i++)
    // {
    //     cur_features.push_back(cur_pts[i].pt);
    // }
    cv::Mat _img = cam0_cur_img_ptr->image.rowRange(0, image_rows); 
    if(EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0,cv::Size(8,8));
        clahe->apply(_img,cur_image);
    }
    else
    {
        cur_image = _img;
    }

    cv::goodFeaturesToTrack(cur_image, cur_features, MAX_CNT , 0.01, MIN_DIST, cv::Mat());
    next_feature_id = 0;
    for (auto &p : cur_features)
    {
        prev_features.push_back(p);
        features_id.push_back(next_feature_id++);
        track_cnt.push_back(1);
    }
    ROS_INFO("cur_features: %d", cur_features.size());
    prev_image = cur_image;

}

void track_feature()
{   

    cv::Mat _img = cam0_cur_img_ptr->image.rowRange(0, image_rows); 
    if(EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0,cv::Size(8,8));
        clahe->apply(_img,cur_image);
    }
    else
    {
        cur_image = _img;
    }

    vector<uchar> status;
    vector<float> error;

    track_features.clear();
    cv::calcOpticalFlowPyrLK(prev_image,
                             cur_image,
                             cur_features,
                             track_features,
                             status,
                             error,
                             cv::Size(21,21),3);

    for(unsigned int i = 0;i<track_features.size();i++)
    {
        if(status[i]==0)continue;
        if(track_features[i].y<0||
           track_features[i].y>cam0_cur_img_ptr->image.rows-1||
           track_features[i].x<0||
           track_features[i].x>cam0_cur_img_ptr->image.cols-1)
           {status[i] = 0;}
    }
    
    removeVector(prev_features, status);
    removeVector(cur_features, status);
    removeVector(track_features, status);
    removeVector(features_id, status);
    removeVector(track_cnt, status);

    //跟踪数目加一
    for (auto &n : track_cnt)
        n++;

    //计算光流前后跟踪像素差值均值
    vector<double> disparities;
    Vector2d distance;
    disparities.clear();
    disparities.reserve(track_features.size());
    for(unsigned int i = 0;i<track_features.size();i++)
    {
           distance.x() = cur_features[i].x - track_features[i].x;
           distance.y() = cur_features[i].y - track_features[i].y; 
           disparities.push_back(distance.norm());
           disparity += distance.norm(); 
    }
    disparity = disparity / disparities.size();
    
    //计算F矩阵
    if( track_features.size() >=8 )
    {
        ROS_DEBUG("FM ransac begins");
        vector<cv::Point2f> un_cur_points(cur_features.size()), un_forw_points(track_features.size());
        
        undistortedPoints(cur_features,un_cur_points);
        undistortedPoints(track_features,un_forw_points);
        
        // for (unsigned int i = 0; i < cur_features.size(); i++)
        // {
        //     // Eigen::Vector3d tmp_p;
        //     // m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
        //     // tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
        //     // tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
        //     // vector<cv::Point2f>& tmp_p(0);
        //     // undistortedPoints(vector<cv::Point2f>(cur_features[i].x,cur_features[i].y), tmp_p);
        //     // // const vector<cv::Point2f>& pts_in, vector<cv::Point2f>& pts_out
        //     un_cur_pts[i] = cur_features[i];

        //     // m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
        //     // tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
        //     // tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
        //     un_forw_pts[i] = track_features[i];
        // }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_points, un_forw_points, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_features.size();
        removeVector(prev_features, status);
        removeVector(cur_features, status);
        removeVector(track_features, status);
        //reduceVector(cur_un_pts, status);
        removeVector(features_id, status);
        removeVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %d", size_a, track_features.size());
    }
}

void add_new_feature()
{
    cv::Mat mask;
    //setMask
    mask = cv::Mat(image_rows, image_cols, CV_8UC1, cv::Scalar(255));

    //打包特征点及其ID，计数
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < track_features.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(track_features[i], features_id[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    track_features.clear();
    features_id.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            track_features.push_back(it.second.first);
            features_id.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }

    int n_max_cnt = MAX_CNT - static_cast<int>(track_features.size());
    vector<cv::Point2f> new_features;
    if(n_max_cnt>0)
    {
        // vector<KeyPoint> new_pts(0);
        // detector_ptr->detect(cam0_cur_img_ptr, new_pts);
        // for(unsigned int i = 0;i<new_pts.size();i++)
        // {
        //     new_features.push_back(cur_pts[i].pt);
        // }
        cv::goodFeaturesToTrack(cur_image, new_features, n_max_cnt, 0.01, MIN_DIST, mask);
    }
    else
    {
        new_features.clear();
    }

    for (auto &p : new_features)
    {
        track_features.push_back(p);
        features_id.push_back(next_feature_id++);
        track_cnt.push_back(1);
    }
    
    prev_image = cur_image;
    cur_features = track_features;
    prev_features = cur_features;
}

void undistortedPoints(const vector<cv::Point2f>& pts_in, vector<cv::Point2f>& pts_out)
{
    if (pts_in.size() == 0) return;

    const string distortion_model = "radtan";

    const cv::Vec4d distortion_coeffs(
        cam0_distortion_coeffs[0],cam0_distortion_coeffs[1],cam0_distortion_coeffs[2],cam0_distortion_coeffs[3]);
    // distortion_coeffs(0) = -0.28340811;
    // distortion_coeffs(1) =  0.07395907;
    // distortion_coeffs(2) =  0.00019359;
    // distortion_coeffs(3) = 1.76187114e-05;

    const cv::Matx33d K(
      cam0_intrinsics[0], 0.0, cam0_intrinsics[2],
      0.0, cam0_intrinsics[1], cam0_intrinsics[3],
      0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);

    const cv::Matx33d R_m(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);

    if( distortion_model == "radtan")
    {
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,R_m, K_new);    
    }

} 
