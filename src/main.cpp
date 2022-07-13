#include "NumCpp.hpp"
#include "fmt/color.h"
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <memory>
#include "tuple"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

typedef nc::NdArray<double> ncD;

struct slam_parameters {
    double sig_x = 0.25;
    double sig_y = 0.1;
    double sig_alpha = 0.1;
    double sig_beta = 0.01;
    double sig_r = 0.08;
    const double sig_x2 = std::pow(sig_x, 2);
    const double sig_y2 = std::pow(sig_y, 2);
    const double sig_alpha2 = std::pow(sig_alpha, 2);
    const double sig_beta2 = std::pow(sig_beta, 2);
    const double sig_r2 = std::pow(sig_r, 2);
};

struct slam_data_np {
    ncD measurement;
    ncD control;
};

void process_input_data(std::vector<std::unique_ptr<slam_data_np>> &slam_data_vec_np) {
    std::string FILENAME = "data.txt";
    std::vector<double> vec_buffer;
    std::unique_ptr<slam_data_np> slamDataNpPtr = nullptr;
    std::ifstream file(FILENAME);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            vec_buffer.clear();
            std::copy(std::istream_iterator<double>(iss),
                      std::istream_iterator<double>(),
                      std::back_inserter(vec_buffer));
            if (vec_buffer.size() > 2) {
                slamDataNpPtr = std::make_unique<slam_data_np>();
                slamDataNpPtr->measurement = ncD(1, vec_buffer.size());
                for (nc::int32 col = 0; col < vec_buffer.size(); ++col) {
                    slamDataNpPtr->measurement(0, col) = vec_buffer[col];
                }
            } else {
                slamDataNpPtr->control = ncD(1, vec_buffer.size());
                for (nc::int32 col = 0; col < vec_buffer.size(); ++col) {
                    slamDataNpPtr->control(0, col) = vec_buffer[col];
                }
                slam_data_vec_np.push_back(std::move(slamDataNpPtr));
                slamDataNpPtr.reset();
            }
        }
        file.close();
    }
}

/**
 *
 * @param measurement 1 x 12 Measurement data vector of the form [beta1, r1, beta2, r2, beta3, r3, ...]
 * @param measurement_cov 2 x 2 Measurement covariance matrix with diagonal elements sig_beta2, and sig_r2
 * @param pose 3 x 1 Initial pose vector of the robot. The elements of initial pose are zero
 * @param pose_cov 3 x 3 Initial pose covariance diagonal matrix
 * @return std::tuple(k, landmark, landmark_cov) Where 'k' is the number of landmarks; 'landmark' is the GCS
 *         pose of the landmark; and 'landmark_cov' is a 12 x 12 landmark covariance matrix
 */
std::tuple<int, ncD, ncD> init_landmarks(ncD &measurement, ncD &measurement_cov, ncD &pose, ncD &pose_cov) {
    int k = std::floor(measurement.shape().rows / 2);
    double x, y, theta;
    ncD Z;
    x = pose(0, 0);
    y = pose(1, 0);
    theta = pose(2, 0);
    Z = measurement.reshape(6, 2);
    auto beta = Z(Z.rSlice(), 0);
    auto r = Z(Z.rSlice(), 1);
    ncD landmark_cov = {measurement_cov(0, 0), measurement_cov(1, 1),
                        measurement_cov(0, 0), measurement_cov(1, 1),
                        measurement_cov(0, 0), measurement_cov(1, 1),
                        measurement_cov(0, 0), measurement_cov(1, 1),
                        measurement_cov(0, 0), measurement_cov(1, 1),
                        measurement_cov(0, 0), measurement_cov(1, 1)};
    landmark_cov = nc::diag(landmark_cov);
    ncD landmark = nc::empty<double>(2 * k, 1);
    for (size_t i = 0; i < k; i++) {
        landmark(2 * i, 0) = (x + r[i] * std::cos(theta + beta[i]));
        landmark((2 * i) + 1, 0) = (x + r[i] * std::sin(theta + beta[i]));
    }
    return std::make_tuple(k, landmark, landmark_cov);
}

void draw_cov_ellipse(const ncD &mu, const ncD &cov, const std::string& color="r") {
    ncD U, S, Vt;
    nc::linalg::svd(cov, U, S, Vt);
    auto a = S(0, 0);
    auto b = S(1, 1);
    auto vx = U(0, 0);
    auto vy = U(0, 1);
    auto theta = std::atan(vy / vx);
    ncD R = {{nc::cos(theta), -nc::sin(theta)},
             {nc::sin(theta), nc::cos(theta)}};
    auto phi = nc::arange<double>(0, 2 * M_PI, M_PI / 50);
    ncD rot;
    for (size_t i = 0; i < 100; i++) {
        ncD rect = {2.4477 * std::sqrt(a) * nc::cos(phi[i]), 2.4477 * std::sqrt(b) * nc::sin(phi[i])};
        rect = rect.reshape(2, 1);
        auto tf_rect = nc::matmul(R,rect) + mu;
        if (rot.isempty()) rot = tf_rect;
        else rot = nc::append(rot, tf_rect, nc::Axis::COL);
    }
    auto x = (rot(0, rot.cSlice())).toStlVector();
    auto y = (rot(1, rot.cSlice())).toStlVector();
    plt::plot(x, y, color);
    plt::show();
}

int main(int argc, char **argv) {
    std::vector<std::unique_ptr<slam_data_np>> slam_data_vec_np;
    process_input_data(slam_data_vec_np);
    auto measurement = slam_data_vec_np[0]->measurement;
    measurement.reshape(12, 1);
    int time_step = 1;
    slam_parameters param;
    auto control_cov = ncD(3, 3);
    control_cov = nc::eye<double>(3);
    control_cov(0, 0) = param.sig_x2;
    control_cov(1, 1) = param.sig_y2;
    control_cov(2, 2) = param.sig_alpha2;
    auto measurement_cov = ncD(2, 2);
    measurement_cov = nc::eye<double>(2);
    measurement_cov(0, 0) = param.sig_beta2;
    measurement_cov(1, 1) = param.sig_r2;
    auto pose = nc::zeros<double>(3, 1);
    auto pose_cov = ncD(3, 3);
    pose_cov = nc::eye<double>(3);
    pose_cov(0, 0) = std::pow(0.02, 2);
    pose_cov(1, 1) = std::pow(0.02, 2);
    pose_cov(2, 2) = std::pow(0.1, 2);
    auto &&[k, landmark, landmark_cov] = init_landmarks(measurement, measurement_cov, pose, pose_cov);
    auto X = nc::vstack({pose, landmark});
    auto P = nc::vstack({nc::hstack({pose_cov, nc::zeros<double>(3, 2 * k)}),
                         nc::hstack({nc::zeros<double>(2 * k, 3), landmark_cov})});
    auto previous_X = X;
    ncD mu_dummy = {0, 0};
    mu_dummy.reshape(2, 1);
    ncD cov_dummy = {{0.0004, 0.0},
                     {0.0,    0.0004}};
    draw_cov_ellipse(mu_dummy, cov_dummy, "b");
    return 0;
}