#include "sciplot/sciplot.hpp"
#include "NumCpp.hpp"
#include "fmt/color.h"
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <memory>
#include "tuple"

using namespace sciplot;

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

std::tuple<int, ncD, ncD> init_landmarks(ncD &measurement, ncD &measurement_cov, ncD &pose, ncD &pose_cov) {
    int k = std::floor(measurement.shape().rows/2);
    double x, y, theta;
    ncD Z;
    x = pose(0,0);
    y = pose(1,0);
    theta = pose(2,0);
    Z = measurement.reshape(6,2);
    Z.print();
    return std::make_tuple(1, nc::zeros<double>(1,1),  nc::zeros<double>(1,1));
}

int main(int argc, char **argv) {
    std::vector<std::unique_ptr<slam_data_np>> slam_data_vec_np;
    process_input_data(slam_data_vec_np);
//    for (auto &item: slam_data_vec_np) {
//        fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold,
//                   "NumCpp measurement data :\n");
//        item->measurement.print();
//        std::cout << "NumCpp control data :\n";
//        item->control.print();
//    }
    auto measurement = slam_data_vec_np[0]->measurement;
    measurement.reshape(12, 1);
    std::cout << measurement.shape();
    int time_step = 1;
    slam_parameters param;
    auto control_cov = ncD(3, 3);
    control_cov = nc::eye<double>(3);
    control_cov(0,0) = param.sig_x2;
    control_cov(1,1) = param.sig_y2;
    control_cov(2,2) = param.sig_alpha2;
    auto measurement_cov = ncD(2, 2);
    measurement_cov = nc::eye<double>(2);
    measurement_cov(0,0) = param.sig_beta2;
    measurement_cov(1,1) = param.sig_r2;
    auto pose = nc::zeros<double>(3,1);
    auto pose_cov = ncD(3,3);
    pose_cov = nc::eye<double>(3);
    pose_cov(0,0) = std::pow(0.02, 2);
    pose_cov(1,1) = std::pow(0.02, 2);
    pose_cov(2,2) = std::pow(0.1, 2);
    init_landmarks(measurement, measurement_cov, pose, pose_cov);
    return 0;
}