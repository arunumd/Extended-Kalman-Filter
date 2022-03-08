#include "sciplot/sciplot.hpp"
#include "NumCpp.hpp"
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <memory>

using namespace sciplot;

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
    const double sig_r2 = std::pow(sig_r,2);
};

//struct slam_data {
//    std::vector<float> measurement;
//    std::vector<float> control;
//};

struct slam_data_np {
    nc::NdArray<double> measurement;
    nc::NdArray<double> control;
};

int main(int argc, char** argv)
{
    std::string FILENAME = "data.txt";
    std::vector<double> vec_buffer;
//    std::unique_ptr<slam_data> slamDataPtr = nullptr;
    std::unique_ptr<slam_data_np> slamDataNpPtr = nullptr;
//    std::vector<std::unique_ptr<slam_data>> slam_data_vec;
    std::vector<std::unique_ptr<slam_data_np>> slam_data_vec_np;
    std::ifstream file(FILENAME);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            vec_buffer.clear();
            std::copy(std::istream_iterator<float>(iss),
                      std::istream_iterator<float>(),
                      std::back_inserter(vec_buffer));
            if (vec_buffer.size() > 2) {
//                slamDataPtr = std::make_unique<slam_data>();
//                slamDataPtr->measurement.assign(vec_buffer.begin(), vec_buffer.end());
                slamDataNpPtr = std::make_unique<slam_data_np>();
                slamDataNpPtr->measurement = nc::NdArray<double>(1, vec_buffer.size());
                for (nc::int32 col = 0; col < vec_buffer.size(); ++col) {
                    slamDataNpPtr->measurement(0, col) = vec_buffer[col];
                }
            } else {
//                slamDataPtr->control.assign(vec_buffer.begin(), vec_buffer.end());
                slamDataNpPtr->control = nc::NdArray<double>(1, vec_buffer.size());
                for (nc::int32 col = 0; col < vec_buffer.size(); ++col) {
                    slamDataNpPtr->control(0, col) = vec_buffer[col];
                }
//                slam_data_vec.push_back(std::move(slamDataPtr));
                slam_data_vec_np.push_back(std::move(slamDataNpPtr));
//                slamDataPtr.reset();
                slamDataNpPtr.reset();
            }
        }
        file.close();
    }
//    std::cout << "Size of slam data is " << slam_data_vec.size() << "\n";
//    for (auto& item: slam_data_vec) {
//        std::cout<< "Measurement data\n";
//        for (auto& element : item->measurement) {
//            std::cout << element << "\t" << std::endl;
//        }
//        std::cout << "Control data\n";
//        for (auto& element : item->control) {
//            std::cout << element << "\t" << std::endl;
//        }
//    }

    for (auto &item: slam_data_vec_np) {
        std::cout << "NumCpp measurement data :\n";
        item->measurement.print();
        std::cout << "NumCpp control data :\n";
        item->control.print();
    }
}