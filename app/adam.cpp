#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` adam.cpp -o adam`python3.12-config --extension-suffix`

namespace py = pybind11;
using Eigen::VectorXd;

class AdamOptimizer {
private:
    VectorXd weights;
    VectorXd m; // Primera estimación del momento
    VectorXd v; // Segunda estimación del momento
    double beta1, beta2, alpha, epsilon;
    int t; // Contador de iteraciones

public:
    AdamOptimizer(int num_features, double lr, double beta1, double beta2, double epsilon)
        : beta1(beta1), beta2(beta2), alpha(lr), epsilon(epsilon), t(0) {
        weights = VectorXd::Zero(num_features);
        m = VectorXd::Zero(num_features);
        v = VectorXd::Zero(num_features);
    }

    void update(const std::vector<double>& gradients) {
        VectorXd g = Eigen::Map<const VectorXd>(gradients.data(), gradients.size());
        t++;

        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g.array().square().matrix();

        VectorXd m_hat = m / (1 - std::pow(beta1, t));
        VectorXd v_hat = v / (1 - std::pow(beta2, t));

        //weights -= alpha * m_hat.array() / (v_hat.array().sqrt() + epsilon).matrix();
        //weights -= alpha * m_hat.array() / (v_hat.array().sqrt() + epsilon).array();
        weights.array() -= alpha * m_hat.array() / (v_hat.array().sqrt() + epsilon).array();


    }

    std::vector<double> get_weights() const {
        return std::vector<double>(weights.data(), weights.data() + weights.size());
    }
};

PYBIND11_MODULE(adam, m) {
    py::class_<AdamOptimizer>(m, "AdamOptimizer")
        .def(py::init<int, double, double, double, double>())
        .def("update", &AdamOptimizer::update)
        .def("get_weights", &AdamOptimizer::get_weights);
}
