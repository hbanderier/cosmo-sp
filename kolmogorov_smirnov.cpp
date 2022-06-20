/*******************************************************************************
 *
 * Kolmogorov-Smirnov test on grid cell level
 *
 * The functions can be used with Python and NumPy and provide a significant
 * speedup compared to a Python implementation.
 * 
 * Compile it like this:
 * c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11
 * --includes` kolmogorov_smirnov.cpp
 * -o kolmogorov_smirnov`python3-config --extension-suffix`
 *
 * The test uses the empirical cumulative distribution function provided by
 * boost. Therefore it will only compile if you have boost installed. Download
 * and install the latest version from https://www.boost.org/ and modify
 * the last include statment below such that it points to your installation.
 *
 * In Python:
 * import kolmogorov_smirnov as ks
 * below = ks.ks(values_a, values_b, n_bins)
 *
 * Copyright (c) 2022 ETH Zurich, Christian Zeman
 * MIT License
 *
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <iomanip>
#include </project/pr133/hbanderi/boost_1_79_0/boost/math/distributions/empirical_cumulative_distribution_function.hpp>

namespace py = pybind11;


py::array_t<size_t> ks(const py::array_t<double> ea,
                       const py::array_t<double> eb,
                       size_t n_bins,
                       double c_alpha)
{
    // read array into buffer info and get pointer and shape
    // shape is expected to be (ny, nx, ne) to make it faster
    py::buffer_info buf_a = ea.request();
    py::buffer_info buf_b = eb.request();
    auto ptr_a = static_cast<double *>(buf_a.ptr);
    auto ptr_b = static_cast<double *>(buf_b.ptr);

    // dimensions
    size_t nt = buf_a.shape[0];
    size_t ny = buf_a.shape[1];
    size_t nx = buf_a.shape[2];
    size_t nm = buf_a.shape[3];

    // critical value for alpha = 0.05
    double cv_005 = c_alpha * std::sqrt(2*nm/(double)(nm*nm));

    // array for results
    auto reject = py::array_t<size_t>(nt*ny*nx);
    py::buffer_info buf_reject = reject.request();
    auto ptr_reject = static_cast<size_t *>(buf_reject.ptr);

    // Kolmogorov-Smirnov test for each grid point
    for (size_t t = 0; t < nt; t++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {

                size_t idx = t*ny*nx*nm + j*nx*nm + i*nm;
                size_t idx_rej = t*ny*nx + j*nx + i;

                // build cumulative distribution functions
                std::vector<double> va(nm);
                std::vector<double> vb(nm);
                std::copy(ptr_a + idx, ptr_a + idx + nm, va.begin());
                std::copy(ptr_b + idx, ptr_b + idx + nm, vb.begin());
                auto ecdf_a = boost::math::empirical_cumulative_distribution_function(std::move(va));
                auto ecdf_b = boost::math::empirical_cumulative_distribution_function(std::move(vb));

                // get minimum and maximum values
                const auto [min_a, max_a] = std::minmax_element(ptr_a+idx, ptr_a+idx+nm);
                const auto [min_b, max_b] = std::minmax_element(ptr_b+idx, ptr_b+idx+nm);
                double min = std::min(*min_a, *min_b);
                double max = std::max(*max_a, *max_b);
                if (max == min)
                {
                    // all the values are the same...
                    ptr_reject[idx_rej] = 0;
                    continue;
                }
                double step = (max-min)/(double)n_bins;

                // calculate maximum distance
                double max_dist = 0;
                for (double x=min; x<max; x=x+step)
                    max_dist = std::max(max_dist, std::abs(ecdf_a(x) - ecdf_b(x)));
  
                // reject if it's above the critical value
                if (max_dist > cv_005)
                    ptr_reject[idx_rej] = 1;
                else
                    ptr_reject[idx_rej] = 0;
            }
        }
    }
            
    // reshape result return it
    reject.resize({nt, ny, nx});
    return reject;
}


PYBIND11_MODULE(kolmogorov_smirnov, m) {
    m.doc() = "Kolmogorov-Smirnov test";
    m.def("ks", &ks, "Kolmogorov-Smirnov test");
}

