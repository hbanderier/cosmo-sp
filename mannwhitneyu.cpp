/*******************************************************************************
 *
 * Mann-Whitney U test on grid cell level
 *
 * The functions can be used with Python and NumPy and provide a significant
 * speedup compared to a Python implementation.
 * 
 * Compile it like this:
 * c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11
 * --includes` mannwhitneyu.cpp -o mannwhitneyu`python3-config --extension-suffix`
 *
 * In Python:
 * import mannwhitneyu as mwu
 * below = mwu.mwu(values_a, values_b, u_crit)
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
#include <algorithm>
#include <iomanip>

namespace py = pybind11;


// get the ranksum from two ensembles
std::tuple<double, double> get_ranksum(const double *a, const double *b,
        size_t nm)
{
    // concatenate
    std::vector<double> ab(2*nm);
    std::copy(a, a + nm, ab.begin());
    std::copy(b, b + nm, ab.begin() + nm);

    // calculate ranks (no ties)
    std::vector<double> rnk_s(2*nm);
    std::vector<double> rnk(2*nm);
    std::iota(rnk_s.begin(), rnk_s.end(), 0);
    std::iota(rnk.begin(), rnk.end(), 0);
    std::stable_sort(rnk_s.begin(), rnk_s.end(),
            [&ab](size_t i1 ,size_t i2){return ab[i1] < ab[i2];} );
    std::stable_sort(rnk.begin(), rnk.end(),
            [&rnk_s](size_t i1 ,size_t i2){return rnk_s[i1] < rnk_s[i2];} );
    
    // ranks++ because the above procedure starts with zero and MWU with one
    for(auto&& r: rnk)
        r++;

    // correct ties in a rather inefficient fashion
    for (size_t i = 0; i < 2*nm; i++) {
        if (i != 2*nm-1) {
            size_t cnt = 1;
            double sum = rnk[i];
            std::vector<size_t> vind;
            vind.push_back(i);
            for (size_t j = i+1; j < 2*nm; j++) {
                if (ab[i] == ab[j]) {
                    cnt++;
                    sum += rnk[j];
                    vind.push_back(j);
                }
            }
            double rnk_tie = sum / cnt;
            for(auto&& ind: vind) {
                rnk[ind] = rnk_tie;
            }
        }
    }

    // calculate sum of ranks
    double sum_a = 0;
    double sum_b = 0;
    for (auto it = rnk.begin(); it != rnk.begin() + nm; it++)
        sum_a += *it;
    for (auto it = rnk.begin() + nm; it != rnk.end(); it++)
        sum_b += *it;

    return std::make_tuple(sum_a, sum_b);
}



py::array_t<size_t> mwu(const py::array_t<double> ea,
                        const py::array_t<double> eb,
                        size_t u_crit)
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

    // array for results
    auto below = py::array_t<size_t>(nt*ny*nx);
    py::buffer_info buf_below = below.request();
    auto ptr_below = static_cast<size_t *>(buf_below.ptr);

    // do Mann-Whitney U test for each grid point
    for (size_t t = 0; t < nt; t++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                
                // get ranksum
                size_t shift = t*ny*nx*nm + j*nx*nm + i*nm;
                size_t rsum_a, rsum_b;
                std::tie(rsum_a, rsum_b) = get_ranksum(ptr_a + shift,
                        ptr_b + shift, nm);

                // calculate u
                double u_a = nm*nm + (nm*(nm+1))/2 - rsum_a;
                double u_b = nm*nm + (nm*(nm+1))/2 - rsum_b;
                double u = std::min(u_a, u_b);
               
                // check if below critical value
                if (u <= u_crit)
                    ptr_below[t*ny*nx + j*nx + i] = 1;
                else
                    ptr_below[t*ny*nx + j*nx + i] = 0;
            }
        }
    }
            
    // reshape result return it
    below.resize({nt, ny, nx});
    return below;
}


py::array_t<size_t> mwu_soil(const py::array_t<double> ea,
                             const py::array_t<double> eb,
                             const py::array_t<double> frl,
                             size_t u_crit)
{
    // read array into buffer info and get pointer and shape
    // shape is expected to be (ny, nx, ne) to make it faster
    py::buffer_info buf_a = ea.request();
    py::buffer_info buf_b = eb.request();
    py::buffer_info buf_frl = frl.request();
    auto ptr_a = static_cast<double *>(buf_a.ptr);
    auto ptr_b = static_cast<double *>(buf_b.ptr);
    auto ptr_frl = static_cast<double *>(buf_frl.ptr);

    // dimensions
    size_t nt = buf_a.shape[0];
    size_t ny = buf_a.shape[1];
    size_t nx = buf_a.shape[2];
    size_t nm = buf_a.shape[3];

    // array for results
    auto below = py::array_t<size_t>(nt*ny*nx);
    py::buffer_info buf_below = below.request();
    auto ptr_below = static_cast<size_t *>(buf_below.ptr);

    // do Mann-Whitney U test for each grid point
    for (size_t t = 0; t < nt; t++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {

                // check if on land
                if (ptr_frl[j*nx+i] < 0.5) {
                    ptr_below[t*ny*nx + j*nx + i] = 0;
                    continue;
                }
                
                // get ranksum
                size_t shift = t*ny*nx*nm + j*nx*nm + i*nm;
                size_t rsum_a, rsum_b;
                std::tie(rsum_a, rsum_b) = get_ranksum(ptr_a + shift,
                        ptr_b + shift, nm);

                // calculate u
                double u_a = nm*nm + (nm*(nm+1))/2 - rsum_a;
                double u_b = nm*nm + (nm*(nm+1))/2 - rsum_b;
                double u = std::min(u_a, u_b);
               
                // check if below critical value
                if (u <= u_crit)
                    ptr_below[t*ny*nx + j*nx + i] = 1;
                else
                    ptr_below[t*ny*nx + j*nx + i] = 0;
            }
        }
    }
            
    // reshape result return it
    below.resize({nt, ny, nx});
    return below;
}


PYBIND11_MODULE(mannwhitneyu, m) {
    m.doc() = "Mann-Whitney U test";
    m.def("mwu", &mwu, "Mann-Whitney U test");
    m.def("mwu_soil", &mwu_soil, "Mann-Whitney U test for soil");
}

