#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;

// This document defines functions for
// atan2 (matrices)
// clamp (float/ints)
// where (float/ints)

template<typename T>
T clamp(T val, T min, T max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

template<typename Mat_T>
Mat_T atan2(const Mat_T& x, const Mat_T& y) {
    Mat_T result;
    for (int i = 0; i < x.n_cols; ++i) {
        for (int j = 0; j < x.n_rows; ++j) {
            result[i, j] = std::atan2(x[i, j], y[i, j]);
        }
    }
    return result;
}

// template<typename Mat_T, typename std::enable_if<is_arma_type<Mat_T>::value>::type>
// Mat_T mod(const Mat_T& x, const float m) {
//     Mat_T result;
//     for (int i = 0; i < x.n_cols; ++i) {
//         for (int j = 0; j < x.n_rows; ++j) {
//             result[i, j] = std::fmod(x[i, j], m);
//         }
//     }
//     return result;
// }

template<typename T, typename Q,
         typename std::enable_if<is_arma_type<Q>::value>::type* = nullptr>
void set_items(T& lhs, const Q& rhs, const std::array<int, 2>& start,
               const std::array<int, 2>& end, bool transpose = false) {
    for (int i = 0; i < end[0]  - start[0] + 1; ++i) {
        for (int j = 0; j < end[1]  - start[1] + 1; ++j) {
            if (transpose) {
                lhs(i + start[0], j + start[1]) = rhs(j, i);
            } else {
                lhs(i + start[0], j + start[1]) = rhs(i, j);
            }
        }
    }
}

template<typename T, typename Q,
         typename std::enable_if<std::is_arithmetic<Q>::value>::type* = nullptr>
void set_items(T& lhs, const Q& rhs, const std::array<int, 2>& start,
               const std::array<int, 2>& end, bool transpose = false) {
    for (int i = 0; i < end[0]  - start[0] + 1; ++i) {
        for (int j = 0; j < end[1]  - start[1] + 1; ++j) {
            if (transpose) {
                lhs(i + start[0], j + start[1]) = rhs;
            } else {
                lhs(i + start[0], j + start[1]) = rhs;
            }
        }
    }
}

template<typename T, typename S>
T mod(const T& x, const S& m) {
    return std::fmod(x, m);
}
