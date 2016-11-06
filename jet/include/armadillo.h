/*
    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices

    Copyright (c) 2016 Wolf Vollprecht <w.vollprecht@gmail.com>
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/numpy.h>
#include <armadillo>
#include <iostream>
using namespace std;

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

// NOTE
//
// Armadillo vectors (Col) are subclasses of Mat and therefore, also catched by
// this meta-template check.

template <typename T> class is_arma_mat {
private:
    template<typename Derived> static std::true_type test(const arma::Mat<Derived> &);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<T>()))::value;
};

template <typename Type>
class type_caster<Type, typename std::enable_if<is_arma_mat<Type>::value>::type> {
public:
    typedef typename Type::elem_type Scalar;

    bool load(handle src, bool) {

        array_t<Scalar> buffer(src, true);
        if (!buffer.check())
            return false;

        buffer_info info = buffer.request();

        if(info.ptr == nullptr) return false;

        if(info.ndim == 1) {
            value = arma::Mat<Scalar>((Scalar * ) info.ptr,
                    info.shape[0],
                    1,
                    false,  // access the same underlying memory as np array!
                            // might be too dangerous?
                    true);
                return true;
        }
        if(info.ndim == 2) {
            // Note: NumPy is Row-major stored and armadillo is
            // column-major. Therefore we have to do this transpose.
            value = arma::Mat<Scalar>((Scalar * ) info.ptr,
                info.shape[0],
                info.shape[1],
                false,  // access the same underlying memory as np array!
                        // might be too dangerous?
                true).t();
            return true;
        }
        return false;
    }
    static handle cast(Type src, return_value_policy /* policy */, handle /* parent */) {
        int sz = sizeof(Scalar);
        if(src.n_cols == 1) { // Implement this either via Col:: ... armadillo or no idea
            return array(buffer_info(
                /* Pointer to buffer */
                const_cast<Scalar *>(src.memptr()),
                /* Size of one scalar */
                sizeof(Scalar),
                /* Python struct-style format descriptor */
                format_descriptor<Scalar>::value,
                /* Number of dimensions */
                1,
                /* Buffer dimensions */
                { (size_t) src.n_rows,
                  (size_t) src.n_cols },
                /* Strides (in bytes) for each index */
                { sz ,
                  sz * src.n_rows}
            )).release();
        } else {
            return array(buffer_info(
                /* Pointer to buffer */
                const_cast<Scalar *>(src.memptr()),
                /* Size of one scalar */
                sizeof(Scalar),
                /* Python struct-style format descriptor */
                format_descriptor<Scalar>::value,
                /* Number of dimensions */
                2,
                /* Buffer dimensions */
                { (size_t) src.n_rows,
                  (size_t) src.n_cols },
                /* Strides (in bytes) for each index */
                { sz,
                  sz * src.n_rows }
            )).release();
        }

        return handle(Py_True).inc_ref();
    }
    PYBIND11_TYPE_CASTER(Type, _("Arma Matrix"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)