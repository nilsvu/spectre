// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"

namespace py = pybind11;

namespace GeneralRelativity::py_bindings {

namespace {
template <size_t Dim, IndexType Index>
void bind_spacetime_impl(py::module& m) {  // NOLINT
  m.def("christoffel_first_kind",
        static_cast<tnsr::abb<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::abb<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::christoffel_first_kind),
        py::arg("d_metric"));

  m.def("christoffel_second_kind",
        static_cast<tnsr::Abb<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::abb<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::AA<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::christoffel_second_kind),
        py::arg("d_metric"), py::arg("inverse_metric"));

  m.def("ricci_scalar",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::aa<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::AA<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::ricci_scalar),
        py::arg("ricci_tensor"), py::arg("inverse_metric"));

  m.def("ricci_tensor",
        static_cast<tnsr::aa<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::Abb<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::aBcc<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::ricci_tensor),
        py::arg("christoffel_2nd_kind"), py::arg("d_christoffel_2nd_kind"));
}

template <size_t Dim>
void bind_impl(py::module& m) {  // NOLINT
  m.def("lapse",
        static_cast<Scalar<DataVector> (*)(const tnsr::I<DataVector, Dim>&,
                                           const tnsr::aa<DataVector, Dim>&)>(
            &::gr::lapse),
        py::arg("shift"), py::arg("spacetime_metric"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::II<DataVector, Dim> (*)(
            const tnsr::II<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("inverse_spatial_metric"), py::arg("normal_vector"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::ii<DataVector, Dim> (*)(
            const tnsr::ii<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spatial_metric"), py::arg("normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::Ij<DataVector, Dim> (*)(
            const tnsr::I<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("normal_vector"), py::arg("normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::aa<DataVector, Dim> (*)(
            const tnsr::aa<DataVector, Dim>&, const tnsr::a<DataVector, Dim>&,
            const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spacetime_metric"), py::arg("spacetime_normal_one_form"),
        py::arg("interface_unit_normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::AA<DataVector, Dim> (*)(
            const tnsr::AA<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("inverse_spacetime_metric"), py::arg("spacetime_normal_vector"),
        py::arg("interface_unit_normal_vector"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::Ab<DataVector, Dim> (*)(
            const tnsr::A<DataVector, Dim>&, const tnsr::a<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spacetime_normal_vector"),
        py::arg("spacetime_normal_one_form"),
        py::arg("interface_unit_normal_vector"),
        py::arg("interface_unit_normal_one_form"));

  m.def(
      "shift",
      static_cast<tnsr::I<DataVector, Dim> (*)(
          const tnsr::aa<DataVector, Dim>&, const tnsr::II<DataVector, Dim>&)>(
          &::gr::shift),
      py::arg("spacetime_metric"), py::arg("inverse_spatial_metric"));

  m.def("spacetime_normal_one_form",
        static_cast<tnsr::a<DataVector, 3> (*)(const Scalar<DataVector>&)>(
            &::gr::spacetime_normal_one_form),
        py::arg("lapse"));

  m.def("spacetime_normal_vector",
        static_cast<tnsr::A<DataVector, 3> (*)(const Scalar<DataVector>&,
                                               const tnsr::I<DataVector, 3>&)>(
            &::gr::spacetime_normal_vector),
        py::arg("lapse"), py::arg("shift"));
}
}  // namespace

PYBIND11_MODULE(_PyGeneralRelativity, m) {  // NOLINT
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_spacetime_impl<1, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<2, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<3, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<1, IndexType::Spacetime>(m);
  py_bindings::bind_spacetime_impl<2, IndexType::Spacetime>(m);
  py_bindings::bind_spacetime_impl<3, IndexType::Spacetime>(m);
  py_bindings::bind_impl<1>(m);
  py_bindings::bind_impl<2>(m);
  py_bindings::bind_impl<3>(m);
  m.def("psi4real",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::ii<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::ijj<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::II<DataVector, 3>&, const tnsr::I<DataVector, 3>&)>(
            &::gr::psi_4_real),
        py::arg("spatial_ricci"), py::arg("extrinsic_curvature"),
        py::arg("cov_deriv_extrinsic_curvature"), py::arg("spatial_metric"),
        py::arg("inverse_spatial_metric"), py::arg("inertial_coords"));
}
}  // namespace GeneralRelativity::py_bindings
