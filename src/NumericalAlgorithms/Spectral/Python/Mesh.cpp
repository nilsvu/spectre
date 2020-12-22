// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace py = pybind11;

namespace py_bindings {

void bind_basis(py::module& m) {  // NOLINT

  py::enum_<Spectral::Basis>(m, "Basis")
      .value("Legendre", Spectral::Basis::Legendre)
      .value("Chebyshev", Spectral::Basis::Chebyshev)
      .value("FiniteDifference", Spectral::Basis::FiniteDifference);
}

void bind_quadrature(py::module& m) {  // NOLINT

  py::enum_<Spectral::Quadrature>(m, "Quadrature")
      .value("Gauss", Spectral::Quadrature::Gauss)
      .value("GaussLobatto", Spectral::Quadrature::GaussLobatto)
      .value("CellCentered", Spectral::Quadrature::CellCentered)
      .value("FaceCentered", Spectral::Quadrature::FaceCentered);
}

template <size_t Dim>
void bind_mesh(py::module& m) {  // NOLINT
  //the bindings here are not complete
  py::class_<Mesh<Dim>>(m, ("Mesh" + std::to_string(Dim) + "D").c_str())
      .def(py::init<const size_t, const Spectral::Basis,
                    const Spectral::Quadrature>())
      .def(py::init<std::array<size_t, Dim>, const Spectral::Basis,
                    const Spectral::Quadrature>())
      .def(py::init<std::array<size_t, Dim>, std::array<Spectral::Basis, Dim>,
                    std::array<Spectral::Quadrature, Dim>>())
      .def(
          "extents",
          [](const Mesh<Dim>& mesh) { return mesh.extents().indices(); },
          "The number of grid points in each dimension of the grid.")
      .def(
          "extents",
          static_cast<size_t (Mesh<Dim>::*)(size_t) const>(&Mesh<Dim>::extents),
          "The number of grid points in the requested dimension of the grid.")
      .def("number_of_grid_points", &Mesh<Dim>::number_of_grid_points,
           "The total number of grid points in all dimensions.")
      .def("basis",
           static_cast<const std::array<Spectral::Basis, Dim>& (Mesh<Dim>::*)()
                           const>(&Mesh<Dim>::basis),
           "The basis chosen in each dimension of the grid.")
      .def("basis",
           static_cast<Spectral::Basis (Mesh<Dim>::*)(const size_t) const>(
               &Mesh<Dim>::basis),
           "The basis chosen in the requested dimension of the grid.")
      .def("quadrature",
           static_cast<Spectral::Quadrature (Mesh<Dim>::*)(const size_t) const>(
               &Mesh<Dim>::quadrature),
           "The quadrature chosen in each dimension of the grid.")
      .def("quadrature",
           static_cast<const std::array<Spectral::Quadrature, Dim>& (
               Mesh<Dim>::*)() const>(&Mesh<Dim>::quadrature),
           "The quadrature chosen in the requested dimension of the grid.");
}
template void bind_mesh<1>(py::module& m);
template void bind_mesh<2>(py::module& m);
template void bind_mesh<3>(py::module& m);

}  // namespace py_bindings
