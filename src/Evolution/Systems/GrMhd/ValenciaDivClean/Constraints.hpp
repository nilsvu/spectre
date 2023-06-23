// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace grmhd::ValenciaDivClean::Tags {

template <size_t Dim>
struct DivBCompute : db::ComputeTag,
                     ::Tags::div<hydro::Tags::MagneticField<DataVector, Dim>> {
  using argument_tags = tmpl::list<
      hydro::Tags::MagneticField<DataVector, Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<Dim>>;
  using return_type = Scalar<DataVector>;
  using base = ::Tags::div<hydro::Tags::MagneticField<DataVector, Dim>>;
  static constexpr auto function(
      const gsl::not_null<Scalar<DataVector>*> div_b,
      const tnsr::I<DataVector, Dim>& magnetic_field,
      const evolution::dg::subcell::ActiveGrid active_grid,
      const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& dg_inv_jacobian,
      const std::optional<InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>&
          subcell_inv_jacobian) {
    divergence(div_b, magnetic_field,
               active_grid == evolution::dg::subcell::ActiveGrid::Dg
                   ? dg_mesh
                   : subcell_mesh,
               active_grid == evolution::dg::subcell::ActiveGrid::Dg
                   ? dg_inv_jacobian
                   : subcell_inv_jacobian.value());
  }
};
}  // namespace grmhd::ValenciaDivClean::Tags
