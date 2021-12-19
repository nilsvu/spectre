// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>

#include "Elliptic/DiscontinuousGalerkin/Python/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/BuildOperatorMatrix.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

namespace detail {

template <typename Tags, typename FaceTags>
struct InitializeBackground;

template <typename... Tags, typename... FaceTags>
struct InitializeBackground<tmpl::list<Tags...>, tmpl::list<FaceTags...>> {
  using return_tags = tmpl::list<Tags..., domain::Tags::Faces<3, FaceTags>...>;
  using argument_tags = tmpl::list<
      domain::Tags::Coordinates<3, Frame::Inertial>, domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::Logical, Frame::Inertial>>;
  void operator()(const gsl::not_null<typename Tags::type*>... args,
                  const gsl::not_null<
                      DirectionMap<3, typename FaceTags::type>*>... face_args,
                  const tnsr::I<DataVector, 3>& x, const Mesh<3>& mesh,
                  const InverseJacobian<DataVector, 3, Frame::Logical,
                                        Frame::Inertial>& inv_jacobian) const {
    Xcts::Solutions::Kerr<> solution{1., {{0., 0., 0.}}, {{0., 0., 0.}}};
    const auto vars = variables_from_tagged_tuple(
        solution.variables(x, mesh, inv_jacobian, tmpl::list<Tags...>{}));
    EXPAND_PACK_LEFT_TO_RIGHT((*args = get<Tags>(vars)));
    for (const auto& direction : Direction<3>::all_directions()) {
      const auto face_vars =
          data_on_slice(vars, mesh.extents(), direction.dimension(),
                        index_to_slice_at(mesh.extents(), direction));
      EXPAND_PACK_LEFT_TO_RIGHT(
          ((*face_args)[direction] = get<FaceTags>(face_vars)));
    }
  }
};

}  // namespace detail

void bind_xcts_operator(py::module& m) {  // NOLINT
  using system =
      Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                             Xcts::Geometry::Curved, 0>;
  using background_tags = tmpl::remove_duplicates<tmpl::append<
      typename system::background_fields,
      typename system::sources_computer_linearized::argument_tags>>;
  using background_face_tags = typename system::background_fields;
  m.def(
      "build_xcts_operator_matrix_".c_str(),
      [](const DomainCreator<3>& domain_creator, const double penalty_parameter,
         const bool massive) {
        return build_operator_matrix<
            system, true,
            tmpl::list<detail::InitializeBackground<background_tags,
                                                    background_face_tags>>>(
            domain_creator, penalty_parameter, massive,
            tuples::TaggedTuple<
                Elasticity::Tags::ConstitutiveRelation<Dim>>{std::make_unique<
                Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(
                1., 2.)});
      },
      py::arg("domain_creator"), py::arg("penalty_parameter"),
      py::arg("massive"));
}

}  // namespace elliptic::dg::py_bindings
