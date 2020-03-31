// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Elliptic/Systems/Poisson/DgSchemes.hpp"

#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::elliptic::dg;

namespace TestHelpers {
namespace Poisson {

namespace {
// Define a simple central flux here for now. We can switch to the elliptic
// internal penalty flux once it is made conformant to the
// `dg::protocols::NumericalFlux` in this PR:
// https://github.com/sxs-collaboration/spectre/pull/1725
template <size_t Dim>
struct CentralFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
 private:
  using poisson_system =
      ::Poisson::FirstOrderSystem<Dim, ::Poisson::Geometry::Euclidean>;
  using all_fields_tags =
      db::get_variables_tags_list<typename poisson_system::fields_tag>;

 public:
  using variables_tags = all_fields_tags;
  using argument_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;
  using package_field_tags = argument_tags;
  using package_extra_tags = tmpl::list<>;
  void package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_n_dot_field_flux,
      const gsl::not_null<tnsr::i<DataVector, Dim>*>
          packaged_n_dot_aux_field_flux,
      const Scalar<DataVector>& n_dot_field_flux,
      const tnsr::i<DataVector, Dim> n_dot_aux_field_flux) const noexcept {
    *packaged_n_dot_field_flux = n_dot_field_flux;
    *packaged_n_dot_aux_field_flux = n_dot_aux_field_flux;
  }
  void operator()(
      const gsl::not_null<Scalar<DataVector>*> numerical_flux_field,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> numerical_flux_aux_field,
      const Scalar<DataVector>& n_dot_field_flux_interior,
      const tnsr::i<DataVector, Dim> n_dot_aux_field_flux_interior,
      const Scalar<DataVector>& n_dot_field_flux_exterior,
      const tnsr::i<DataVector, Dim> n_dot_aux_field_flux_exterior) const
      noexcept {
    get(*numerical_flux_field) =
        0.5 * (get(n_dot_field_flux_interior) - get(n_dot_field_flux_exterior));
    for (size_t d = 0; d < Dim; d++) {
      numerical_flux_aux_field->get(d) =
          0.5 * (n_dot_aux_field_flux_interior.get(d) -
                 n_dot_aux_field_flux_exterior.get(d));
    }
  }
};
}  // namespace

template <size_t Dim>
Matrix strong_first_order_dg_operator_matrix(
    const DomainCreator<Dim>& domain_creator) {
  using system =
      ::Poisson::FirstOrderSystem<Dim, ::Poisson::Geometry::Euclidean>;
  const typename system::fluxes fluxes_computer{};

  // Shortcuts for tags
  using field_tag = ::Poisson::Tags::Field;
  using field_gradient_tag =
      ::Tags::deriv<field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
  using all_fields_tags =
      db::get_variables_tags_list<typename system::fields_tag>;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, all_fields_tags,
                                       tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  /// [boundary_scheme]
  // Choose a numerical flux
  using NumericalFlux = CentralFlux<Dim>;
  const NumericalFlux numerical_fluxes_computer{};
  // Define the boundary scheme
  using BoundaryData = ::dg::FirstOrderScheme::BoundaryData<NumericalFlux>;
  const auto package_boundary_data =
      [&numerical_fluxes_computer](
          const Mesh<Dim - 1>& face_mesh,
          const tnsr::i<DataVector, Dim>& /*face_normal*/,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags> &
          /*div_fluxes*/) -> BoundaryData {
    return ::dg::FirstOrderScheme::package_boundary_data(
        numerical_fluxes_computer, face_mesh, n_dot_fluxes,
        get<::Tags::NormalDotFlux<field_tag>>(n_dot_fluxes),
        get<::Tags::NormalDotFlux<field_gradient_tag>>(n_dot_fluxes));
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const auto result, const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Mesh<Dim>& mesh, const ::dg::MortarId<Dim>& mortar_id,
          const Mesh<Dim - 1>& mortar_mesh,
          const ::dg::MortarSize<Dim - 1>& mortar_size) {
        const size_t dimension = mortar_id.first.dimension();
        auto boundary_contribution =
            std::decay_t<decltype(*result)>{dg::FirstOrderScheme::boundary_flux(
                local_boundary_data, remote_boundary_data,
                numerical_fluxes_computer, magnitude_of_face_normal,
                mesh.extents(dimension), mesh.slice_away(dimension),
                mortar_mesh, mortar_size)};
        add_slice_to_data(result, std::move(boundary_contribution),
                          mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), mortar_id.first));
      };
  /// [boundary_scheme]

  /// [build_operator_matrix]
  return helpers::build_operator_matrix<system>(
      domain_creator, [&fluxes_computer, &package_boundary_data,
                       &apply_boundary_contribution](const auto&... args) {
        return helpers::apply_first_order_dg_operator<system>(
            args..., fluxes_computer,
            // The Poisson fluxes and sources need no arguments, so we return
            // empty tuples
            [](const auto&...) { return std::tuple<>(); },
            [](const auto&...) { return std::tuple<>(); },
            package_boundary_data, apply_boundary_contribution);
      });
  /// [build_operator_matrix]
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template Matrix strong_first_order_dg_operator_matrix( \
      const DomainCreator<DIM(data)>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Poisson
}  // namespace TestHelpers
