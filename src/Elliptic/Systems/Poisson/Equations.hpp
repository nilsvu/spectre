// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace Poisson {
template <size_t Dim, Geometry BackgroundGeometry>
struct Fluxes;
template <size_t Dim, Geometry BackgroundGeometry>
struct Sources;
}  // namespace Poisson
/// \endcond

namespace Poisson {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat spatial metric in Cartesian coordinates.
 */
template <size_t Dim>
void flat_cartesian_fluxes(
    gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient);

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <size_t Dim>
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const tnsr::i<DataVector, Dim>& field_gradient);

/*!
 * \brief Compute the fluxes $F^i=\gamma^{ij} n_j u$ where $n_j$ is the
 * `face_normal`.
 *
 * The `face_normal_vector` is $\gamma^{ij} n_j$.
 */
template <size_t Dim>
void fluxes_on_face(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataVector>& field);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
template <size_t Dim>
void add_curved_sources(gsl::not_null<Scalar<DataVector>*> source_for_field,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const tnsr::I<DataVector, Dim>& flux_for_field);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes<Dim, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const Scalar<DataVector>& field,
                    const tnsr::i<DataVector, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes<Dim, Geometry::Curved> {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, Dim>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const Scalar<DataVector>& field,
                    const tnsr::i<DataVector, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Add the sources \f$S\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct Sources<Dim, Geometry::Curved> {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, Dim>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> equation_for_field,
                    const tnsr::i<DataVector, Dim>& christoffel_contracted,
                    const Scalar<DataVector>& field,
                    const tnsr::I<DataVector, Dim>& field_flux);
};

template <size_t Dim>
struct ModifyBoundaryData {
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using volume_tags = tmpl::list<domain::Tags::Element<Dim>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> field,
                    gsl::not_null<Scalar<DataVector>*> n_dot_flux,
                    const DirectionalId<Dim>& mortar_id, const bool sending,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const Element<Dim>& element,
                    const tnsr::I<DataVector, Dim>& x) {
    const auto& element_id = element.id();
    const ElementId<Dim> reg_id{0, {{{2, 1}, {2, 1}}}};
    const bool field_is_regularized = element_id == reg_id;
    const bool neighbor_is_regularized = mortar_id.id == reg_id;
    if (field_is_regularized == neighbor_is_regularized) {
      // Both elements solve for the same field. Nothing to do.
      return;
    }
    if (field_is_regularized) {
      // We're on an element that's regularized and sending to or receiving from
      // an element that's not regularized. We have to add or subtract the
      // singular field.
      const double sign = sending ? 1. : -1.;
      // Assuming a regularization u(x) = u_R(x) + x^2 + y^2
      const DataVector singular_field = square(get<0>(x)) + square(get<1>(x));
      tnsr::I<DataVector, Dim> grad_singular_field{};
      get<0>(grad_singular_field) = 2. * get<0>(x);
      get<1>(grad_singular_field) = 2. * get<1>(x);
      const DataVector singular_field_n_dot_flux =
          get<0>(face_normal) * get<0>(grad_singular_field) +
          get<1>(face_normal) * get<1>(grad_singular_field);
      get(*field) += sign * singular_field;
      get(*n_dot_flux) += singular_field_n_dot_flux;
    }
  }
};

}  // namespace Poisson
