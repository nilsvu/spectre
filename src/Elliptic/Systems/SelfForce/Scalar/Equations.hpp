// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::OptionTags {
class SchwarzSmootherGroup;
}

namespace ScalarSelfForce {

/*!
 * \brief The first-order flux
 * $F^i=\{\partial_{r_\star}, \alpha\partial_{\cos\theta}\}\Psi_m$.
 */
void fluxes(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const tnsr::i<ComplexDataVector, 2>& field_gradient);

/*!
 * \brief The first-order flux on an element face
 * $F^i=\{n_{r_\star}, \alpha n_{\cos\theta}\}\Psi_m$.
 */
void fluxes_on_face(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);

/*!
 * \brief The source term $\beta \Psi_m + \gamma_i F^i$.
 */
void add_sources(gsl::not_null<Scalar<ComplexDataVector>*> source,
                 const Scalar<ComplexDataVector>& beta,
                 const tnsr::i<ComplexDataVector, 2>& gamma,
                 const Scalar<ComplexDataVector>& field,
                 const tnsr::I<ComplexDataVector, 2>& flux);

struct Fluxes {
  using argument_tags = tmpl::list<Tags::Alpha>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const Scalar<ComplexDataVector>& /*field*/,
                    const tnsr::i<ComplexDataVector, 2>& field_gradient);
  static void apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::i<DataVector, 2>& /*face_normal*/,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);
};

struct Sources {
  using argument_tags = tmpl::list<Tags::Beta, Tags::Gamma>;
  static void apply(gsl::not_null<Scalar<ComplexDataVector>*> scalar_equation,
                    const Scalar<ComplexDataVector>& beta,
                    const tnsr::i<ComplexDataVector, 2>& gamma,
                    const Scalar<ComplexDataVector>& field,
                    const tnsr::I<ComplexDataVector, 2>& flux);
};

struct ModifyBoundaryData {
 private:
  static constexpr size_t Dim = 2;
  using SchwarzOptionsGroup = elliptic::OptionTags::SchwarzSmootherGroup;
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, SchwarzOptionsGroup>;
  using singular_vars_on_mortars_tag =
      ::Tags::Variables<tmpl::list<Tags::SingularField,
                                   ::Tags::NormalDotFlux<Tags::SingularField>>>;

 public:
  using argument_tags = tmpl::list<Tags::FieldIsRegularized,
                                   overlaps_tag<Tags::FieldIsRegularized>,
                                   overlaps_tag<singular_vars_on_mortars_tag>>;
  using volume_tags = argument_tags;
  static void apply(
      gsl::not_null<Scalar<ComplexDataVector>*> field,
      gsl::not_null<Scalar<ComplexDataVector>*> n_dot_flux,
      const DirectionalId<Dim>& mortar_id, const bool sending,
      const tnsr::i<DataVector, Dim>& /*face_normal*/,
      const bool field_is_regularized,
      const DirectionalIdMap<Dim, bool>& neighbors_field_is_regularized,
      const DirectionalIdMap<Dim, typename singular_vars_on_mortars_tag::type>&
          singular_vars_on_mortars) {
    if (field_is_regularized == neighbors_field_is_regularized.at(mortar_id)) {
      // Both elements solve for the same field. Nothing to do.
      return;
    }
    if (field_is_regularized) {
      // We're on an element that's regularized and sending to or receiving from
      // an element that's not regularized. We have to add or subtract the
      // singular field.
      const double sign = sending ? 1. : -1.;
      const auto& singular_field =
          get<Tags::SingularField>(singular_vars_on_mortars.at(mortar_id));
      const auto& singular_field_n_dot_flux =
          get<::Tags::NormalDotFlux<Tags::SingularField>>(
              singular_vars_on_mortars.at(mortar_id));
      get(*field) += sign * get(singular_field);
      get(*n_dot_flux) += get(singular_field_n_dot_flux);
    }
  }
};

}  // namespace ScalarSelfForce
