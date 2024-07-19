// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::OptionTags {
class SchwarzSmootherGroup;
}

namespace GrSelfForce {

void fluxes(gsl::not_null<tnsr::Iaa<ComplexDataVector, 3>*> flux,
            const Scalar<ComplexDataVector>& principal_factor,
            const tnsr::iaa<ComplexDataVector, 3>& field_gradient);

void fluxes_on_face(gsl::not_null<tnsr::Iaa<ComplexDataVector, 3>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const tnsr::aa<ComplexDataVector, 3>& field);

void add_sources(gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> source,
                 const tnsr::aaBB<ComplexDataVector, 3>& beta,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                 const tnsr::aa<ComplexDataVector, 3>& field,
                 const tnsr::Iaa<ComplexDataVector, 3>& flux);

struct Fluxes {
  using argument_tags = tmpl::list<Tags::Alpha>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_re,
                    gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_im,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::aa<DataVector, 3>& /*field_re*/,
                    const tnsr::aa<DataVector, 3>& /*field_im*/,
                    const tnsr::iaa<DataVector, 3>& field_gradient_re,
                    const tnsr::iaa<DataVector, 3>& field_gradient_im);
  static void apply(const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_re,
                    const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_im,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::i<DataVector, 3>& /*face_normal*/,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const tnsr::aa<DataVector, 3>& field_re,
                    const tnsr::aa<DataVector, 3>& field_im);
};

struct Sources {
  using argument_tags =
      tmpl::list<Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
  static void apply(gsl::not_null<tnsr::aa<DataVector, 3>*> scalar_equation_re,
                    gsl::not_null<tnsr::aa<DataVector, 3>*> scalar_equation_im,
                    const tnsr::aaBB<ComplexDataVector, 3>& beta,
                    const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                    const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                    const tnsr::aa<DataVector, 3>& field_re,
                    const tnsr::aa<DataVector, 3>& field_im,
                    const tnsr::Iaa<DataVector, 3>& flux_re,
                    const tnsr::Iaa<DataVector, 3>& flux_im);
};

struct ModifyBoundaryData {
 private:
  static constexpr size_t Dim = 3;
  using SchwarzOptionsGroup = elliptic::OptionTags::SchwarzSmootherGroup;
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, SchwarzOptionsGroup>;

 public:
  using argument_tags =
      tmpl::list<Tags::FieldIsRegularized,
                 overlaps_tag<Tags::FieldIsRegularized>, Tags::SingularField,
                 ::Tags::NormalDotFlux<Tags::SingularField>>;
  using volume_tags = tmpl::list<Tags::FieldIsRegularized,
                                 overlaps_tag<Tags::FieldIsRegularized>>;
  static void apply(
      gsl::not_null<tnsr::aa<DataVector, 3>*> field_re,
      gsl::not_null<tnsr::aa<DataVector, 3>*> field_im,
      gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_flux_re,
      gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_flux_im,
      const DirectionalId<Dim>& mortar_id, const bool sending,
      const tnsr::i<DataVector, Dim>& /*face_normal*/,
      const bool field_is_regularized,
      const DirectionalIdMap<Dim, bool>& neighbors_field_is_regularized,
      const tnsr::aa<ComplexDataVector, 3>& singular_field,
      const tnsr::aa<ComplexDataVector, 3>& singular_field_n_dot_flux) {
    if (field_is_regularized == neighbors_field_is_regularized.at(mortar_id)) {
      // Both elements solve for the same field. Nothing to do.
      return;
    }
    if (field_is_regularized) {
      // We're on an element that's regularized and sending to or receiving from
      // an element that's not regularized. We have to add or subtract the
      // singular field.
      const double sign = sending ? 1. : -1.;
      for (size_t i = 0; i < singular_field.size(); ++i) {
        (*field_re)[i] += sign * real(singular_field[i]);
        (*field_im)[i] += sign * imag(singular_field[i]);
        (*n_dot_flux_re)[i] += real(singular_field_n_dot_flux[i]);
        (*n_dot_flux_im)[i] += imag(singular_field_n_dot_flux[i]);
      }
    }
  }
};

}  // namespace GrSelfForce
