// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

class Angular : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the metric perturbation";
    using type = int;
  };

  static constexpr Options::String help = "Angular boundary condition";
  using options = tmpl::list<MModeNumber>;

  Angular() = default;
  Angular(const Angular&) = default;
  Angular& operator=(const Angular&) = default;
  Angular(Angular&&) = default;
  Angular& operator=(Angular&&) = default;
  ~Angular() = default;

  explicit Angular(int m_mode_number);

  /// \cond
  explicit Angular(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Angular);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<Angular>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    // TODO
    return {m_mode_number_ == 0 ? elliptic::BoundaryConditionType::Neumann
                                : elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<tnsr::aa<DataVector, 3>*> field_re,
             gsl::not_null<tnsr::aa<DataVector, 3>*> field_im,
             gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_field_gradient_re,
             gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_field_gradient_im,
             const tnsr::iaa<DataVector, 3>& deriv_field_re,
             const tnsr::iaa<DataVector, 3>& deriv_field_im) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<tnsr::aa<DataVector, 3>*> field_correction_re,
      gsl::not_null<tnsr::aa<DataVector, 3>*> field_correction_im,
      gsl::not_null<tnsr::aa<DataVector, 3>*>
          n_dot_field_correction_gradient_re,
      gsl::not_null<tnsr::aa<DataVector, 3>*>
          n_dot_field_correction_gradient_im,
      const tnsr::iaa<DataVector, 3>& deriv_field_correction_re,
      const tnsr::iaa<DataVector, 3>& deriv_field_correction_im) const;

 private:
  friend bool operator==(const Angular& lhs, const Angular& rhs);

  int m_mode_number_{};
};

bool operator!=(const Angular& lhs, const Angular& rhs);

}  // namespace GrSelfForce::BoundaryConditions
