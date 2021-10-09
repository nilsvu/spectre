// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity {
namespace ConstitutiveRelations {

class Layered : public ConstitutiveRelation<3> {
 public:
  static constexpr size_t Dim = 3;
  static constexpr size_t volume_dim = 3;

  struct LayerBoundaries {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "The z-coordinate of the boundaries between layers"};
  };

  struct Materials {
    using type = std::vector<std::unique_ptr<ConstitutiveRelation<Dim>>>;
    static constexpr Options::String help = {"The material in each layer"};
  };

  using options = tmpl::list<LayerBoundaries, Materials>;

  static constexpr Options::String help = {
      "A constitutive relation that describes an isotropic, homogeneous "
      "material in terms of two elastic moduli. These bulk and shear moduli "
      "indicate the material's resistance to volume and shape changes, "
      "respectively. Both are measured in units of stress, typically Pascals."};

  Layered() = default;
  Layered(Layered&&) = default;
  Layered& operator=(Layered&&) = default;
  ~Layered() override = default;

  Layered& operator=(const Layered& rhs) {
    layer_boundaries_ = rhs.layer_boundaries_;
    materials_.resize(rhs.materials_.size());
    for (size_t i=0;i<materials_.size();++i) {
      materials_[i] = rhs.materials_[i]->get_clone();
    }
    return *this;
  };

  Layered(const Layered& rhs) {
    *this = rhs;
  };

  Layered(std::vector<double> layer_boundaries, std::vector<std::unique_ptr<ConstitutiveRelation<Dim>>> materials) : layer_boundaries_(std::move(layer_boundaries)),
  materials_(std::move(materials)){}

  std::unique_ptr<ConstitutiveRelation<Dim>> get_clone() const override{
      return std::make_unique<Layered>(*this);
  }

  /// The constitutive relation that characterizes the elastic properties of a
  /// material
  void stress(gsl::not_null<tnsr::II<DataVector, Dim>*> stress,
              const tnsr::ii<DataVector, Dim>& strain,
              const tnsr::I<DataVector, Dim>& x) const override;

// NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    p | layer_boundaries_;
    p | materials_;
  }

  explicit Layered(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(ConstitutiveRelation<Dim>), Layered);

 private:
  std::vector<double> layer_boundaries_{};
std::vector<std::unique_ptr<ConstitutiveRelation<Dim>>> materials_;
};

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
