// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

// IWYU wants to include things we definitely don't need...
// IWYU pragma: no_include <memory> // Needed in cpp file
// IWYU pragma: no_include <pup.h>  // Not needed

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp" // Not needed

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
/// Create a 3D Domain in the shape of a sphere consisting of six wedges
/// and a central cube. For an image showing how the wedges are aligned in
/// this Domain, see the documentation for Shell.
class Sphere : public DomainCreator<3> {
 private:
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the sphere circumscribing the inner cube."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the Sphere."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,angular]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "Options for the boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                                   InitialGridPoints, UseEquiangularMap>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          BoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates a 3D Sphere with seven Blocks.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in the radial direction can be set\n"
      "independently of the number of gridpoints in the angular directions.\n"
      "The number of gridpoints along the dimensions of the cube is equal\n"
      "to the number of gridpoints along the angular dimensions of the "
      "wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "directions, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Sphere(typename InnerRadius::type inner_radius,
         typename OuterRadius::type outer_radius,
         typename InitialRefinement::type initial_refinement,
         typename InitialGridPoints::type initial_number_of_grid_points,
         typename UseEquiangularMap::type use_equiangular_map,
         std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
             boundary_condition = nullptr,
         const Options::Context& context = {});

  Sphere() = default;
  Sphere(const Sphere&) = delete;
  Sphere(Sphere&&) = default;
  Sphere& operator=(const Sphere&) = delete;
  Sphere& operator=(Sphere&&) = default;
  ~Sphere() override = default;

  Domain<3> create_domain() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_ = false;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;
};
}  // namespace creators
}  // namespace domain
