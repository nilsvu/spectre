// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

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
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level. Specify one of: a single number, a "
        "list representing [phi, theta, r], or such a list for every block "
        "in the domain. The central cube always uses the value for 'theta' "
        "in both y- and z-direction."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points. Specify one of: a single number, a "
        "list representing [phi, theta, r], or such a list for every block "
        "in the domain. The central cube always uses the value for 'theta' "
        "in both y- and z-direction."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates. Equiangular "
        "coordinates give better gridpoint spacings in the angular "
        "directions, while equidistant coordinates give better gridpoint "
        "spacings in the center block."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the spherical shell "
        "between InnerRadius and OuterRadius."};
  };

  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. The innermost shell must have a 'Linear' distribution because "
        "it changes in sphericity. The 'RadialPartitioning' determines the "
        "number of shells."};
    static size_t lower_bound_on_size() { return 1; }
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "Options for the boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<InnerRadius, OuterRadius, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, RadialPartitioning, RadialDistribution>;

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
      "A 3D cubed sphere. Six wedges surround a central cube. Additional "
      "spherical shells, each composed of six wedges, can be added with the "
      "'RadialPartitioning' option."};

  Sphere(double inner_radius, double outer_radius,
         const typename InitialRefinement::type& initial_refinement,
         const typename InitialGridPoints::type& initial_number_of_grid_points,
         bool use_equiangular_map, std::vector<double> radial_partitioning = {},
         std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
             {domain::CoordinateMaps::Distribution::Linear},
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

  std::vector<std::array<size_t, 3>> initial_extents() const override {
    return initial_number_of_grid_points_;
  }

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    return initial_refinement_;
  }

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  double inner_radius_{};
  double outer_radius_{};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  bool use_equiangular_map_ = false;
  std::vector<double> radial_partitioning_{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};
}  // namespace creators
}  // namespace domain
