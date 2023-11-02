// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/SurfaceFinder/Component.hpp"
#include "ParallelAlgorithms/SurfaceFinder/SurfaceFinder.hpp"
#include "ParallelAlgorithms/SurfaceFinder/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace elliptic {

namespace OptionTags {

struct StarSurfaceResolution {
  using type = std::pair<size_t, size_t>;
  static constexpr Options::String help = "StarSurfaceResolution";
};

}  // namespace OptionTags

namespace Tags {

struct StarSurfaceResolution : db::SimpleTag {
  using type = std::pair<size_t, size_t>;
  using option_tags = tmpl::list<OptionTags::StarSurfaceResolution>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& ylm_resolution) {
    return ylm_resolution;
  }
};

struct StarSurfaceAngularCoords : db::SimpleTag {
 private:
  static constexpr size_t Dim = 3;

 public:
  using type = std::vector<std::optional<
      IdPair<domain::BlockId, tnsr::I<double, 2, Frame::BlockLogical>>>>;

  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                                 OptionTags::StarSurfaceResolution>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options(
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator,
      const std::pair<size_t, size_t>& ylm_resolution) {
    // Get angular collocation points of radial rays
    const ylm::Spherepack ylm{get<0>(ylm_resolution), get<1>(ylm_resolution)};
    const tnsr::I<double, 3, Frame::BlockLogical> midpoint_logical_coords{
        {0.0, 0.0, 0.0}};
    const auto& [theta, phi] = ylm.theta_phi_points();
    type result{theta.size()};

    const auto domain = domain_creator->create_domain();
    tnsr::I<double, 3, Frame::Grid> grid_point{};
    for (const auto& block : domain.blocks()) {
      // Skip stationary blocks because we can't adjust the surface in them
      // anyway
      if (not(block.is_time_dependent() and block.has_distorted_frame())) {
        continue;
      }
      const auto midpoint_grid_coords =
          block.moving_mesh_logical_to_grid_map()(midpoint_logical_coords);
      const double midpoint_grid_radius = get(magnitude(midpoint_grid_coords));
      const tnsr::I<DataVector, 3, Frame::Grid> grid_points{
          {{midpoint_grid_radius * sin(theta) * cos(phi),
            midpoint_grid_radius * sin(theta) * sin(phi),
            midpoint_grid_radius * cos(theta)}}};
      for (size_t s = 0; s < grid_points.begin()->size(); ++s) {
        get<0>(grid_point) = get<0>(grid_points)[s];
        get<1>(grid_point) = get<1>(grid_points)[s];
        get<2>(grid_point) = get<2>(grid_points)[s];
        const auto block_logical_point =
            block_logical_coordinates_single_point(grid_point, block);
        if (block_logical_point.has_value()) {
          ASSERT(equal_within_roundoff(get<2>(*block_logical_point), 0.),
                 "Unexpected");
          result[s] =
              IdPair<domain::BlockId, tnsr::I<double, 2, Frame::BlockLogical>>{
                  domain::BlockId{block.id()},
                  tnsr::I<double, 2, Frame::BlockLogical>{
                      {get<0>(*block_logical_point),
                       get<1>(*block_logical_point)}}};
        }
      }
    }
    return result;
  }
};

}  // namespace Tags

namespace Actions {

template <typename TemporalIdTag>
struct AssembleStarSurface;

template <typename TemporalIdTag>
struct ConformToStarSurface {
  using const_global_cache_tags =
      tmpl::list<Tags::StarSurfaceAngularCoords, Tags::StarSurfaceResolution>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // const auto& specific_enthalpy =
    //     db::get<hydro::Tags::SpecificEnthalpy<DataVector>>(box);
    const auto& specific_enthalpy =
        db::get<Xcts::Tags::ConformalFactor<DataVector>>(box);
    const double surface_value = 1.05;

    // Get angular collocation points of radial rays
    const auto angular_coords = element_logical_coordinates(
        std::vector<ElementId<Dim>>{element_id}, db::get<Tags::StarSurfaceAngularCoords>(box)).at(element_id);

    // Find the surface along the radial rays
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    auto radius_of_surface = SurfaceFinder::find_radial_surface(
        specific_enthalpy, surface_value, mesh,
        angular_coords.element_logical_coords);

    // Map logical radius to grid radius
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const ElementMap<Dim, Frame::Grid> element_map{
        element_id, domain.blocks()[element_id.block_id()]};
    bool found_surface = false;
    for (size_t i = 0; i < radius_of_surface.size(); ++i) {
      if (radius_of_surface[i].has_value()) {
        found_surface = true;
        const tnsr::I<double, 3, Frame::ElementLogical> element_logical_coords{
            {get<0>(angular_coords.element_logical_coords)[i],
             get<1>(angular_coords.element_logical_coords)[i],
             radius_of_surface[i].value()}};
        radius_of_surface[i] = magnitude(element_map(element_logical_coords));
      }
    }
    if (not found_surface) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    // Send the surface radii to a singleton to assemble a Strahlkorper and
    // deform the domain
    using SurfaceFinderComponent =
        SurfaceFinder::Component<Metavariables, TemporalIdTag>;
    Parallel::simple_action<Actions::AssembleStarSurface<TemporalIdTag>>(
        Parallel::get_parallel_component<SurfaceFinderComponent>(cache),
        db::get<TemporalIdTag>(box), std::move(radius_of_surface),
        angular_coords.offsets);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename TemporalIdTag>
struct AssembleStarSurface {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename TemporalIdTag::type& temporal_id,
      const std::vector<std::optional<double>>& surface_radii_in_grid_frame,
      const std::vector<size_t>& offsets) {
    const auto& ylm_resolution = db::get<Tags::StarSurfaceResolution>(box);

    db::mutate<SurfaceFinder::Tags::FilledRadii<typename TemporalIdTag::type>>(
        [&temporal_id, &surface_radii_in_grid_frame, &offsets](
            const gsl::not_null<
                std::unordered_map<typename TemporalIdTag::type,
                                   std::pair<DataVector, std::vector<bool>>>*>
                filled_radii) {
          for (size_t i = 0; i < surface_radii_in_grid_frame.size(); ++i) {
            if (surface_radii_in_grid_frame[i].has_value()) {
              get<0>((*filled_radii)[temporal_id])[offsets[i]] =
                  *surface_radii_in_grid_frame[i];
              get<1>((*filled_radii)[temporal_id])[offsets[i]] = true;
            }
          }
        },
        make_not_null(&box));

    const auto& filled_radii =
        db::get<SurfaceFinder::Tags::FilledRadii<typename TemporalIdTag::type>>(
            box)
            .at(temporal_id);
    for (const bool filled : get<1>(filled_radii)) {
      if (not filled) {
        return;
      }
    }
    const ylm::Strahlkorper<Frame::Grid> strahlkorper{
        get<0>(ylm_resolution), get<1>(ylm_resolution), get<0>(filled_radii),
        std::array<double, 3>{{0.0, 0.0, 0.0}}};
    Parallel::printf("strahlkorper: %s\n", strahlkorper);
  }
};

}  // namespace Actions
}  // namespace elliptic
