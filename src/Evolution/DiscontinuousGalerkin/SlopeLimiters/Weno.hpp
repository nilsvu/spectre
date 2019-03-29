// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SizeOfElement.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/HwenoModifiedSolution.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoType.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
template <size_t>
class Mesh;

namespace PUP {
class er;
}  // namespace PUP

namespace SlopeLimiters {
template <size_t VolumeDim, typename TagsToLimit>
class Weno;
}  // namespace SlopeLimiters

namespace Tags {
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
/// \endcond

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief An implementation of the simple WENO limiter of Zhong, Chu, 2013.
///
/// This limiter uses a minmod troubled-cell indicator to indentify elements
/// that need limiting. On these "troubled" elements, the limited solution is
/// obtained by WENO reconstruction --- a linear superposition of the local
/// solution and solution estimates obtained from each neighboring element. For
/// this simple WENO limiter, the solution estimates are obtained by simply
/// extrapolating the neighbor solution onto the troubled element. For more
/// detail, see \cite Zhong2013.
///
/// This implementation uses the Minmod limiter described in \cite XXX to
/// identify troubled cells. The reconstruction step uses the oscillation
/// indicator of \cite YYY to assign the weights to each of the different
/// solution estimates.
///
/// Limitations:
/// -
template <size_t VolumeDim, typename... Tags>
class Weno<VolumeDim, tmpl::list<Tags...>> {
 public:
  /// \brief The WenoType
  ///
  /// One of `SlopeLimiters::WenoType`. See `SlopeLimiters::Weno`
  /// documentation for details.
  struct Type {
    using type = WenoType;
    static constexpr OptionString help = {"Type of WENO limiter"};
  };
  /// \brief The linear weight given to each neighbor
  ///
  /// This linear weight gets combined with the oscillation indicator to
  /// compute the weight for each WENO estimated solution. Larger values are
  /// better suited for problems with strong shocks, and smaller values are
  /// better suited to smooth problems.
  struct NeighborWeight {
    using type = double;
    static type default_value() noexcept { return 0.001; }
    static type lower_bound() noexcept { return 1e-6; }
    static type upper_bound() noexcept { return 0.1; }
    static constexpr OptionString help = {
        "Linear weight for each neighbor element's solution"};
  };
  /// \brief Turn the limiter off
  ///
  /// This option exists to temporarily disable the limiter for debugging
  /// purposes. For problems where limiting is not needed, the preferred
  /// approach is to not compile the limiter into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the limiter"};
  };
  using options = tmpl::list<Type, NeighborWeight, DisableForDebugging>;
  static constexpr OptionString help = {"A WENO limiter."};

  /// \brief Constuct a Weno limiter
  explicit Weno(const WenoType weno_type, const double neighbor_linear_weight,
                const bool disable_for_debugging = false) noexcept
      : weno_type_(weno_type),
        neighbor_linear_weight_(neighbor_linear_weight),
        disable_for_debugging_(disable_for_debugging) {}

  Weno() noexcept = default;
  Weno(const Weno& /*rhs*/) = default;
  Weno& operator=(const Weno& /*rhs*/) = default;
  Weno(Weno&& /*rhs*/) noexcept = default;
  Weno& operator=(Weno&& /*rhs*/) noexcept = default;
  ~Weno() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | weno_type_;
    p | neighbor_linear_weight_;
    p | disable_for_debugging_;
  }

  /// \brief Data to send to neighbor elements.
  struct PackagedData {
    // Data for troubled-cell indicator:
    tuples::TaggedTuple<::Tags::Mean<Tags>...> means;
    std::array<double, VolumeDim> element_size =
        make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());
    // Data for simple WENO reconstruction:
    Variables<tmpl::list<Tags...>> volume_data;
    Mesh<VolumeDim> mesh;

    // clang-tidy: google-runtime-references
    void pup(PUP::er& p) noexcept {  // NOLINT
      p | means;
      p | element_size;
      p | volume_data;
      p | mesh;
    }
  };

  using package_argument_tags =
      tmpl::list<Tags..., ::Tags::SizeOfElement<VolumeDim>,
                 ::Tags::Mesh<VolumeDim>>;

  /// \brief Package data for sending to neighbor elements.
  void package_data(const gsl::not_null<PackagedData*>& packaged_data,
                    const db::item_type<Tags>&... tensors,
                    const std::array<double, VolumeDim>& element_size,
                    const Mesh<VolumeDim>& mesh,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept {
    if (UNLIKELY(disable_for_debugging_)) {
      // Do not initialize packaged_data
      return;
    }

    const auto wrap_compute_means =
        [&mesh, &packaged_data ](auto tag, const auto& tensor) noexcept {
      for (size_t i = 0; i < tensor.size(); ++i) {
        // Compute the mean using the local orientation of the tensor and mesh.
        get<::Tags::Mean<decltype(tag)>>(packaged_data->means)[i] =
            mean_value(tensor[i], mesh);
      }
      return '0';
    };
    expand_pack(wrap_compute_means(Tags{}, tensors)...);

    packaged_data->element_size =
        orientation_map.permute_from_neighbor(element_size);

    (packaged_data->volume_data).initialize(mesh.number_of_grid_points());
    const auto wrap_copy_tensor = [&packaged_data](
        auto tag, const auto& tensor) noexcept {
      get<decltype(tag)>(packaged_data->volume_data) = tensor;
      return '0';
    };
    expand_pack(wrap_copy_tensor(Tags{}, tensors)...);
    packaged_data->volume_data = orient_variables(
        packaged_data->volume_data, mesh.extents(), orientation_map);

    packaged_data->mesh = orientation_map(mesh);
  }

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<::Tags::Element<VolumeDim>, ::Tags::Mesh<VolumeDim>,
                 ::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Limits the solution on the element.
  bool operator()(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
      const std::array<double, VolumeDim>& element_size,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept {
    if (UNLIKELY(disable_for_debugging_)) {
      // Do not modify input tensors
      return false;
    }

    // Enforce restrictions on h-refinement, p-refinement
    // (This allows the rest of the code to make simplying assumptions)
    if (UNLIKELY(alg::any_of(element.neighbors(),
                             [](const auto& direction_neighbors) noexcept {
                               return direction_neighbors.second.size() != 1;
                             }))) {
      ERROR("The Weno limiter does not yet support h-refinement");
      // Removing this limitation will require:
      // - Coming up with a good way to construct fits when the neighbors are
      //   "smaller" elements. I (FH) expect this to be a straightforward
      //   extension, no work required.
      // - Deciding whether to pay the cost of doing matrix math at each h-ref
      //   boundary, or to *greatly* increase the complexity and quantity of
      //   caching to handle all possible h-ref situations.
      // - Coming up with a good way to do the WENO weighted sum with multiple
      //   neighbors in each direction. I (FH) also expect this to be easy.
    }
    alg::for_each(
        neighbor_data, [&mesh, this ](const auto& neighbor_and_data) noexcept {
          if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
            ERROR("The Weno limiter does not yet support p-refinement");
            // Removing this limitation will require:
            // - Generalizing the fitting code to handle neighbors with a
            // coarser
            //   or finer grid. The infrastructure is already in place. But if
            //   the neighbor has a finer grid, its data will have to be
            //   promoted to a denser grid so the problem is not
            //   underconstrained. (This is related to projection...)
            // - Deciding whether to do more matrix math at p-refinement bdries
            //   or greatly increase the complexity and quantity of caching.
          }
        });

    // Troubled-cell detection for Weno flags the cell for limiting if
    // any component of any tensor needs limiting.
    const auto minmod_tci_type = MinmodType::LambdaPiN;
    const double minmod_tci_tvbm_constant = 0.0;
    const bool cell_is_troubled =
        Minmod_detail::troubled_cell_indicator<VolumeDim, PackagedData,
                                               Tags...>(
            (*tensors)..., neighbor_data, minmod_tci_type,
            minmod_tci_tvbm_constant, element, mesh, element_size);

    if (not cell_is_troubled) {
      // No limiting is needed
      return false;
    }

    // Extrapolate all data
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<tmpl::list<Tags...>>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        modified_neighbor_solutions;

    if (weno_type_ == WenoType::Hweno) {
      for (const auto& neighbor_and_data : neighbor_data) {
        modified_neighbor_solutions[neighbor_and_data.first].initialize(
            mesh.number_of_grid_points());
      }
      const auto wrap_hweno_neighbor_solution_one_tensor = [
        this, &element, &mesh, &neighbor_data, &modified_neighbor_solutions
      ](auto tag, const auto& tensor) noexcept {
        for (const auto& neighbor_and_data : neighbor_data) {
          const auto& primary_neighbor = neighbor_and_data.first;
          auto& modified_tensor = get<decltype(tag)>(
              modified_neighbor_solutions.at(primary_neighbor));
          compute_hweno_modified_neighbor_solution<decltype(tag)>(
              make_not_null(&modified_tensor), *tensor, element, mesh,
              neighbor_data, primary_neighbor);
        }
        return '0';
      };
      expand_pack(wrap_hweno_neighbor_solution_one_tensor(Tags{}, tensors)...);
    } else if (weno_type_ == WenoType::SimpleWeno) {
      for (const auto& neighbor_and_data : neighbor_data) {
        const auto& neighbor = neighbor_and_data.first;
        const auto& data = neighbor_and_data.second;

        // Interpolate from neighbor onto self
        // Actually implement this by interpolating ---
        // - from the neighbor mesh
        // - onto self mesh, in self direction, with coord offset
        const auto& source_mesh = data.mesh;
        const auto& direction = neighbor.first;
        const auto target_1d_logical_coords =
            Weno_detail::local_grid_points_in_neighbor_logical_coords(
                mesh, source_mesh, element, direction);
        const intrp::RegularGrid<VolumeDim> interpolant(
            source_mesh, mesh, target_1d_logical_coords);
        modified_neighbor_solutions.insert(std::make_pair(
            neighbor, interpolant.interpolate(data.volume_data)));
      }
    } else {
      ERROR("WENO limiter not implemented for WenoType: " << weno_type_);
    }

    // Reconstruct from extrapolated data
    const auto wrap_reconstruct_one_tensor =
        [ this, &mesh, &
          modified_neighbor_solutions ](auto tag, const auto& tensor) noexcept {
      Weno_detail::reconstruct_from_weighted_sum<decltype(tag)>(
          tensor, mesh, neighbor_linear_weight_, modified_neighbor_solutions);
      return '0';
    };
    expand_pack(wrap_reconstruct_one_tensor(Tags{}, tensors)...);
    return true;  // cell_is_troubled
  }

 private:
  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                         const Weno<LocalDim, LocalTagList>& rhs) noexcept;

  WenoType weno_type_;
  double neighbor_linear_weight_;
  bool disable_for_debugging_;
};

template <size_t LocalDim, typename LocalTagList>
bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                const Weno<LocalDim, LocalTagList>& rhs) noexcept {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Weno<VolumeDim, TagList>& lhs,
                const Weno<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters
