// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags.hpp"       // IWYU pragma: keep
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/DirectionMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoGridHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t>
class Mesh;
/// \endcond

namespace SlopeLimiters {
namespace Hweno_detail {

// Caching class that holds various precomputed terms used in the constrained-
// fit algebra on each element. Two simplifying assumptions are made,
//
// 1. the element has at most one neighbor per dimension
//    This greatly reduces the number of possible values for the matrix A_jk and
//    simplifies its caching.
//
// 2. the mesh on every neighbor is the same as on the element
//    This simplifies the interface (only pass in one mesh vs many).
//
// Taken together, these assumptions enable the cached result for any element to
// be used for every element, greatly simplifying the overall caching in a
// simulation. However, of course, these assumptions are incompatible with both
// h-refinement and p-refinement.
template <size_t VolumeDim>
class HwenoConstrainedFitCache {
 public:
  HwenoConstrainedFitCache(const Element<VolumeDim>& element,
                           const Mesh<VolumeDim>& mesh) noexcept;

  const Matrix& get_Ajk_inverse_matrix(
      const Direction<VolumeDim>& primary_dir,
      const Direction<VolumeDim>& skipped_dir) const noexcept {
    ASSERT(primary_dir != skipped_dir,
           "Invalid inputs: primary_dir == skipped_dir == " << skipped_dir);
    return Ajk_inverse_matrices.at(primary_dir).at(skipped_dir);
  }

  DataVector quadrature_weights;
  DirectionMap<VolumeDim, Matrix> interpolation_matrices;
  DirectionMap<VolumeDim, DataVector>
      quadrature_weights_dot_interpolation_matrices;
  DirectionMap<VolumeDim, DirectionMap<VolumeDim, Matrix>> Ajk_inverse_matrices;
};

// Find the neighboring element where the tensor component given by `Tag` and
// `tensor_index` has the most different mean from `local_mean`. Note that the
// neighbor given by `primary_neighbor` is NOT included in the maximization.
template <typename Tag, size_t VolumeDim, typename Package>
std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>
find_neighbor_with_most_different_mean(
    const double local_mean, const size_t tensor_index,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor) noexcept {
  // Initialize with a negative value to guarantee that at least one comparison
  // will exceed this, because this will set the most different neighbor.
  double running_max_difference = -1.;
  std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>
      most_different_neighbor{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    if (neighbor == primary_neighbor) {
      // Do not count the primary neighbor toward the maximization
      continue;
    }
    const double difference = fabs(
        get<::Tags::Mean<Tag>>(neighbor_and_data.second.means)[tensor_index] -
        local_mean);
    if (difference > running_max_difference) {
      running_max_difference = difference;
      most_different_neighbor = neighbor;
    }
  }
  ASSERT(most_different_neighbor != primary_neighbor,
         "Logic error: erroneously found most_different_neighbor == "
         "primary_neighbor == "
             << primary_neighbor);
  return most_different_neighbor;
}

// Compute the vector b_j for the constrained fit. For details, see the
// documentation of `compute_hweno_modified_neighbor_solution` below.
template <typename Tag, size_t VolumeDim, typename Package>
DataVector compute_vector_bj(
    const Mesh<VolumeDim>& mesh, const size_t tensor_index,
    const DataVector& quadrature_weights,
    const DirectionMap<VolumeDim, Matrix>& interpolation_matrices,
    const DirectionMap<VolumeDim, DataVector>&
        quadrature_weights_dot_interpolation_matrices,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        skipped_neighbor) noexcept {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  DataVector bj(number_of_grid_points, 0.);

  // Loop over all neighbors
  for (const auto& neighbor_and_data : neighbor_data) {
    if (neighbor_and_data.first == skipped_neighbor) {
      continue;
    }

    const auto& direction = neighbor_and_data.first.first;
    const auto& neighbor_mesh = mesh;
    const auto& neighbor_quadrature_weights = quadrature_weights;
    const auto& interpolation_matrix = interpolation_matrices.at(direction);
    const auto& quadrature_weights_dot_interpolation_matrix =
        quadrature_weights_dot_interpolation_matrices.at(direction);

    const auto& neighbor_tensor_component =
        get<Tag>(neighbor_and_data.second.volume_data)[tensor_index];

    // Add terms from the primary neighbor
    if (neighbor_and_data.first == primary_neighbor) {
      for (size_t i = 0; i < neighbor_mesh.number_of_grid_points(); ++i) {
        for (size_t j = 0; j < number_of_grid_points; ++j) {
          bj[j] += neighbor_tensor_component[i] *
                   neighbor_quadrature_weights[i] * interpolation_matrix(i, j);
        }
      }
    }
    // Add terms from the other neighbors
    else {
      const double quadrature_weights_dot_u = [&]() noexcept {
        double result = 0.;
        for (size_t i = 0; i < neighbor_mesh.number_of_grid_points(); ++i) {
          result +=
              neighbor_tensor_component[i] * neighbor_quadrature_weights[i];
        }
        return result;
      }
      ();
      bj += quadrature_weights_dot_u *
            quadrature_weights_dot_interpolation_matrix;
    }
  }
  return bj;
}

// Solve the constrained fit problem that gives the HWENO modified solution,
// for one particular tensor component. For details, see documentation of
// `compute_hweno_modified_neighbor_solution` below.
template <typename Tag, size_t VolumeDim, typename Package>
void solve_constrained_fit(
    const gsl::not_null<DataVector*> constrained_fit_result,
    const DataVector& u, const size_t tensor_index,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        skipped_neighbor) noexcept {
  // Cache the linear algebra quantities.
  //
  // Because we don't yet support h/p-refinement, we only have to worry about
  // interior vs. boundary elements. With no refinement, all interior elements
  // have the same matrices, so a single shared cache suffices. The boundary
  // elements will have different matrices depending on which/how many sides
  // are boundaries; for simplicity, we give one cache to each boundary element.
  // This does cause some unneeded duplication, e.g., if multiple elements have
  // an external boundary on the lower-xi side only.
  //
  // The pointer `cache` is pointed into the correct cache from which to read
  // the linear algebra quantities for each particular element.
  //
  // Note that for fully general grids, with h- and p-refinement, the caching
  // will need to be *far* more complicated.
  static const HwenoConstrainedFitCache<VolumeDim> interior_cache(element,
                                                                  mesh);
  static std::unordered_map<ElementId<VolumeDim>,
                            HwenoConstrainedFitCache<VolumeDim>>
      boundary_cache{};
  const HwenoConstrainedFitCache<VolumeDim>* cache;
  // With no h-refinement, elements at external boundaries can be identified
  // by the number of neighbors.
  if (element.neighbors().size() == two_to_the(VolumeDim)) {
    cache = &interior_cache;
  } else {
    const auto& id = element.id();
    if (boundary_cache.find(id) == boundary_cache.end()) {
      boundary_cache.insert(std::make_pair(
          id, HwenoConstrainedFitCache<VolumeDim>(element, mesh)));
    }
    cache = &(boundary_cache.at(id));
  }

  const DataVector& w = cache->quadrature_weights;
  const DirectionMap<VolumeDim, DataVector>& w_dot_Ms =
      cache->quadrature_weights_dot_interpolation_matrices;
  const DirectionMap<VolumeDim, Matrix>& Ms = cache->interpolation_matrices;
  const Matrix& Ajk_inverse = cache->get_Ajk_inverse_matrix(
      primary_neighbor.first, skipped_neighbor.first);
  const DataVector bj =
      compute_vector_bj<Tag>(mesh, tensor_index, w, Ms, w_dot_Ms, neighbor_data,
                             primary_neighbor, skipped_neighbor);

  const size_t number_of_points = bj.size();
  DataVector A_inverse_b(number_of_points, 0.);
  DataVector A_inverse_w(number_of_points, 0.);
  for (size_t j = 0; j < number_of_points; ++j) {
    for (size_t k = 0; k < number_of_points; ++k) {
      A_inverse_b[j] += Ajk_inverse(j, k) * bj[k];
      A_inverse_w[j] += Ajk_inverse(j, k) * w[k];
    }
  }

  // Compute Lagrange multiplier:
  const double lagrange_multiplier =
  // Clang 6 believes the capture of w to be incorrect, presumably because
  // it is checking the storage duration of the object referenced by w,
  // rather than w itself. gcc (correctly, I believe) requires the capture,
  // at least in versions 6 and 7.
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-lambda-capture"
#endif  // __clang__
      [&number_of_points, &A_inverse_b, &A_inverse_w, &u, &w ]() noexcept {
#if defined(__clang__) && __clang_major__ > 4
#pragma GCC diagnostic pop
#endif  // __clang__
    double numerator = 0.;
    double denominator = 0.;
    for (size_t j = 0; j < number_of_points; ++j) {
      numerator += w[j] * (A_inverse_b[j] - u[j]);
      denominator += w[j] * A_inverse_w[j];
    }
    return -numerator / denominator;
  }
  ();

  // Compute solution:
  *constrained_fit_result = A_inverse_b + lagrange_multiplier * A_inverse_w;
}

}  // namespace Hweno_detail

/*!
 * \ingroup SlopeLimitersGroup
 * \brief Compute the HWENO modified solution for a particular tensor,
 * and associated with a particular neighbor element
 *
 * The HWENO limiter reconstructs its new solution from a linear combination of
 * the local DG solution and a "modified" solution from each neighbor element.
 * This function computes the modified solution for a particular tensor and
 * neighbor, following Section 3 of \cite Zhu2016.
 *
 * The modified solution associated with a particular neighbor (the "primary"
 * neighbor) is obtained by solving a constrained fit over the local element,
 * the primary neighbor, and the other neighbors of the local element.
 * This fit seeks to minimize in a least-squared sense:
 * 1. The distance between the modified solution and the original solution on
 *    the primary neighbor.
 * 1. The distance between the cell average of the modified solution and the
 *    cell average of the original solution on each _other_ neighbor. Note
 *    however that one other neighbor is skipped in this minimization: the
 *    neighbor where the tensor component being modified has the most different
 *    mean from the local element (this prevents an outlier, e.g., near a shock,
 *    from biasing the fit).
 *
 * Simultaneously, the solution must satisfying the constraint that
 * 1. The cell average of the modified solution on the local element must equal
 *    the cell average of the local element's original solution.
 *
 * Below we give the mathematical form of the constraints described above and
 * show how these are translated into a numerical algorithm.
 *
 * Consider an element \f$I_0\f$ with neighbors \f$I_1, I_2, ...\f$. For a
 * given tensor component \f$u\f$, the values on each of these elements are
 * \f$u_0, u_1, u_2, ...\f$. Taking for the sake of example the primary
 * neighbor to be \f$I_1\f$, the modified solution \f$\phi\f$ must minimize
 *
 * \f[
 * \chi^2 = \int_{I_1} (\phi - u_1)^2 dV
 *          + \sum_{l} \left( \int_{I_l} ( \phi - u_l ) dV \right)^2,
 * \f]
 *
 * subject to the constaint
 *
 * \f[
 * C = \int_{I_0} ( \phi - u_0 ) dV = 0.
 * \f]
 *
 * where \f$l\f$ ranges over all other (non-primary) neighbors but one,
 * skipping also the neighbor where the mean of \f$u\f$ is the most different
 * from the mean of \f$u_0\f$. Note that in 1D, this implies that \f$l\f$
 * ranges over the empty set; for each modified solution, one neighbor is the
 * primary neighbor and the other is the skipped neighbor.
 *
 * The integrals are evaluated by quadrature. We denote the quadrature weights
 * by \f$\omega_j\f$ and the values of some data \f$X\f$ at the quadrature
 * nodes by \f$X_j\f$. We use subscripts \f$i,j\f$ to denote quadrature nodes
 * on the neighbor and local elements, respectively. The minimization becomes
 *
 * \f[
 * \chi^2 = \sum_{i} \omega_{1,i} ( \phi_{i} - u_{1,i} )^2
 *          + \sum_{l} \left( \sum_{i} \omega_{l,i} ( \phi_{i} - u_{l,i} )
 *                     \right)^2,
 * \f]
 *
 * subject to the constraint
 *
 * \f[
 * C = \sum_{j} \omega_{0,j} ( \phi_{j} - u_{0,j} ) dV = 0.
 * \f]
 *
 * Note that \f$\phi\f$ is a function defined on the local element \f$I_0\f$,
 * and so is fully represented by its values \f$\phi_{j}\f$ at the quadrature
 * points on this element. When evaluating \f$\phi\f$ on element \f$I_l\f$, we
 * obtain the function values \f$\phi_{i}\f$ by polynomial extrapolation,
 * \f$\phi_{i} = \sum_{j} M^{(l)}_{ij} \phi_{j}\f$, where \f$M^{(l)}_{ij}\f$ is
 * the interpolation/extrapolation matrix that interpolates data defined at grid
 * points \f$x_j\f$ and evalutes it at grid points \f$x_i\f$. Thus,
 *
 * \f[
 * \chi^2 = \sum_{i} \omega_{1,i}
 *                   \left( \sum_{j} M^{(1)}_{ij} \phi_{j} - u_{1,i} \right)^2
 *          + \sum_{l} \left(
 *                     \sum_{i} \omega_{l,i}
 *                              \left(
 *                              \sum_{j} M^{(l)}_{ij} \phi_{j} - u_{l,i}
 *                              \right)
 *                     \right)^2.
 * \f]
 *
 * The solution to this optimization problem is found in the standard way,
 * using a Lagrange multiplier \f$\lambda\f$ to impose the constraint:
 *
 * \f[
 * 0 = \frac{d}{d \phi_{j}} \left( \chi^2 + \lambda C \right).
 * \f]
 *
 * Working out the differentiation with respect to \f$\phi_j\f$ leads to the
 * linear problem that must be inverted to obtain the solution,
 *
 * \f[
 * 0 = A_{jk} \phi_{k} - b_{j} - \lambda \omega_{0,j},
 * \f]
 *
 * where
 *
 * \f{align*}{
 * A_{jk} &= \sum_{i} \left( \omega_{1,i} M^{(1)}_{ij} M^{(1)}_{ik} \right)
 *          + \sum_{l} \left(
 *                     \sum_{i} \left( \omega_{l,i} M^{(l)}_{ik} \right)
 *                     \cdot
 *                     \sum_{i} \left( \omega_{l,i} M^{(l)}_{ij} \right)
 *                     \right)
 * \\
 * b_{j} &= \sum_{i} \left( \omega_{1,i} u_{1,i} M^{(1)}_{ij} \right)
 *         + \sum_{l} \left(
 *                    \sum_{i} \left( \omega_{l,i} u_{l,i} \right)
 *                    \cdot
 *                    \sum_{i} \left( \omega_{l,i} M^{(l)}_{ij} \right)
 *                    \right).
 * \f}
 *
 * Finally, the solution to the constrained fit is
 *
 * \f{align*}{
 * \lambda &= - \frac{ \sum_{j} \omega_{0,j}
 *                              \left( (A^{-1})_{jk} b_{k} - u_{0,j} \right)
 *              }{ \sum_{j} \omega_{0,j} (A^{-1})_{jk} \omega_{0,k} }
 * \\
 * \phi_{j} &= (A^{-1})_{jk} ( b_{k} + \lambda \omega_{0,k} ).
 * \f}
 *
 * Note that the matrix \f$A\f$ does not depend on the values of the tensor
 * \f$u\f$, so its inverse \f$A^{-1}\f$ can be precomputed and stored.
 *
 * Note also that the implementation currently does not support h- or
 * p-refinement; this is checked by some assertions. The implementation is
 * untested for grids where elements are curved, and it should not be expected
 * to work in these cases.
 */
template <typename Tag, size_t VolumeDim, typename Package>
void compute_hweno_modified_neighbor_solution(
    const gsl::not_null<db::item_type<Tag>*> modified_tensor,
    const db::item_type<Tag>& local_tensor, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, Package,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>&
        primary_neighbor) noexcept {
  ASSERT(Weno_detail::check_element_has_one_similar_neighbor_in_direction(
             element, primary_neighbor.first),
         "Found some amount of h-refinement; this is not supported");
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    ASSERT(neighbor_and_data.second.mesh == mesh,
           "Found some amount of p-refinement; this is not supported");
  });

  for (size_t tensor_index = 0; tensor_index < local_tensor.size();
       ++tensor_index) {
    const auto& tensor_component = local_tensor[tensor_index];
    const auto skipped_neighbor =
        Hweno_detail::find_neighbor_with_most_different_mean<Tag>(
            mean_value(tensor_component, mesh), tensor_index, neighbor_data,
            primary_neighbor);
    Hweno_detail::solve_constrained_fit<Tag>(
        make_not_null(&(*modified_tensor)[tensor_index]),
        local_tensor[tensor_index], tensor_index, element, mesh, neighbor_data,
        primary_neighbor, skipped_neighbor);
  }
}

}  // namespace SlopeLimiters
