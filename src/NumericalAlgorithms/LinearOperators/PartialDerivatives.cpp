// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <cstddef>
#include <functional>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace partial_derivatives_detail {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  dgemm_<true>(
      'N', 'N',
      matrix.rows(),              // rows of matrix and result
      size / matrix.columns(),    // columns of result and input
      matrix.columns(),           // columns of matrix and rows of input
      1.0,                        // overall multiplier
      matrix.data(),              // matrix
      matrix.spacing(),           // rows of matrix including padding
      input,                      // input
      matrix.columns(),           // rows of input
      add_to_result ? 1.0 : 0.0,  // overwrite output with result or add to it
      result,                     // result
      matrix.rows());             // rows of result
}
void apply_matrix_in_first_dim(std::complex<double>* result,
                               const std::complex<double>* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  // BLAS zgemm operates on complex matrices, so we need to copy the real matrix
  // to a complex matrix with zero imaginary part before calling zgemm.
  // Possible performance optimization: avoid the copy here by storing the
  // complex matrix in a static cache. We probably only want to add this to
  // Spectral.hpp once profiling shows that it becomes necessary.
  const blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
      matrix_complex{matrix};
  zgemm_<true>('N', 'N',
               matrix.rows(),            // rows of matrix and result
               size / matrix.columns(),  // columns of result and input
               matrix.columns(),         // columns of matrix and rows of input
               std::complex{1.0, 0.0},   // overall multiplier
               matrix_complex.data(),    // matrix
               matrix.spacing(),         // rows of matrix including padding
               input,                    // input
               matrix.columns(),         // rows of input
               std::complex{add_to_result ? 1.0 : 0.0,
                            0.0},  // overwrite output with result or add to it
               result,             // result
               matrix.rows());     // rows of result
  // This implementation is ~1.35x slower than the implementation above (based
  // on the "Partial derivatives complex" benchmark in
  // Test_PartialDerivatives.cpp).
  //   DataVector buffer(size * 2);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 reinterpret_cast<const double*>(input), 2, size);
  //   apply_matrix_in_first_dim(buffer.data(),
  //                             reinterpret_cast<const double*>(result),
  //                             matrix, size * 2, add_to_result);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 buffer.data(), size, 2);
}
#ifdef SPECTRE_KOKKOS
template <size_t DerivDim, size_t Dim, bool AddToResult>
void apply_matrix_in_dim(
    Kokkos::View<double**> result,        // [total_num_points, num_components]
    const Kokkos::View<double**>& input,  // [total_num_points, num_components]
    const MatrixViewRO& matrix,           // [num_points_this_dim^2]
    const Mesh<Dim>& mesh,
    const std::array<Kokkos::View<double*>,  // [total_num_points]
                     Dim * Dim>& inv_jacobian) {
  const size_t num_components = input.extent(1);
  const std::array<size_t, Dim> extents = mesh.extents().indices();
  const std::array<size_t, Dim - 1> extents_transverse =
      all_but_specified_element_of(extents, DerivDim);
  const size_t total_num_points = mesh.extents().product();
  const size_t num_points_this_dim = extents[DerivDim];
  const size_t num_points_transverse = total_num_points / num_points_this_dim;
  ASSERT(input.extent(0) == total_num_points,
         "Input has " << input.extent(0) << " points, but mesh has "
                      << total_num_points << " points.");
  ASSERT(matrix.extent(0) == matrix.extent(1), "Matrix must be square, but has "
                                                   << matrix.extent(0) << "x"
                                                   << matrix.extent(1));
  ASSERT(matrix.extent(0) == num_points_this_dim,
         "Matrix has " << matrix.extent(0) << " rows, but mesh has "
                       << num_points_this_dim << " points in dim " << DerivDim
                       << '.');

  // Precompute strides for column-major flattening
  std::array<size_t, Dim> strides;
  {
    size_t s = 1;
    for (size_t d = 0; d < Dim; ++d) {
      strides[d] = s;
      s *= extents[d];
    }
  }

  // Precompute storage indices for the inverse Jacobian. Also store as pointers
  // because capturing the full `inv_jacobian` by value and indexing it in
  // nested Kokkos lambdas has a surprisingly large overhead.
  using JacobianStructure = Tensor_detail::Structure<
      Symmetry<1, 2>,
      Tensor_detail::TensorIndexType<Dim, UpLo::Up, Frame::NoFrame,
                                     IndexType::Spatial>,
      Tensor_detail::TensorIndexType<Dim, UpLo::Lo, Frame::NoFrame,
                                     IndexType::Spatial>>;
  std::array<const double*, Dim> inv_jac_components{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(inv_jac_components, d) =
        inv_jacobian[JacobianStructure::get_storage_index(DerivDim, d)].data();
  }

  // Parallelization strategy on GPU hardware:
  //
  // We currently sweep over the tensor data Dim times, once for each logical
  // dimension, and assemble the result by contracting with the Jacobian:
  //   \partial_i u = J^\hat{j}_i \partial_\hat{j} u
  // (calling this function Dim times, the first time with AddToResult=false and
  // then later with AddToResult=true to do the sum over $\hat{j}$). In each
  // sweep we parallelize over all independent "stripes" of tensor data in the
  // derivative dimension (typically O(10^3) stripes of length N~6-20 per DG
  // element) by assigning a team of threads (thread block) to each stripe. We
  // load the stripe data into shared team memory, then apply the
  // differentiation matrix to the stripe by parallelizing over the N outputs /
  // rows of the matrix with the thread team (TeamThreadRange) and parallelizing
  // over the N inputs / columns of the matrix with a vectorized sum
  // (ThreadVectorRange). The number of threads in a team (team size) and the
  // number of vector lanes (vector length) are configurable via the TeamPolicy.
  // Currently we just let Kokkos pick the team size automatically and set the
  // vector length to 1 because that gave the best result in a preliminary
  // benchmark with 32 random tensor components in a 3D element with 20^3 grid
  // points (18.5 us on an A100).
  //
  // Notes on performance and possible improvements:
  //
  // - We currently parallelize a single element over GPU threads. It would be
  //   better to batch elements together to saturate the GPU with more work.
  //   This would just extend the `num_components` over the tensor data in all
  //   batched elements to apply the logical differentiation, with the caveat
  //   that the Jacobian data has to be strided over the different elements.
  // - Currently the amount of work per team is very small because it works only
  //   on a single stripe of tensor data with length N~6-20. It could be better
  //   to have each team work on multiple stripes at once, each thread in the
  //   team working on one stripe, and to batch these threads/stripes into
  //   vector lanes so that one warp completes multiple stripes at once (doing
  //   the full matrix multiply in one vector lane). This approach probably
  //   needs to batch elements together or launch kernels for multiple elements
  //   at once to saturate the GPU with enough teams (thread blocks).
  // - The data layout of the input and output data is very important and hasn't
  //   been tested very thoroughly yet, in particular in situations where memory
  //   bandwidth is the bottleneck. It is also important to balance the optimal
  //   data layout for differentiation with the optimal layout for pointwise RHS
  //   evaluations, which is likely different (differentiation prefers locality
  //   of grid points and pointwise evaluations prefer locality of tensor
  //   components). Also, large systems of PDEs like GH might prefer locality of
  //   tensor components whereas small systems like scalar wave might prefer
  //   locality of grid points.

  const size_t num_teams = num_components * num_points_transverse;
  const size_t shmem_size = num_points_this_dim * (1 + Dim) * sizeof(double);
  using TeamPolicy = Kokkos::TeamPolicy<>;
  TeamPolicy team_policy(num_teams, Kokkos::AUTO);
  team_policy = team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

  Kokkos::parallel_for(
      "apply_matrix_in_dim", team_policy,
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team_member) {
        const size_t team_rank = team_member.league_rank();
        const size_t component_index = team_rank / num_points_transverse;
        size_t remaining = team_rank % num_points_transverse;
        // Decode stripe index in the transverse dims
        std::array<size_t, Dim - 1> stripe_index{};
        for (size_t t = 0; t < Dim - 1; ++t) {
          stripe_index[t] = remaining % extents_transverse[t];
          remaining /= extents_transverse[t];
        }
        // Map stripe to an offset and stride into the column-major data layout
        size_t base_offset = 0;
        {
          size_t t = 0;
          for (size_t d = 0; d < Dim; ++d) {
            if (d == DerivDim) {
              continue;
            }
            base_offset += stripe_index[t++] * strides[d];
          }
        }
        const size_t stride = strides[DerivDim];

        // Preload the input data for this component and stripe
        double* shmem = (double*)team_member.team_shmem().get_shmem(shmem_size);
        Kokkos::View<double*, Kokkos::MemoryUnmanaged> input_stripe(
            shmem, num_points_this_dim);
        Kokkos::View<double* [Dim], Kokkos::MemoryUnmanaged> inv_jac_stripe(
            shmem + num_points_this_dim, num_points_this_dim);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_points_this_dim),
            [=](int j) {
              const size_t point_index =
                  base_offset + static_cast<size_t>(j) * stride;
              input_stripe[j] = input(point_index, component_index);
              for (size_t d = 0; d < Dim; ++d) {
                inv_jac_stripe(j, d) = inv_jac_components[d][point_index];
              }
            });

        team_member.team_barrier();

        // Apply the differentiation matrix and the Jacobian
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, num_points_this_dim),
            [=](int i) {
              // Dense matrix-vector product along DerivDim
              double sum = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, num_points_this_dim),
                  [=](int j, double& local_sum) {
                    local_sum += matrix(i, j) * input_stripe[j];
                  },
                  sum);
              // Contract with Jacobian and write out to result
              const size_t point_index =
                  base_offset + static_cast<size_t>(i) * stride;
              for (size_t d = 0; d < Dim; ++d) {
                const double contracted = inv_jac_stripe(i, d) * sum;
                const size_t deriv_component_index = component_index * Dim + d;
                (void)result;  // capture `result` for `if constexpr`
                if constexpr (AddToResult) {
                  result(point_index, deriv_component_index) += contracted;
                } else {
                  result(point_index, deriv_component_index) = contracted;
                }
              }
            });
      });
}
// generate instantations
#define INSTANTIATE_APPLY_MATRIX_IN_DIM(DerivDim, Dim, AddToResult)       \
  template void apply_matrix_in_dim<DerivDim, Dim, AddToResult>(          \
      Kokkos::View<double**> result, const Kokkos::View<double**>& input, \
      const MatrixViewRO& matrix, const Mesh<Dim>& mesh,                  \
      const std::array<Kokkos::View<double*>, Dim * Dim>& inv_jacobian);
INSTANTIATE_APPLY_MATRIX_IN_DIM(0, 1, false)
INSTANTIATE_APPLY_MATRIX_IN_DIM(0, 2, false)
INSTANTIATE_APPLY_MATRIX_IN_DIM(0, 3, false)
INSTANTIATE_APPLY_MATRIX_IN_DIM(1, 2, true)
INSTANTIATE_APPLY_MATRIX_IN_DIM(1, 3, true)
INSTANTIATE_APPLY_MATRIX_IN_DIM(2, 3, true)
#undef INSTANTIATE_APPLY_MATRIX_IN_DIM
#endif  // SPECTRE_KOKKOS
}  // namespace partial_derivatives_detail

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const gsl::not_null<gsl::span<typename DataType::value_type>*> buffer,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  static_assert(
      Dim > 0 and Dim < 4,
      "logical_partial_derivative is only implemented for 1, 2, and 3d");
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(buffer->size() >= num_grid_points,
         "The buffer in logical_partial_derivative must be at least of size "
             << num_grid_points << " but is of size " << buffer->size());

  set_number_of_grid_points(logical_derivative_of_u,
                            mesh.number_of_grid_points());
  if (Dim == 3 and mesh.basis(1) == Spectral::Basis::SphericalHarmonic) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      const Matrix& differentiation_matrix_xi =
          Spectral::differentiation_matrix(mesh.slice_through(0));
      const auto& ylm = ylm::get_spherepack_cache(mesh.extents(1) - 1);
      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 0_st)).data(),
            u[storage_index].data(), differentiation_matrix_xi,
            num_grid_points);
        const auto du = std::array{
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 1_st)).data(),
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 2_st)).data()};
        ylm.gradient_all_offsets(du, make_not_null(u[storage_index].data()),
                                 mesh.extents(0));
      }
    } else {
      ERROR(
          "Support for complex numbers with spherical harmonics is not yet "
          "implemented for logical_partial_derivative.");
    }
  } else if (Dim == 3 and mesh.basis(0) == Spectral::Basis::ZernikeB3) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      const size_t n_r = mesh.extents(0);
      const size_t l_max = mesh.extents(1) - 1;
      ASSERT(l_max < 2 * n_r - 1,
             "ZernikeB3 radial resolution is insufficient for the requested "
             "angular resolution. Need l_max < 2*n_r-1, but l_max="
                 << l_max << " and n_r=" << n_r);
      const auto& ylm = ylm::get_spherepack_cache(l_max);
      const size_t n_spectral = ylm.spectral_size();
      const size_t n_components = u.size();

      // Angular derivatives
      for (size_t storage_index = 0; storage_index < n_components;
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto du_ang = std::array{
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 1_st)).data(),
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 2_st)).data()};
        ylm.gradient_all_offsets(du_ang, make_not_null(u[storage_index].data()),
                                 n_r);
      }

      // Radial derivatives, using parity-aware differentiation.
      // Functions on the ball with angular degree l behave as r^l near the
      // origin, giving even parity for even l and odd parity for odd l.
      const Matrix& D_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB3, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& D_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB3, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);

      // n_spectral is the SPHEREPACK buffer size = 2*(l_max+1)^2, including
      // padding. Valid modes total (l_max+1)^2; get_spherepack_cache always
      // sets m_max = l_max so each degree l contributes 2l+1 valid modes.
      const size_t n_valid_modes = (l_max + 1) * (l_max + 1);
      const size_t n_even_modes = (l_max % 2 == 0)
                                      ? (l_max + 1) * (l_max + 2) / 2
                                      : l_max * (l_max + 1) / 2;
      const size_t n_odd_modes = n_valid_modes - n_even_modes;

      const size_t spec_total = n_components * n_spectral * n_r;
      const size_t even_total = n_even_modes * n_components * n_r;
      const size_t odd_total = n_odd_modes * n_components * n_r;
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const auto all_buf = cpp20::make_unique_for_overwrite<double[]>(
          spec_total + 2 * (even_total + odd_total));
      double* const spec_buf = all_buf.get();
      double* const even_data = spec_buf + spec_total;      // NOLINT
      double* const odd_data = even_data + even_total;      // NOLINT
      double* const even_result = odd_data + odd_total;     // NOLINT
      double* const odd_result = even_result + even_total;  // NOLINT

      // SH physical -> spectral
      for (size_t n = 0; n < n_components; ++n) {
        ylm.phys_to_spec_all_offsets(
            make_not_null(spec_buf + n * n_spectral * n_r),  // NOLINT
            make_not_null(u[n].data()), n_r);
      }

      // Gather even-l and odd-l radial profiles into compact buffers.
      // Mode at SPHEREPACK offset s has its n_r values at
      // spec_buf[...+s*n_r].
      ylm::SpherepackIterator spec_iter{l_max, ylm.m_max()};
      for (size_t n = 0; n < n_components; ++n) {
        const double* spec = spec_buf + n * n_spectral * n_r;    // NOLINT
        double* even_dest = even_data + n * n_even_modes * n_r;  // NOLINT
        double* odd_dest = odd_data + n * n_odd_modes * n_r;     // NOLINT
        size_t even_col = 0;
        size_t odd_col = 0;
        spec_iter.reset();
        while (spec_iter) {
          const size_t s = spec_iter();
          if (spec_iter.l() % 2 == 0) {
            std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                      even_dest + even_col * n_r);           // NOLINT
            ++even_col;
          } else {
            std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                      odd_dest + odd_col * n_r);             // NOLINT
            ++odd_col;
          }
          ++spec_iter;
        }
      }

      // Apply radial differentiation matrices
      if (n_even_modes > 0) {
        partial_derivatives_detail::apply_matrix_in_first_dim(
            even_result, even_data, D_even, even_total);
      }
      if (n_odd_modes > 0) {
        partial_derivatives_detail::apply_matrix_in_first_dim(
            odd_result, odd_data, D_odd, odd_total);
      }

      // Copy differentiated radial profiles back to spectral buffer
      for (size_t n = 0; n < n_components; ++n) {
        double* spec = spec_buf + n * n_spectral * n_r;  // NOLINT
        const double* even_src =
            even_result + n * n_even_modes * n_r;                    // NOLINT
        const double* odd_src = odd_result + n * n_odd_modes * n_r;  // NOLINT
        size_t even_col = 0;
        size_t odd_col = 0;
        spec_iter.reset();
        while (spec_iter) {
          const size_t s = spec_iter();
          if (spec_iter.l() % 2 == 0) {
            std::copy(even_src + even_col * n_r,        // NOLINT
                      even_src + (even_col + 1) * n_r,  // NOLINT
                      spec + s * n_r);                  // NOLINT
            ++even_col;
          } else {
            std::copy(odd_src + odd_col * n_r,        // NOLINT
                      odd_src + (odd_col + 1) * n_r,  // NOLINT
                      spec + s * n_r);                // NOLINT
            ++odd_col;
          }
          ++spec_iter;
        }
      }

      // SH spectral -> physical
      for (size_t n = 0; n < n_components; ++n) {
        const auto u_tensor_index = u.get_tensor_index(n);
        ylm.spec_to_phys_all_offsets(
            make_not_null(
                // NOLINTNEXTLINE(readability-redundant-smartptr-get)
                logical_derivative_of_u->get(prepend(u_tensor_index, 0_st))
                    .data()),
            make_not_null(spec_buf + n * n_spectral * n_r), n_r);  // NOLINT
      }
    } else {
      ERROR(
          "Support for complex numbers with true filled sphere is not yet "
          "implemented for logical_partial_derivative.");
    }
  } else if (Dim == 2 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);
      const size_t M = n_phi / 2;

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);

        // Do phi derivative
        dgemm_<true>(
            'N',
            'T',  // transpose differentiation matrix to act on rows of data
            n_r, n_phi, n_phi, 1.0, u[storage_index].data(), n_r,
            diff_matrix_phi.data(), diff_matrix_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(phi_deriv_tensor_index).data(), n_r);

        // Get u in terms of r and m index
        dgemm_<true>('N', 'T', n_r, n_phi, n_phi, 1.0, u[storage_index].data(),
                     n_r, nodal_to_modal_phi.data(),
                     nodal_to_modal_phi.spacing(), 0.0,
                     // NOLINTNEXTLINE(readability-redundant-smartptr-get)
                     logical_derivative_of_u->get(r_deriv_tensor_index).data(),
                     n_r);  // C and ldc

        // Do radial derivative
        // m = 0
        dgemv_('N', n_r, n_r, 1.0, diff_matrix_r_even.data(),
               diff_matrix_r_even.spacing(),
               // NOLINTNEXTLINE(readability-redundant-smartptr-get)
               logical_derivative_of_u->get(r_deriv_tensor_index).data(), 1,
               0.0,  // beta = 0.0
               (*buffer).data(), 1);
        // j is index into Fourier modal vector
        // {u_0, u_1, u_{-1}, u_2, u_{-2}, ... , u_M, u_{-M}}
        size_t j = 1;
        for (size_t m = 1; m <= M; ++m) {
          const Matrix& diff_r =
              m % 2 == 0 ? diff_matrix_r_even : diff_matrix_r_odd;
          dgemm_<true>(
              'N', 'N', n_r, 2, n_r, 1.0, diff_r.data(), diff_r.spacing(),
              // NOLINTNEXTLINE(readability-redundant-smartptr-get)
              logical_derivative_of_u->get(r_deriv_tensor_index).data() +
                  j * n_r,
              n_r,
              0.0,  // beta = 0.0
              // NOLINTNEXTLINE
              (*buffer).data() + j * n_r, n_r);
          j += 2;
        }
        // Transform from m back to phi
        dgemm_<true>(
            'N', 'T', n_r, n_phi, n_phi, 1.0, (*buffer).data(), n_r,
            modal_to_nodal_phi.data(), modal_to_nodal_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(r_deriv_tensor_index).data(), n_r);
      }
    } else {
      ERROR(
          "Support for complex numbers with disk is not yet implemented for "
          "logical_partial_derivative.");
    }
  } else if (Dim == 3 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_z = mesh.extents(2);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& diff_matrix_z =
          Spectral::differentiation_matrix(mesh.slice_through(2));

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const DataType& u_component = u[storage_index];
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        DataType& r_deriv_component =
            logical_derivative_of_u->get(r_deriv_tensor_index);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);
        DataType& phi_deriv_component =
            logical_derivative_of_u->get(phi_deriv_tensor_index);
        const auto z_deriv_tensor_index = prepend(u_tensor_index, 2_st);
        DataType& z_deriv_component =
            logical_derivative_of_u->get(z_deriv_tensor_index);

        // phi derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r, n_phi * n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_phi,
            num_grid_points);
        raw_transpose(make_not_null(phi_deriv_component.data()), buffer->data(),
                      n_phi * n_z, n_r);

        // See comments in disk_apply() in the .tpp for considerations about
        // the following code. For this function, we don't have enough buffer
        // space to cleanly use the approach in the .tpp, so we use the
        // masking approach instead. Note that timing shows this to only be a
        // few percent slower

        // reuse transposed data to go to angular spectral space
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            nodal_to_modal_phi, num_grid_points);
        // r_deriv_component holds transposed angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_phi * n_z, n_r);
        std::copy(buffer->data(), buffer->data() + num_grid_points,
                  z_deriv_component.data());
        // buffer and z_deriv_component hold angular modal rep

        // buffer stores even components, z_deriv_component stores odd
        for (size_t k = 0; k < n_z; ++k) {
          size_t offset = k * n_phi * n_r;
          for (size_t i = 0; i < n_phi; ++i) {
            if (i == 0) {
              // This is an even column -> zero odd vals
              std::fill(z_deriv_component.data() + offset,
                        z_deriv_component.data() + offset + n_r, 0.0);
            } else {
              if ((i - 1) / 2 % 2 == 1) {
                // This is an even column with even next to it -> zero odd vals
                std::fill(z_deriv_component.data() + offset,
                          z_deriv_component.data() + offset + 2 * n_r, 0.0);
              } else {
                // This is an even column with odd next to it -> zero even vals
                std::fill(buffer->data() + offset,
                          buffer->data() + offset + 2 * n_r, 0.0);
              }
              ++i;
              offset += n_r;
            }
            offset += n_r;
          }
        }
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), buffer->data(), diff_matrix_r_even,
            num_grid_points, false);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            diff_matrix_r_odd, num_grid_points, true);
        // r_deriv_component holds the radial derivative in angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_r, n_phi * n_z);
        // buffer holds transposed radial derivative in angular modal rep

        partial_derivatives_detail::apply_matrix_in_first_dim(
            z_deriv_component.data(), buffer->data(), modal_to_nodal_phi,
            num_grid_points);
        // z_deriv_component holds transposed radial derivative in angular nodal
        // rep

        raw_transpose(make_not_null(r_deriv_component.data()),
                      z_deriv_component.data(), n_phi * n_z, n_r);

        // z derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r * n_phi, n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_z,
            num_grid_points);
        raw_transpose(make_not_null(z_deriv_component.data()), buffer->data(),
                      n_z, n_r * n_phi);
      }
    } else {
      ERROR(
          "Support for complex numbers with cylinder is not yet implemented "
          "for logical_partial_derivative.");
    }
  } else {
    const Matrix empty_matrix{};
    std::array<std::reference_wrapper<const Matrix>, Dim> diff_matrices{
        make_array<Dim, std::reference_wrapper<const Matrix>>(empty_matrix)};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(diff_matrices, d) =
          std::cref(Spectral::differentiation_matrix(mesh.slice_through(d)));
    }

    // It would be possible to check if the memory is contiguous and then
    // differentiate all components at once. Note that the buffer in that case
    // would also need to be the size of all components.
    for (size_t storage_index = 0; storage_index < u.size(); ++storage_index) {
      const auto u_tensor_index = u.get_tensor_index(storage_index);
      const auto xi_deriv_tensor_index = prepend(u_tensor_index, 0_st);
      partial_derivatives_detail::apply_matrix_in_first_dim(
          // NOLINTNEXTLINE(readability-redundant-smartptr-get)
          logical_derivative_of_u->get(xi_deriv_tensor_index).data(),
          u[storage_index].data(), diff_matrices[0].get(), num_grid_points);
      for (size_t i = 1; i < Dim; ++i) {
        const auto deriv_tensor_index = prepend(u_tensor_index, i);
        DataType& deriv_component =
            logical_derivative_of_u->get(deriv_tensor_index);
        size_t chunk_size =
            diff_matrices[0].get().rows() *
            (i == 1 ? 1 : gsl::at(diff_matrices, 1).get().rows());
        raw_transpose(make_not_null(deriv_component.data()),
                      u[storage_index].data(), chunk_size,
                      num_grid_points / chunk_size);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), deriv_component.data(),
            gsl::at(diff_matrices, i).get(), num_grid_points);
        chunk_size =
            i == 1 ? (Dim == 2 ? gsl::at(diff_matrices, 1).get().rows()
                               : gsl::at(diff_matrices, 1).get().rows() *
                                     gsl::at(diff_matrices, 2).get().rows())
                   : gsl::at(diff_matrices, 2).get().rows();
        raw_transpose(make_not_null(deriv_component.data()), buffer->data(),
                      chunk_size, num_grid_points / chunk_size);
      }
    }
  }
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  using ValueType = typename DataType::value_type;  // double or complex<double>
  std::vector<ValueType> buffer(mesh.number_of_grid_points());
  gsl::span<ValueType> buffer_view{buffer.data(), buffer.size()};
  logical_partial_derivative(logical_derivative_of_u,
                             make_not_null(&buffer_view), u, mesh);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
auto logical_partial_derivative(const Tensor<DataType, SymmList, IndexList>& u,
                                const Mesh<Dim>& mesh)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      result{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&result), u, mesh);
  return result;
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>& logical_partial_derivative_of_u,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  for (size_t storage_index = 0;
       storage_index < Tensor<DataType, SymmList, IndexList>::size();
       ++storage_index) {
    const auto u_multi_index =
        Tensor<DataType, SymmList,
               IndexList>::structure::get_canonical_tensor_index(storage_index);
    for (size_t i = 0; i < Dim; i++) {
      const auto du_multi_index = prepend(u_multi_index, i);
      du->get(du_multi_index) =
          inverse_jacobian.get(0, i) *
          logical_partial_derivative_of_u.get(prepend(u_multi_index, 0_st));
      for (size_t j = 1; j < Dim; j++) {
        du->get(du_multi_index) +=
            inverse_jacobian.get(j, i) *
            logical_partial_derivative_of_u.get(prepend(u_multi_index, j));
      }
    }
  }
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      logical_partial_derivative_of_u{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&logical_partial_derivative_of_u), u,
                             mesh);
  partial_derivative<DataType, SymmList, IndexList>(
      du, logical_partial_derivative_of_u, inverse_jacobian);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
auto partial_derivative(
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>
      result{mesh.number_of_grid_points()};
  partial_derivative(make_not_null(&result), u, mesh, inverse_jacobian);
  return result;
}

#define GET_DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define GET_TENSOR(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      Frame::ElementLogical >                                                  \
          logical_partial_derivative(const GET_TENSOR(data) < GET_DTYPE(data), \
                                     GET_DIM(data), GET_FRAME(data) > &u,      \
                                     const Mesh<GET_DIM(data)>& mesh);         \
  template void                                                                \
      partial_derivative<GET_DTYPE(data), GET_TENSOR(data) < GET_DTYPE(data),  \
                         GET_DIM(data), GET_FRAME(data)>::symmetry,            \
      GET_TENSOR(                                                              \
          data)<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>::index_list > \
          (const gsl::not_null<TensorMetafunctions::prepend_spatial_index<     \
                                   GET_TENSOR(data) < GET_DTYPE(data),         \
                                   GET_DIM(data), GET_FRAME(data)>,            \
                               GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* >    \
               du,                                                             \
           const TensorMetafunctions::prepend_spatial_index<                   \
               GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),              \
               GET_FRAME(data)>,                                               \
           GET_DIM(data), UpLo::Lo,                                            \
           Frame::ElementLogical > &logical_partial_derivative_of_u,           \
           const InverseJacobian<DataVector, GET_DIM(data),                    \
                                 Frame::ElementLogical, GET_FRAME(data)>       \
               & inverse_jacobian);                                            \
  template void partial_derivative(                                            \
      const gsl::not_null<TensorMetafunctions::prepend_spatial_index<          \
                              GET_TENSOR(data) < GET_DTYPE(data),              \
                              GET_DIM(data), GET_FRAME(data)>,                 \
                          GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* > du,     \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,                   \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      GET_FRAME(data) >                                                        \
          partial_derivative(                                                  \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),         \
              GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,           \
              const InverseJacobian<DataVector, GET_DIM(data),                 \
                                    Frame::ElementLogical, GET_FRAME(data)>&   \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (tnsr::a, tnsr::A, tnsr::i, tnsr::I, tnsr::ab, tnsr::Ab,
                         tnsr::aB, tnsr::AB, tnsr::ij, tnsr::iJ, tnsr::Ij,
                         tnsr::IJ, tnsr::iA, tnsr::ia, tnsr::aa, tnsr::AA,
                         tnsr::ii, tnsr::II, tnsr::ijj, tnsr::Ijj, tnsr::iaa))

#undef INSTANTIATION

// Some additional mixed-dimension instantiations
template TensorMetafunctions::prepend_spatial_index<
    tnsr::aa<ComplexDataVector, 3, Frame::Inertial>, 2, UpLo::Lo,
    Frame::Inertial>
partial_derivative(const tnsr::aa<ComplexDataVector, 3, Frame::Inertial>& u,
                   const Mesh<2>& mesh,
                   const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                                         Frame::Inertial>& inverse_jacobian);

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template TensorMetafunctions::prepend_spatial_index<                         \
      Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo, Frame::ElementLogical> \
  logical_partial_derivative(const Scalar<GET_DTYPE(data)>& u,                 \
                             const Mesh<GET_DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATE_JACOBIANS(r, data)                                       \
  template TensorMetafunctions::prepend_spatial_index<                       \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                     \
      Frame::ElementLogical, GET_FRAME(data)>,                               \
      GET_DIM(data), UpLo::Lo,                                               \
      GET_FRAME(data) >                                                      \
          partial_derivative(                                                \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),       \
              Frame::ElementLogical, GET_FRAME(data) > &u,                   \
              const Mesh<GET_DIM(data)>& mesh,                               \
              const InverseJacobian<DataVector, GET_DIM(data),               \
                                    Frame::ElementLogical, GET_FRAME(data)>& \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIANS, (DataVector), (1, 2, 3),
                        (Frame::Inertial), (InverseJacobian))

#undef INSTANTIATE_JACOBIANS

#define INSTANTIATE_SCALAR(r, data)                                            \
  template void partial_derivative<GET_DTYPE(data), Symmetry<>, index_list<>>( \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const tnsr::i<GET_DTYPE(data), GET_DIM(data), Frame::ElementLogical>&    \
          logical_partial_derivative_of_u,                                     \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template void partial_derivative(                                            \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>            \
  partial_derivative(                                                          \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef INSTANTIATE_SCALAR
#undef GET_FRAME
#undef GET_DIM
#undef GET_TENSOR
