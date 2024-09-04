// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace dg::detail {
namespace {
// We use a separate function in the  xi direction to avoid the expensive
// SliceIterator
template <typename ValueType>
void lift_boundary_terms_gauss_points_impl_xi_dir(
    const gsl::not_null<ValueType*> volume_dt_vars,
    const size_t num_independent_components, const size_t num_volume_pts,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const ValueType>& boundary_corrections,
    const DataVector& boundary_lifting_term,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian) {
  using VectorType =
      tmpl::conditional_t<std::is_same_v<ValueType, std::complex<double>>,
                          ComplexDataVector, DataVector>;
  VectorType volume_dt_vars_view{};
  DataVector volume_inv_det_jacobian_view{};
  for (size_t component_index = 0; component_index < num_independent_components;
       ++component_index) {
    const size_t stripe_size = num_volume_pts / num_boundary_pts;
    for (size_t boundary_index = 0; boundary_index < num_boundary_pts;
         ++boundary_index) {
      volume_dt_vars_view.set_data_ref(volume_dt_vars.get() +
                                           component_index * num_volume_pts +
                                           boundary_index * stripe_size,
                                       stripe_size);
      // safe const_cast since used as a view
      volume_inv_det_jacobian_view.set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(get(volume_det_inv_jacobian).data()) +
              boundary_index * stripe_size,
          stripe_size);
      // Minus sign because we brought this from the LHS to the RHS.
      volume_dt_vars_view -=
          volume_inv_det_jacobian_view * boundary_lifting_term *
          get(face_det_jacobian)[boundary_index] *
          get(magnitude_of_face_normal)[boundary_index] *
          boundary_corrections[component_index * num_boundary_pts +
                               boundary_index];
    }
  }
}

// We use a separate function in the  xi direction to avoid the expensive
// SliceIterator
template <typename ValueType>
void lift_boundary_terms_gauss_points_impl_xi_dir(
    const gsl::not_null<ValueType*> volume_dt_vars,
    const size_t num_independent_components, const size_t num_volume_pts,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const ValueType>& upper_boundary_corrections,
    const DataVector& upper_boundary_lifting_term,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const gsl::span<const ValueType>& lower_boundary_corrections,
    const DataVector& lower_boundary_lifting_term,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian) {
  using VectorType =
      tmpl::conditional_t<std::is_same_v<ValueType, std::complex<double>>,
                          ComplexDataVector, DataVector>;
  VectorType volume_dt_vars_view{};
  DataVector volume_inv_det_jacobian_view{};
  for (size_t component_index = 0; component_index < num_independent_components;
       ++component_index) {
    const size_t stripe_size = num_volume_pts / num_boundary_pts;
    for (size_t boundary_index = 0; boundary_index < num_boundary_pts;
         ++boundary_index) {
      volume_dt_vars_view.set_data_ref(volume_dt_vars.get() +
                                           component_index * num_volume_pts +
                                           boundary_index * stripe_size,
                                       stripe_size);
      // safe const_cast since used as a view
      volume_inv_det_jacobian_view.set_data_ref(
          // NOLINTNEXTLINE
          const_cast<double*>(get(volume_det_inv_jacobian).data()) +
              boundary_index * stripe_size,
          stripe_size);
      // Minus sign because we brought this from the LHS to the RHS.
      volume_dt_vars_view -=
          volume_inv_det_jacobian_view *
          (upper_boundary_lifting_term *
               get(upper_face_det_jacobian)[boundary_index] *
               get(upper_magnitude_of_face_normal)[boundary_index] *
               upper_boundary_corrections[component_index * num_boundary_pts +
                                          boundary_index] +
           lower_boundary_lifting_term *
               get(lower_face_det_jacobian)[boundary_index] *
               get(lower_magnitude_of_face_normal)[boundary_index] *
               lower_boundary_corrections[component_index * num_boundary_pts +
                                          boundary_index]);
    }
  }
}
}  // namespace

template <typename ValueType, size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    const gsl::not_null<ValueType*> volume_dt_vars,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const ValueType>& boundary_corrections,
    const DataVector& boundary_lifting_term,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian) {
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  if (dimension == 0) {
    lift_boundary_terms_gauss_points_impl_xi_dir(
        volume_dt_vars, num_independent_components, num_volume_pts,
        volume_det_inv_jacobian, num_boundary_pts, boundary_corrections,
        boundary_lifting_term, magnitude_of_face_normal, face_det_jacobian);
    return;
  }

  // Developer note: A potential optimization is to re-order (not transpose!)
  // the volume time derivative for all variables before lifting, so that the
  // lifting can be done using vectorized math and DataVectors as views. It
  // would need to be tested whether that actually increases performance.
  // Another alternative would be to use a SIMD library, such as nsimd or xsimd.

  const size_t stripe_size = volume_mesh.extents(dimension);
  size_t boundary_index = 0;
  for (StripeIterator si{volume_mesh.extents(), dimension}; si;
       (void)++si, (void)++boundary_index) {
    // Loop over each stripe in this logical direction. This is effectively
    // looping over each boundary grid point.
    for (size_t component_index = 0;
         component_index < num_independent_components; ++component_index) {
      for (size_t index_on_stripe = 0; index_on_stripe < stripe_size;
           ++index_on_stripe) {
        const size_t volume_index = si.offset() + si.stride() * index_on_stripe;
        volume_dt_vars.get()[component_index * num_volume_pts + volume_index] -=
            get(volume_det_inv_jacobian)[volume_index] *
            boundary_lifting_term[index_on_stripe] *
            get(face_det_jacobian)[boundary_index] *
            get(magnitude_of_face_normal)[boundary_index] *
            boundary_corrections[component_index * num_boundary_pts +
                                 boundary_index];
      }
    }
  }
}

template <typename ValueType, size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    const gsl::not_null<ValueType*> volume_dt_vars,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,
    const size_t num_boundary_pts,
    const gsl::span<const ValueType>& upper_boundary_corrections,
    const DataVector& upper_boundary_lifting_term,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const gsl::span<const ValueType>& lower_boundary_corrections,
    const DataVector& lower_boundary_lifting_term,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian) {
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  if (dimension == 0) {
    lift_boundary_terms_gauss_points_impl_xi_dir(
        volume_dt_vars, num_independent_components, num_volume_pts,
        volume_det_inv_jacobian, num_boundary_pts, upper_boundary_corrections,
        upper_boundary_lifting_term, upper_magnitude_of_face_normal,
        upper_face_det_jacobian, lower_boundary_corrections,
        lower_boundary_lifting_term, lower_magnitude_of_face_normal,
        lower_face_det_jacobian);
    return;
  }
  // Developer note: A potential optimization is to re-order (not transpose!)
  // the volume time derivative for all variables before lifting, so that the
  // lifting can be done using vectorized math and DataVectors as views. It
  // would need to be tested whether that actually increases performance.
  // Another alternative would be to use a SIMD library, such as nsimd or xsimd.

  const size_t stripe_size = volume_mesh.extents(dimension);
  size_t boundary_index = 0;
  for (StripeIterator si{volume_mesh.extents(), dimension}; si;
       (void)++si, (void)++boundary_index) {
    // Loop over each stripe in this logical direction. This is effectively
    // looping over each boundary grid point.
    for (size_t component_index = 0;
         component_index < num_independent_components; ++component_index) {
      for (size_t index_on_stripe = 0; index_on_stripe < stripe_size;
           ++index_on_stripe) {
        const size_t volume_index = si.offset() + si.stride() * index_on_stripe;
        volume_dt_vars.get()[component_index * num_volume_pts + volume_index] -=
            get(volume_det_inv_jacobian)[volume_index] *
            (upper_boundary_lifting_term[index_on_stripe] *
                 get(upper_face_det_jacobian)[boundary_index] *
                 get(upper_magnitude_of_face_normal)[boundary_index] *
                 upper_boundary_corrections[component_index * num_boundary_pts +
                                            boundary_index] +
             lower_boundary_lifting_term[index_on_stripe] *
                 get(lower_face_det_jacobian)[boundary_index] *
                 get(lower_magnitude_of_face_normal)[boundary_index] *
                 lower_boundary_corrections[component_index * num_boundary_pts +
                                            boundary_index]);
      }
    }
  }
}

template <typename ValueType, size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    const gsl::not_null<ValueType*> volume_data,
    const size_t num_independent_components, const Mesh<Dim>& volume_mesh,
    const Direction<Dim>& direction,
    const gsl::span<const ValueType>& boundary_corrections) {
  const size_t dimension = direction.dimension();
  const auto& lower_and_upper_lifting_matrix =
      Spectral::boundary_interpolation_matrices(
          volume_mesh.slice_through(dimension));
  // One row of values: \ell_{\breve{\imath}}(\xi=\pm1)
  const Matrix& lifting_matrix = direction.side() == Side::Lower
                                     ? lower_and_upper_lifting_matrix.first
                                     : lower_and_upper_lifting_matrix.second;
  // Loop over each stripe in this logical direction. This is effectively
  // looping over each boundary grid point.
  const size_t stripe_size = volume_mesh.extents(dimension);
  size_t boundary_index = 0;
  const size_t num_volume_pts = volume_mesh.number_of_grid_points();
  const size_t num_boundary_pts =
      volume_mesh.slice_away(dimension).number_of_grid_points();
  for (StripeIterator si{volume_mesh.extents(), dimension}; si;
       (void)++si, (void)++boundary_index) {
    for (size_t component_index = 0;
         component_index < num_independent_components; ++component_index) {
      for (size_t index_on_stripe = 0; index_on_stripe < stripe_size;
           ++index_on_stripe) {
        const size_t volume_index = si.offset() + si.stride() * index_on_stripe;
        volume_data.get()[component_index * num_volume_pts + volume_index] +=
            lifting_matrix(0, index_on_stripe) *
            boundary_corrections[component_index * num_boundary_pts +
                                 boundary_index];
      }
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(r, data)                                                 \
  template void lift_boundary_terms_gauss_points_impl(                       \
      gsl::not_null<DTYPE(data)*> volume_dt_vars,                            \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,   \
      size_t num_boundary_pts,                                               \
      const gsl::span<const DTYPE(data)>& boundary_corrections,              \
      const DataVector& boundary_lifting_term,                               \
      const Scalar<DataVector>& magnitude_of_face_normal,                    \
      const Scalar<DataVector>& face_det_jacobian);                          \
  template void lift_boundary_terms_gauss_points_impl(                       \
      gsl::not_null<DTYPE(data)*> volume_dt_vars,                            \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      size_t dimension, const Scalar<DataVector>& volume_det_inv_jacobian,   \
      size_t num_boundary_pts,                                               \
      const gsl::span<const DTYPE(data)>& upper_boundary_corrections,        \
      const DataVector& upper_boundary_lifting_term,                         \
      const Scalar<DataVector>& upper_magnitude_of_face_normal,              \
      const Scalar<DataVector>& upper_face_det_jacobian,                     \
      const gsl::span<const DTYPE(data)>& lower_boundary_corrections,        \
      const DataVector& lower_boundary_lifting_term,                         \
      const Scalar<DataVector>& lower_magnitude_of_face_normal,              \
      const Scalar<DataVector>& lower_face_det_jacobian);                    \
  template void lift_boundary_terms_gauss_points_impl(                       \
      gsl::not_null<DTYPE(data)*> volume_data,                               \
      size_t num_independent_components, const Mesh<DIM(data)>& volume_mesh, \
      const Direction<DIM(data)>& direction,                                 \
      const gsl::span<const DTYPE(data)>& boundary_corrections);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace dg::detail
