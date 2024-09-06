// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackHelper.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

/// Items related to spherical harmonics
namespace ylm {

/*!
 * \ingroup SpectralGroup
 *
 * \brief Defines the C++ interface to SPHEREPACK.
 *
 * \details The class `Spherepack` defines the C++ interface to the fortran
 * library SPHEREPACK used for computations on the surface of a sphere.
 *
 * Given a real-valued, scalar function \f$g(\theta, \phi)\f$, SPHEREPACK
 * expands it as:
 *
 * \f{align}
 * g(\theta, \phi)
 * &=\frac{1}{2}\sum_{l=0}^{l_{\max}}\bar P_l^0(\cos\theta) a_{l0}
 * +\sum_{l=1}^{l_{\max}}\sum_{m=1}^{\min(l, m_{\max})}\bar P_l^m(\cos\theta)\{
 *   a_{lm}\cos m\phi -b_{lm}\sin m\phi\}\label{eq:spherepack_expansion}
 * \f}
 *
 * where \f$a_{lm}\f$ and \f$b_{lm}\f$ are real-valued
 * spectral coefficient arrays used by
 * SPHEREPACK, \f$P_l^m(x)\f$ are defined as
 *
 * \f{align}
 * \bar P_l^m(x)&=\sqrt{\frac{(2l+1)(l-m)!}{2(l+m)!}}\;P_{lm}(x)
 * \f}
 *
 * and \f$P_{nm}(x)\f$ are the associated Legendre polynomials as defined,
 * for example, in Jackson's "Classical Electrodynamics".
 *
 * #### Relationship to standard spherical harmonics
 *
 * The standard expansion of \f$g(\theta, \phi)\f$ in terms of scalar
 * spherical harmonics is
 * \f{align}
 * g(\theta, \phi)
 * &=
 * \sum_{l=0}^{l_{\max}}\sum_{m=-\min(l, m_{\max})}^{\min(l, m_{\max})}
 * A_{lm} Y_{lm}(\theta,\phi),
 * \f}
 * where \f$Y_{lm}(\theta,\phi)\f$ are the usual complex-valued scalar
 * spherical harmonics (as defined, for example, in
 * Jackson's "Classical Electrodynamics")
 * and \f$A_{lm}\f$ are complex coefficients.
 *
 * The relationship between the complex coefficients \f$A_{lm}\f$ and
 * SPHEREPACK's real-valued \f$a_{lm}\f$ and \f$b_{lm}\f$ is
 * \f{align}
 * a_{l0} & = \sqrt{\frac{2}{\pi}}A_{l0}&\qquad l\geq 0,\\
 * a_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Re}(A_{lm})
 * &\qquad l\geq 1, m\geq 1, \\
 * b_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Im}(A_{lm})
 * &\qquad l\geq 1, m\geq 1.
 * \f}
 *
 * \note If \f$g\f$ is real,
 * \f$A_{lm} = (-1)^m A^\star_{l -m}\f$ (where \f${}^\star\f$ means
 * a complex conjugate); this is why we don't need to consider \f$m<0\f$
 * in the previous formulas or in SPHEREPACK's expansion.
 *
 * #### Relationship to real-valued spherical harmonics
 *
 * Sometimes it is useful to expand a real-valued function in the form
 * \f{align}
 * g(\theta, \phi)
 * &= \sum_{l=0}^\infty\sum_{m=0}^l
 * \left[
 * c_{lm}\mathrm{Re}(Y_{lm}(\theta, \phi))+
 * d_{nm}\mathrm{Im}(Y_{lm}(\theta, \phi))
 * \right].
 * \f}
 * The coefficients here are therefore
 * \f{align}
 * c_{l0} &= A_{l0},\\
 * c_{lm} &= 2\mathrm{Re}(A_{lm}) \qquad m\geq 1,\\
 * d_{lm} &=-2\mathrm{Im}(A_{lm}).
 * \f}
 *
 * #### Modal and nodal representations
 *
 * Internally, SPHEREPACK can represent its expansion in two ways which we
 * will refer to as modal and nodal representations:
 *
 * -# modal: The spectral coefficient arrays \f$a_{lm}\f$ and \f$b_{lm}\f$,
 * referred to as `spectral_coefs` in the methods below. For this C++ interface,
 * they are saved in a single `DataVector`. To help you index the coefficients
 * as expected by this interface, use the class `SpherepackIterator`.
 *
 * -# nodal: The values at certain collocation points, referred to as
 * `collocation_values` in the methods below. This is an array of the expanded
 * function \f$g(\theta,\phi)\f$ evaluated at collocation values
 * \f$(\theta_i,\phi_j)\f$, where \f$\theta_i\f$ are Gauss-Legendre quadrature
 * nodes in the interval \f$(0, \pi)\f$ with \f$i = 0, ..., l_{\max}\f$, and
 * \f$\phi_j\f$ is distributed uniformly in \f$(0, 2\pi)\f$ with \f$i = 0, ...,
 * 2m_{\max}\f$. The angles of the collocation points can be computed with the
 * method `theta_phi_points`.
 *
 * To convert between the two representations the methods `spec_to_phys` and
 * `phys_to_spec` can be used. For internal calculations SPHEREPACK will usually
 * convert to spectral coefficients first, so it is in general more efficient to
 * use these directly.
 *
 * Most methods of SPHEREPACK will compute the requested values of e.g.
 * `gradient` or `scalar_laplacian` at the collocation points, effectively
 * returning an expansion in nodal form as defined above. To evaluate the
 * function at arbitrary angles \f$\theta\f$, \f$\phi\f$, these values have to
 * be "interpolated" (i.e. the new expansion evaluated) using `interpolate`.
 *
 * Spherepack stores two types of quantities:
 *   1. storage_, which is filled in the constructor and is always const.
 *   2. memory_pool_, which is dynamic and thread_local, and is overwritten
 *      by various member functions that need temporary storage.
 */
class Spherepack {
 public:
  /// Type returned by gradient function.
  using FirstDeriv = tnsr::i<DataVector, 2, Frame::ElementLogical>;
  /// Type returned by second derivative function.
  using SecondDeriv = tnsr::ij<DataVector, 2, Frame::ElementLogical>;

  /// Struct to hold cached information at a set of target interpolation
  /// points.
  template <typename T>
  struct InterpolationInfo {
    InterpolationInfo(size_t l_max, size_t m_max, const gsl::span<double> pmm,
                      const std::array<T, 2>& target_points);
    T cos_theta;
    // cos(m*phi)
    DynamicBuffer<T> cos_m_phi;
    // sin(m*phi)
    DynamicBuffer<T> sin_m_phi;
    // pbar_factor[m] = Pbar(m,m)*sin(theta)^m
    DynamicBuffer<T> pbar_factor;

    size_t size() const { return num_points_; }
    size_t m_max() const { return m_max_; }
    size_t l_max() const { return l_max_; }

   private:
    size_t l_max_;
    size_t m_max_;
    size_t num_points_;
  };

  /// Here l_max and m_max are the largest fully-represented l and m in
  /// the Ylm expansion.
  Spherepack(size_t l_max, size_t m_max);

  /// @{
  /// Static functions to return the correct sizes of vectors of
  /// collocation points and spectral coefficients for a given l_max
  /// and m_max.  Useful for allocating space without having to create
  /// a Spherepack.
  SPECTRE_ALWAYS_INLINE static constexpr size_t physical_size(
      const size_t l_max, const size_t m_max) {
    return (l_max + 1) * (2 * m_max + 1);
  }
  /// \note `spectral_size` is the size of the buffer that holds the
  /// coefficients; it is not the number of coefficients (which is
  /// \f$m_{\rm max}^2+(l_{\rm max}-m_{\rm max})(2m_{\rm max}+1)\f$).
  /// To simplify its internal indexing, SPHEREPACK uses a buffer with
  /// more space than necessary. See SpherepackIterator for
  /// how to index the coefficients in the buffer.
  SPECTRE_ALWAYS_INLINE static constexpr size_t spectral_size(
      const size_t l_max, const size_t m_max) {
    return 2 * (l_max + 1) * (m_max + 1);
  }
  /// @}

  /// @{
  /// Sizes in physical and spectral space for this instance.
  size_t l_max() const { return l_max_; }
  size_t m_max() const { return m_max_; }
  size_t physical_size() const { return n_theta_ * n_phi_; }
  size_t spectral_size() const { return spectral_size_; }
  /// @}

  std::array<size_t, 2> physical_extents() const {
    return {{n_theta_, n_phi_}};
  }

  /// @{
  /// Collocation points theta and phi.
  ///
  /// The phi points are uniform in phi, with the first point
  /// at phi=0.
  ///
  /// The theta points are Gauss-Legendre in \f$\cos(\theta)\f$,
  /// so there are no points at the poles.
  SPECTRE_ALWAYS_INLINE const std::vector<double>& theta_points() const {
    return storage_.theta;
  }
  SPECTRE_ALWAYS_INLINE const std::vector<double>& phi_points() const {
    return storage_.phi;
  }
  std::array<DataVector, 2> theta_phi_points() const;
  /// @}

  /// @{
  /// Spectral transformations.
  /// To act on a slice of the input and output arrays, specify strides
  /// and offsets.
  void phys_to_spec(gsl::not_null<double*> spectral_coefs,
                    gsl::not_null<const double*> collocation_values,
                    size_t physical_stride = 1, size_t physical_offset = 0,
                    size_t spectral_stride = 1,
                    size_t spectral_offset = 0) const {
    phys_to_spec_impl(spectral_coefs, collocation_values, physical_stride,
                      physical_offset, spectral_stride, spectral_offset, false);
  }
  void spec_to_phys(gsl::not_null<double*> collocation_values,
                    gsl::not_null<const double*> spectral_coefs,
                    size_t spectral_stride = 1, size_t spectral_offset = 0,
                    size_t physical_stride = 1,
                    size_t physical_offset = 0) const {
    spec_to_phys_impl(collocation_values, spectral_coefs, spectral_stride,
                      spectral_offset, physical_stride, physical_offset, false);
  };
  /// @}

  /// @{
  /// Spectral transformations where `collocation_values` and
  /// `spectral_coefs` are assumed to point to 3-dimensional arrays
  /// (I1 x S2 topology), and the transformations are done for all
  /// 'radial' points at once by internally looping over all values of
  /// the offset from zero to `stride`-1 (the physical and spectral
  /// strides are equal and are called `stride`).
  void phys_to_spec_all_offsets(gsl::not_null<double*> spectral_coefs,
                                gsl::not_null<const double*> collocation_values,
                                size_t stride) const {
    phys_to_spec_impl(spectral_coefs, collocation_values, stride, 0, stride, 0,
                      true);
  }
  void spec_to_phys_all_offsets(gsl::not_null<double*> collocation_values,
                                gsl::not_null<const double*> spectral_coefs,
                                size_t stride) const {
    spec_to_phys_impl(collocation_values, spectral_coefs, stride, 0, stride, 0,
                      true);
  };
  /// @}

  /// @{
  /// Simpler, less general interfaces to `phys_to_spec` and `spec_to_phys`.
  /// Acts on a slice of the input and returns a unit-stride result.
  DataVector phys_to_spec(const DataVector& collocation_values,
                          size_t physical_stride = 1,
                          size_t physical_offset = 0) const;
  DataVector spec_to_phys(const DataVector& spectral_coefs,
                          size_t spectral_stride = 1,
                          size_t spectral_offset = 0) const;
  /// @}

  /// @{
  /// Simpler, less general interfaces to `phys_to_spec_all_offsets`
  /// and `spec_to_phys_all_offsets`.  Result has the same stride as
  /// the input.
  DataVector phys_to_spec_all_offsets(const DataVector& collocation_values,
                                      size_t stride) const;
  DataVector spec_to_phys_all_offsets(const DataVector& spectral_coefs,
                                      size_t stride) const;
  /// @}

  /// Computes Pfaffian derivative (df/dtheta, csc(theta) df/dphi) at
  /// the collocation values.
  /// To act on a slice of the input and output arrays, specify stride
  /// and offset (assumed to be the same for input and output).
  void gradient(const std::array<double*, 2>& df,
                gsl::not_null<const double*> collocation_values,
                size_t physical_stride = 1, size_t physical_offset = 0) const;

  /// Same as `gradient`, but takes the spectral coefficients (rather
  /// than collocation values) of the function.  This is more
  /// efficient if one happens to already have the spectral
  /// coefficients.
  /// To act on a slice of the input and output arrays, specify strides
  /// and offsets.
  void gradient_from_coefs(const std::array<double*, 2>& df,
                           gsl::not_null<const double*> spectral_coefs,
                           size_t spectral_stride = 1,
                           size_t spectral_offset = 0,
                           size_t physical_stride = 1,
                           size_t physical_offset = 0) const {
    gradient_from_coefs_impl(df, spectral_coefs, spectral_stride,
                             spectral_offset, physical_stride, physical_offset,
                             false);
  }

  /// @{
  /// Same as `gradient` but pointers are assumed to point to
  /// 3-dimensional arrays (I1 x S2 topology), and the gradient is
  /// done for all 'radial' points at once by internally looping
  /// over all values of the offset from zero to `stride`-1.
  void gradient_all_offsets(const std::array<double*, 2>& df,
                            gsl::not_null<const double*> collocation_values,
                            size_t stride = 1) const;

  SPECTRE_ALWAYS_INLINE void gradient_from_coefs_all_offsets(
      const std::array<double*, 2>& df,
      gsl::not_null<const double*> spectral_coefs, size_t stride = 1) const {
    gradient_from_coefs_impl(df, spectral_coefs, stride, 0, stride, 0, true);
  }
  /// @}

  /// @{
  /// Simpler, less general interfaces to `gradient`.
  /// Acts on a slice of the input and returns a unit-stride result.
  FirstDeriv gradient(const DataVector& collocation_values,
                      size_t physical_stride = 1,
                      size_t physical_offset = 0) const;
  FirstDeriv gradient_from_coefs(const DataVector& spectral_coefs,
                                 size_t spectral_stride = 1,
                                 size_t spectral_offset = 0) const;
  /// @}

  /// @{
  /// Simpler, less general interfaces to `gradient_all_offsets`.
  /// Result has the same stride as the input.
  FirstDeriv gradient_all_offsets(const DataVector& collocation_values,
                                  size_t stride = 1) const;
  FirstDeriv gradient_from_coefs_all_offsets(const DataVector& spectral_coefs,
                                             size_t stride = 1) const;
  /// @}

  /// Computes Laplacian in physical space.
  /// To act on a slice of the input and output arrays, specify stride
  /// and offset (assumed to be the same for input and output).
  void scalar_laplacian(gsl::not_null<double*> scalar_laplacian,
                        gsl::not_null<const double*> collocation_values,
                        size_t physical_stride = 1,
                        size_t physical_offset = 0) const;

  /// Same as `scalar_laplacian` above, but the input is the spectral
  /// coefficients (rather than collocation values) of the function.
  /// This is more efficient if one happens to already have the
  /// spectral coefficients.
  /// To act on a slice of the input and output arrays, specify strides
  /// and offsets.
  void scalar_laplacian_from_coefs(gsl::not_null<double*> scalar_laplacian,
                                   gsl::not_null<const double*> spectral_coefs,
                                   size_t spectral_stride = 1,
                                   size_t spectral_offset = 0,
                                   size_t physical_stride = 1,
                                   size_t physical_offset = 0) const;

  /// @{
  /// Simpler, less general interfaces to `scalar_laplacian`.
  /// Acts on a slice of the input and returns a unit-stride result.
  DataVector scalar_laplacian(const DataVector& collocation_values,
                              size_t physical_stride = 1,
                              size_t physical_offset = 0) const;
  DataVector scalar_laplacian_from_coefs(const DataVector& spectral_coefs,
                                         size_t spectral_stride = 1,
                                         size_t spectral_offset = 0) const;
  /// @}

  /// Computes Pfaffian first and second derivative in physical space.
  /// The first derivative is \f$df(i) = d_i f\f$, and the
  /// second derivative is \f$ddf(i,j) = d_i (d_j f)\f$,
  /// where \f$d_0 = d/d\theta\f$ and \f$d_1 = csc(\theta) d/d\phi\f$.
  /// ddf is not symmetric.
  /// To act on a slice of the input and output arrays, specify stride
  /// and offset (assumed to be the same for input and output).
  void second_derivative(const std::array<double*, 2>& df,
                         gsl::not_null<SecondDeriv*> ddf,
                         gsl::not_null<const double*> collocation_values,
                         size_t physical_stride = 1,
                         size_t physical_offset = 0) const;

  /// Simpler, less general interface to second_derivative
  std::pair<FirstDeriv, SecondDeriv> first_and_second_derivative(
      const DataVector& collocation_values) const;

  /// Computes the integral over the sphere.
  SPECTRE_ALWAYS_INLINE double definite_integral(
      gsl::not_null<const double*> collocation_values,
      size_t physical_stride = 1, size_t physical_offset = 0) const {
    // clang-tidy: 'do not use pointer arithmetic'
    return ddot_(n_theta_ * n_phi_, storage_.quadrature_weights.data(), 1,
                 collocation_values.get() + physical_offset,  // NOLINT
                 physical_stride);
  }

  /// Returns weights \f$w_i\f$ such that \f$sum_i (c_i w_i)\f$
  /// is the definite integral, where \f$c_i\f$ are collocation values
  /// at point i.
  SPECTRE_ALWAYS_INLINE const std::vector<double>& integration_weights() const {
    return storage_.quadrature_weights;
  }

  /// Adds a constant (i.e. \f$f(\theta,\phi)\f$ += \f$c\f$) to the function
  /// given by the spectral coefficients, by modifying the coefficients.
  SPECTRE_ALWAYS_INLINE static void add_constant(
      const gsl::not_null<DataVector*> spectral_coefs, const double c) {
    // The factor of sqrt(8) is because of the normalization of
    // SPHEREPACK's coefficients.
    (*spectral_coefs)[0] += sqrt(8.0) * c;
  }

  /// Returns the average of \f$f(\theta,\phi)\f$ over \f$(\theta,\phi)\f$.
  SPECTRE_ALWAYS_INLINE static double average(
      const DataVector& spectral_coefs) {
    // The factor of sqrt(8) is because of the normalization of
    // SPHEREPACK's coefficients.  All other coefficients average to zero.
    return spectral_coefs[0] / sqrt(8.0);
  }

  /// Sets up the `InterpolationInfo` structure for interpolating onto
  /// a set of target \f$(\theta,\phi)\f$ points.  Does not depend on
  /// the function being interpolated.
  template <typename T>
  InterpolationInfo<T> set_up_interpolation_info(
      const std::array<T, 2>& target_points) const;

  /// Interpolates from `collocation_values` onto the points that have
  /// been passed into the `set_up_interpolation_info` function.
  /// To interpolate a different function on the same spectral grid, there
  /// is no need to recompute `interpolation_info`.
  /// If you specify stride and offset, acts on a slice of the input values.
  /// The output has unit stride.
  template <typename T>
  void interpolate(gsl::not_null<T*> result,
                   gsl::not_null<const double*> collocation_values,
                   const InterpolationInfo<T>& interpolation_info,
                   size_t physical_stride = 1,
                   size_t physical_offset = 0) const;

  /// Same as `interpolate`, but assumes you have spectral coefficients.
  /// This is more efficient if you already have the spectral coefficients
  /// available.
  /// If you specify stride and offset, acts on a slice of the input coefs.
  /// The output has unit stride.
  template <typename T, typename R>
  void interpolate_from_coefs(gsl::not_null<T*> result, const R& spectral_coefs,
                              const InterpolationInfo<T>& interpolation_info,
                              size_t spectral_stride = 1,
                              size_t spectral_offset = 0) const;

  /// Simpler interface to `interpolate`.  If you need to call this
  /// repeatedly on different `spectral_coefs` or `collocation_values`
  /// for the same target points, this is inefficient; instead use
  /// `set_up_interpolation_info` and the functions that use
  /// `InterpolationInfo`.
  template <typename T>
  T interpolate(const DataVector& collocation_values,
                const std::array<T, 2>& target_points) const;
  template <typename T>
  T interpolate_from_coefs(const DataVector& spectral_coefs,
                           const std::array<T, 2>& target_points) const;

  /// Takes spectral coefficients compatible with `*this`, and either
  /// prolongs them or restricts them to be compatible with `target`.
  /// This is done by truncation (restriction) or padding with zeros
  /// (prolongation).
  DataVector prolong_or_restrict(const DataVector& spectral_coefs,
                                 const Spherepack& target) const;

 private:
  // Spectral transformations and gradient.
  // If `loop_over_offset` is true, then `collocation_values` and
  // `spectral_coefs` are assumed to point to 3-dimensional
  // arrays (I1 x S2 topology), and the transformations are done for
  // all 'radial' points at once by looping over all values of the
  // offset from zero to stride-1.  If `loop_over_offset` is true,
  // `physical_stride` must equal `spectral_stride`.
  void phys_to_spec_impl(gsl::not_null<double*> spectral_coefs,
                         gsl::not_null<const double*> collocation_values,
                         size_t physical_stride = 1, size_t physical_offset = 0,
                         size_t spectral_stride = 1, size_t spectral_offset = 0,
                         bool loop_over_offset = false) const;
  void spec_to_phys_impl(gsl::not_null<double*> collocation_values,
                         gsl::not_null<const double*> spectral_coefs,
                         size_t spectral_stride = 1, size_t spectral_offset = 0,
                         size_t physical_stride = 1, size_t physical_offset = 0,
                         bool loop_over_offset = false) const;
  void gradient_from_coefs_impl(const std::array<double*, 2>& df,
                                gsl::not_null<const double*> spectral_coefs,
                                size_t spectral_stride = 1,
                                size_t spectral_offset = 0,
                                size_t physical_stride = 1,
                                size_t physical_offset = 0,
                                bool loop_over_offset = false) const;
  void calculate_collocation_points();
  void calculate_interpolation_data();
  void fill_scalar_work_arrays();
  void fill_vector_work_arrays();
  size_t l_max_, m_max_, n_theta_, n_phi_;
  size_t spectral_size_;
  // memory_pool_ will be shared by multiple instances of
  // Spherepack on the same thread.  Because these instances are on
  // the same thread, member functions of two or more of these
  // instances cannot be called simultaneously.  Note that member
  // functions do not make any assumptions about the contents of
  // memory_pool_ on entry, so between calls to member functions it is
  // safe to resize objects in memory_pool_ or to overwrite them with
  // arbitrary data.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Spherepack_detail::MemoryPool memory_pool_{};
  Spherepack_detail::ConstStorage storage_;
};  // class Spherepack

bool operator==(const Spherepack& lhs, const Spherepack& rhs);
bool operator!=(const Spherepack& lhs, const Spherepack& rhs);

}  // namespace ylm
