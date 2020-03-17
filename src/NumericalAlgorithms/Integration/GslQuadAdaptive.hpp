// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <tuple>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Numerical integration algorithms
namespace integrate {

namespace detail {

template <typename FunctionType>
double integrand(double x, void* params) {
  const auto function = reinterpret_cast<FunctionType*>(params);
  return (*function)(x);
}

// The GSL functions require the integrand to have this particular function
// signature. In particular, any extra parameters to the functions must be
// passed as a void* and re-interpreted appropriately. We interpret the
// pointer as an instance of GslQuadAdaptiveImpl and use it to retrieve the
// integrand function and its extra parameters.
// The class builds a numerical integrator with the maximum number of
// subintervals passed into the constructor.
class GslQuadAdaptiveImpl {
 public:
  GslQuadAdaptiveImpl(size_t max_intervals) noexcept
      : max_intervals_(max_intervals) {
    initialize();
  }

  GslQuadAdaptiveImpl() = default;
  GslQuadAdaptiveImpl(const GslQuadAdaptiveImpl&) = delete;
  GslQuadAdaptiveImpl& operator=(const GslQuadAdaptiveImpl&) = delete;
  GslQuadAdaptiveImpl(GslQuadAdaptiveImpl&&) noexcept = default;
  GslQuadAdaptiveImpl& operator=(GslQuadAdaptiveImpl&& rhs) noexcept = default;
  ~GslQuadAdaptiveImpl() noexcept = default;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept {
    p | max_intervals_;
    if (p.isUnpacking()) {
      initialize();
    }
  }

 private:
  struct gsl_integration_workspace_deleter {
    void operator()(gsl_integration_workspace* workspace) const noexcept {
      gsl_integration_workspace_free(workspace);
    }
  };

  void initialize() noexcept {
    workspace_ = std::unique_ptr<gsl_integration_workspace,
                                 gsl_integration_workspace_deleter>{
        gsl_integration_workspace_alloc(max_intervals_)};
  }

  mutable gsl_function gsl_integrand_;

 protected:
  template <typename IntegrandType>
  gsl_function* gsl_integrand(IntegrandType&& integrand) const noexcept {
    gsl_integrand_.function = &detail::integrand<IntegrandType>;
    gsl_integrand_.params = &integrand;
    return &gsl_integrand_;
  }

  size_t max_intervals_;
  std::unique_ptr<gsl_integration_workspace, gsl_integration_workspace_deleter>
      workspace_;
};
}  // namespace detail

enum class IntegralType {
  StandardGaussKronrod,            // gsl_integration_qag()
  IntegrableSingularitiesPresent,  // gsl_integration_qags()
  IntegrableSingularitiesKnown,    // gsl_integration_qagp()
  InfiniteInterval,                // gsl_integration_qagi()
  UpperBoundaryInfinite,           // gsl_integration_qagiu()
  LowerBoundaryInfinite            // gsl_integration_qagil()
};

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A wrapper around the GSL adaptive integration procedures
 *
 * All templates take upper bounds to the absolute and relative error;
 * `tolerance_abs` and `tolerance_rel(default = 0.0)`, respectively.
 * For details on the algorithm see the GSL documentation on `gsl_integration`.
 *
 * Here is an example how to use this class. For the function:
 * \snippet Test_GslQuad.cpp integrated_function
 * the integration should look like:
 * \snippet Test_GslQuad.cpp integration_example
 */
template <IntegralType TheIntegralType>
class GslQuadAdaptive;

/*!
 * The algorithm for "StandardGaussKronrod" uses the QAG algorithm to employ an
 * adaptive Gauss-Kronrod n-points integration rule. Its function takes a
 * `lower_boundary` and `upper_boundary` to the integration region, an upper
 * bound for the absolute error `tolerance_abs` and a `key`. The latter is an
 * index to the array [15, 21, 31, 41, 51, 61], where each element denotes how
 * many function evaluations take place in each subinterval. The GSL
 * documentation mentions that a higher-order rule serves better for smooth
 * functions, whereas a lower-order rule saves time for functions with local
 * difficulties, such as discontinuities.
 */
template <>
class GslQuadAdaptive<IntegralType::StandardGaussKronrod>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, double lower_boundary,
                    double upper_boundary, double tolerance_abs, int key,
                    double tolerance_rel = 0.) const noexcept {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    const auto status = gsl_integration_qag(
        gsl_integrand(std::forward<IntegrandType>(integrand)), lower_boundary,
        upper_boundary, tolerance_abs, tolerance_rel, this->max_intervals_, key,
        this->workspace_.get(), &result, &error);
    return result;
  }
};

// /*!
//  * The algorithm for "IntegrableSingularitiesPresent" concentrates new,
//  * increasingly smaller subintervals around an unknown singularity and makes
//  * successive approximations to the integral which should converge towards a
//  * limit. The integration region is defined by `lower_boundary` and
//  * `upper_boundary`.
//  */
// template <typename... Args>
// class GslQuadAdaptive<IntegralType::IntegrableSingularitiesPresent, Args...>
//     : public detail::GslQuadAdaptiveImpl<Args...> {
//  public:
//   using detail::GslQuadAdaptiveImpl<Args...>::GslQuadAdaptiveImpl;
//   Result operator()(double lower_boundary, double upper_boundary,
//                     double tolerance_abs, double tolerance_rel = 0.) const
//       noexcept {
//     Result result;
//     int status = gsl_integration_qags(
//         &(this->integrand_), lower_boundary, upper_boundary, tolerance_abs,
//         tolerance_rel, this->max_intervals_, this->workspace_.get(),
//         &result.value, &result.error);
//     return result;
//   }
// };

// /*!
//  * The algorithm for "IntegrableSingularitiesKnown" uses user-defined
//  * subintervals given by a vector of doubles `points`, where each element
//  * denotes an interval boundary.
//  */
// template <typename... Args>
// class GslQuadAdaptive<IntegralType::IntegrableSingularitiesKnown, Args...>
//     : public detail::GslQuadAdaptiveImpl<Args...> {
//  public:
//   using detail::GslQuadAdaptiveImpl<Args...>::GslQuadAdaptiveImpl;
//   Result operator()(std::vector<double> points, double tolerance_abs,
//                     double tolerance_rel = 0.) const noexcept {
//     Result result;
//     int status = gsl_integration_qagp(
//         &(this->integrand_), points.data(), points.size(), tolerance_abs,
//         tolerance_rel, this->max_intervals_, this->workspace_.get(),
//         &result.value, &result.error);
//     return result;
//   }
// };

// /*!
//  * The algorithm for "InfiniteInterval" maps the semi-open interval (0, 1] to
//  an
//  * infinite interval \f$ (-\infty, +\infty) \f$. Its function takes no
//  * parameters other than a limit `tolerance_abs` for the absolute_error.
//  */
// template <typename... Args>
// class GslQuadAdaptive<IntegralType::InfiniteInterval, Args...>
//     : public detail::GslQuadAdaptiveImpl<Args...> {
//  public:
//   using detail::GslQuadAdaptiveImpl<Args...>::GslQuadAdaptiveImpl;
//   Result operator()(double tolerance_abs, double tolerance_rel = 0.) noexcept
//   {
//     Result result;
//     int status = gsl_integration_qagi(
//         &(this->integrand_), tolerance_abs, tolerance_rel,
//         this->max_intervals_, this->workspace_.get(), &result.value,
//         &result.error);
//     return result;
//   }
// };

// /*!
//  * The algorithm for "UpperBoundaryInfinite" maps the semi-open interval (0,
//  1]
//  * to a semi-infinite interval \f$(a, +\infty)\f$, where a is given by
//  * `lower_boundary`.
//  */
// template <typename... Args>
// class GslQuadAdaptive<IntegralType::UpperBoundaryInfinite, Args...>
//     : public detail::GslQuadAdaptiveImpl<Args...> {
//  public:
//   using detail::GslQuadAdaptiveImpl<Args...>::GslQuadAdaptiveImpl;
//   Result operator()(double lower_boundary, double tolerance_abs,
//                     double tolerance_rel = 0.) noexcept {
//     Result result;
//     int status = gsl_integration_qagiu(
//         &(this->integrand_), lower_boundary, tolerance_abs, tolerance_rel,
//         this->max_intervals_, this->workspace_.get(), &result.value,
//         &result.error);
//     return result;
//   }
// };

// /*!
//  * The algorithm for "LowerBoundaryInfinite" maps the semi-open interval (0,
//  1]
//  * to a semi-infinite interval \f$(-\infty, b)\f$, where b is given by
//  * `upper_boundary`.
//  */
// template <typename... Args>
// class GslQuadAdaptive<IntegralType::LowerBoundaryInfinite, Args...>
//     : public detail::GslQuadAdaptiveImpl<Args...> {
//  public:
//   using detail::GslQuadAdaptiveImpl<Args...>::GslQuadAdaptiveImpl;
//   Result operator()(double upper_boundary, double tolerance_abs,
//                     double tolerance_rel = 0.) noexcept {
//     Result result;
//     int status = gsl_integration_qagil(
//         &(this->integrand_), upper_boundary, tolerance_abs, tolerance_rel,
//         this->max_intervals_, this->workspace_.get(), &result.value,
//         &result.error);
//     return result;
//   }
// };
}  // namespace integrate
