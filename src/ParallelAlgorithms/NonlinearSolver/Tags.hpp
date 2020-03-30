// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver {
namespace Tags {

/*
 * \brief The correction \f$\delta x\f$ to improve a solution \f$x_0\f$
 *
 * \details A linear problem \f$Ax=b\f$ can be equivalently formulated as the
 * problem \f$A\delta x=b-A x_0\f$ for the correction \f$\delta x\f$ to an
 * initial guess \f$x_0\f$. More importantly, we can use a correction scheme
 * to solve a nonlinear problem \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly
 * solving a linearization of it. For instance, a Newton-Raphson scheme
 * iteratively refines an initial guess \f$x_0\f$ by repeatedly solving the
 * linearized problem
 *
 * \f[
 * \frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k)\delta x_k =
 * b-A_\mathrm{nonlinear}(x_k)
 * \f]
 *
 * for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 */
template <typename Tag>
struct Correction : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};

}  // namespace Tags
}  // namespace NonlinearSolver
