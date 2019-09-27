// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmGroup.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {

/// \brief ParallelComponent responsible for collecting data from
/// `Element`s and interpolating it onto `InterpolationTarget`s.
///
/// For requirements on Metavariables, see InterpolationTarget
template <class Metavariables>
struct Interpolator {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<Actions::InitializeInterpolator,
                 Parallel::Actions::TerminatePhase>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<Interpolator>(local_cache)
        .start_phase(next_phase);
  };
};

namespace detail {
template <class InterpolationTargetTag, class = cpp17::void_t<>>
struct get_vars_to_interpolate {
  using type = tmpl::list<>;
};

template <class InterpolationTargetTag>
struct get_vars_to_interpolate<
    InterpolationTargetTag,
    cpp17::void_t<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>> {
  using type = typename InterpolationTargetTag::vars_to_interpolate_to_target;
};
}  // namespace detail

template <typename InterpolationTargetTags>
using collect_interpolator_source_vars =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        InterpolationTargetTags, detail::get_vars_to_interpolate<tmpl::_1>>>>;

namespace detail {
template <class InterpolationTargetTag, class = cpp17::void_t<>>
struct get_broadcast_tags {
  using type = tmpl::list<>;
};

template <class InterpolationTargetTag>
struct get_broadcast_tags<
    InterpolationTargetTag,
    cpp17::void_t<
        typename InterpolationTargetTag::broadcast_tags>> {
  using type = typename InterpolationTargetTag::broadcast_tags;
};
}  // namespace detail

template <typename InterpolationTargetTags>
using collect_broadcast_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        InterpolationTargetTags, detail::get_broadcast_tags<tmpl::_1>>>>;
}  // namespace intrp
