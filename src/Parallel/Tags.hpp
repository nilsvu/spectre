// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"

namespace Parallel::Tags {

template <typename SectionIdTag>
struct SectionBase : db::BaseTag {
  using section_id_tag = SectionIdTag;
};

template <typename SectionIdTag, typename ParallelComponent>
struct Section : db::SimpleTag, SectionBase<SectionIdTag> {
  using section_id_tag = SectionIdTag;
  using chare_type = typename ParallelComponent::chare_type;
  using charm_type = Parallel::charm_types_with_parameters<
      ParallelComponent, typename Parallel::get_array_index<
                             chare_type>::template f<ParallelComponent>>;
  using type = typename charm_type::cproxy_section;
};
}  // namespace Parallel::Tags
