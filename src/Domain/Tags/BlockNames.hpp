// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/OptionTags.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {

template <size_t Dim>
struct BlockNames : db::SimpleTag {
  using type = std::vector<std::string>;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept {
    return domain_creator->block_names();
  }
};

}  // namespace domain::Tags
