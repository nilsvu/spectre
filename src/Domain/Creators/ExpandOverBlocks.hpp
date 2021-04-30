// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/algorithm/string/join.hpp>
#include <cstddef>
#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace domain {

namespace ExpandOverBlocks_detail {
template <typename T, typename U, typename = std::void_t<>>
struct is_value_type {
  static constexpr bool value = false;
};

template <typename T, typename U>
struct is_value_type<T, U, std::void_t<typename T::value_type>> {
  static constexpr bool value = std::is_same_v<U, typename T::value_type>;
};
}  // namespace ExpandOverBlocks_detail

/*!
 * \brief Produce a distribution of type `T` over all blocks and dimensions in
 * the domain, based on values `T` of variable isotropy and homogeneity.
 *
 * This class is useful to option-create values for e.g. the initial refinement
 * level or initial number of grid points for domain creators. It can be used
 * with `std::visit` and a `std::variant` with (a subset of) these types:
 *
 * - `T`: Repeat over all blocks and dimensions (isotropic and homogeneous).
 * - `std::array<T, Dim>`: Repeat over all blocks (homogeneous).
 * - `std::vector<std::array<T, Dim>>>`: Only check if the size matches the
 *   number of blocks, throwing a `std::length_error` if it doesn't.
 * - `std::unordered_map<std::string, std::array<T, Dim>>`: Map block names, or
 *   names of block groups, to values. The map must cover all blocks once the
 *   groups are expanded. To use this option you must pass the list of block
 *   names and groups to the constructor.
 *
 * Note that the call-operators `throw` when they encounter errors, such as
 * mismatches in the number of blocks. The exceptions can be used to output
 * user-facing error messages in an option-parsing context.
 *
 * Here's an example for using this class:
 *
 * \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_example
 *
 * Here's an example using block names and groups:
 *
 * \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_named_example
 *
 * \tparam T The type distributed over the domain
 */
template <typename T>
struct ExpandOverBlocks {
  ExpandOverBlocks(size_t num_blocks);
  ExpandOverBlocks(
      std::vector<std::string> block_names,
      std::unordered_map<std::string, std::unordered_set<std::string>>
          block_groups = {});

  /// Repeat over all blocks and dimensions (isotropic and homogeneous)
  template <
      typename U,
      Requires<ExpandOverBlocks_detail::is_value_type<T, U>::value> = nullptr>
  std::vector<T> operator()(const U& value) const {
    return {num_blocks_, make_array<std::tuple_size_v<T>>(value)};
  }

  /// Repeat over all blocks (homogeneous)
  std::vector<T> operator()(const T& value) const;

  /// Only check if the size matches the number of blocks, throwing a
  /// `std::length_error` if it doesn't
  std::vector<T> operator()(const std::vector<T>& value) const;

  /// Map block names, or names of block groups, to values. The map must cover
  /// all blocks once the groups are expanded. To use this option you must pass
  /// the list of block names and groups to the constructor. Here's an example:
  ///
  /// \snippet Test_ExpandOverBlocks.cpp expand_over_blocks_named_example
  std::vector<T> operator()(
      const std::unordered_map<std::string, T>& value) const;

 private:
  size_t num_blocks_;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
};

template <typename T>
ExpandOverBlocks<T>::ExpandOverBlocks(size_t num_blocks)
    : num_blocks_(num_blocks) {}

template <typename T>
ExpandOverBlocks<T>::ExpandOverBlocks(
    std::vector<std::string> block_names,
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups)
    : num_blocks_(block_names.size()),
      block_names_(std::move(block_names)),
      block_groups_(std::move(block_groups)) {}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(const T& value) const {
  if constexpr (tt::is_a_v<std::unique_ptr, T>) {
    std::vector<T> expanded{};
    expanded.reserve(num_blocks_);
    for (size_t i = 0; i < num_blocks_; ++i) {
      expanded.push_back(value->get_clone());
    }
    return expanded;
  } else {
    return {num_blocks_, value};
  }
}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(
    const std::vector<T>& value) const {
  if (value.size() != num_blocks_) {
    throw std::length_error{"You supplied " + std::to_string(value.size()) +
                            " values, but the domain creator has " +
                            std::to_string(num_blocks_) + " blocks."};
  }
  if constexpr (tt::is_a_v<std::unique_ptr, T>) {
    std::vector<T> expanded{};
    expanded.reserve(num_blocks_);
    for (const auto& v : value) {
      expanded.push_back(v->get_clone());
    }
    return expanded;
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ExpandOverBlocks<T>::operator()(
    const std::unordered_map<std::string, T>& value) const {
  ASSERT(num_blocks_ == block_names_.size(),
         "Construct 'ExpandOverBlocks' with block names to use the "
         "map-over-block-names feature.");
  // Expand group names
  auto value_per_block = [&value]() {
    if constexpr (tt::is_a_v<std::unique_ptr, T>) {
      std::unordered_map<std::string, T> copy{};
      for (const auto& [k, v] : value) {
        copy.emplace(k, v->get_clone());
      }
      return copy;
    } else {
      return value;
    }
  }();
  for (const auto& [name, block_value] : value) {
    const auto found_group = block_groups_.find(name);
    if (found_group != block_groups_.end()) {
      for (const auto& expanded_name : found_group->second) {
        if (value_per_block.count(expanded_name) == 0) {
          value_per_block[expanded_name] = [&local_block_value =
                                                block_value]() {
            if constexpr (tt::is_a_v<std::unique_ptr, T>) {
              return local_block_value->get_clone();
            } else {
              return local_block_value;
            }
          }();
        } else {
          throw std::invalid_argument{
              "Duplicate block name '" + expanded_name +
              // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
              "' (expanded from '" + name + "')."};
        }
      }
      value_per_block.erase(name);
    }
  }
  if (value_per_block.size() != num_blocks_) {
    throw std::length_error{
        "You supplied " + std::to_string(value_per_block.size()) +
        " values, but the domain creator has " + std::to_string(num_blocks_) +
        " blocks: " + boost::algorithm::join(block_names_, ", ")};
  }
  std::vector<T> result{};
  result.reserve(num_blocks_);
  for (const auto& block_name : block_names_) {
    const auto found_value = value_per_block.find(block_name);
    if (found_value != value_per_block.end()) {
      result.emplace_back(std::move(found_value->second));
    } else {
      throw std::out_of_range{"Value for block '" + block_name +
                              "' is missing. Did you misspell its name?"};
    }
  }
  return result;
}

}  // namespace domain
