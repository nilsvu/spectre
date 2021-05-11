// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "ckmulticast.h"

#include "Parallel/Algorithms/AlgorithmArrayDeclarations.hpp"

template <typename... Args>
struct ArrayMessage : public CkMcastBaseMsg,
                      public CMessage_ArrayMessage<Args...> {
  std::tuple<Args...> data;
  explicit ArrayMessage(Args&&... args) : data(std::forward<Args>(args)...) {}
  void pup(PUP::er& p) {
    CMessage_ArrayMessage<Args...>::pup(p);
    p | data;
  }
};
