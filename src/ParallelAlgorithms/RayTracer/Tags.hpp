// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace ray_tracing::Tags {

template <typename DataType, size_t Dim, typename Frame>
struct Position : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct Momentum : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType>
struct Redshift : db::SimpleTag {
  using type = Scalar<DataType>;
};

struct IntegrationTime : db::SimpleTag {
  using type = double;
};

}  // namespace ray_tracing::Tags
