//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief A toy example to illustrate how we can compose logical operations over encrypted data
 */

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"
#include "cuda/experimental/stf.cuh"

using namespace cuda::experimental::stf;

#include <memory>

class ciphertext;

class plaintext
{
public:
  plaintext(const stackable_ctx& ctx)
      : ctx(ctx)
  {}

  plaintext(stackable_ctx& ctx, ::std::vector<char> v)
      : values(mv(v))
      , ctx(ctx)
  {
    ld = ::std::make_unique<stackable_logical_data<slice<char>>>(ctx.logical_data(values.data(), values.size()));
  }

  void set_symbol(std::string s)
  {
    ld->set_symbol(s);
    symbol = s;
  }

  std::string get_symbol() const
  {
    return symbol;
  }

  ::std::string symbol;

#if 0
  const stackable_logical_data<slice<char>>& data() const
  {
    return ld;
  }

  stackable_logical_data<slice<char>>& data()
  {
    return ld;
  }
#endif

  // This will asynchronously fill string s
  void convert_to_vector(std::vector<char>& v)
  {
    ctx.host_launch(ld->read()).set_symbol("to_vector")->*[&](auto dl) {
      v.resize(dl.size());
      for (size_t i = 0; i < dl.size(); i++)
      {
        v[i] = dl(i);
      }
    };
  }

  ciphertext encrypt() const;

  mutable ::std::unique_ptr<stackable_logical_data<slice<char>>> ld;

  template <typename... Pack>
  void push(Pack&&... pack)
  {
    ld->push(::std::forward<Pack>(pack)...);
  }

#if 0
  void pop()
  {
    ld->pop();
  }
#endif

private:
  std::vector<char> values;
  mutable stackable_ctx ctx;
};

class ciphertext
{
public:
  ciphertext() = default;

  ciphertext(const ciphertext& other)
      : ctx(other.ctx)
  {
    fprintf(stderr, "CTX copy ctor (src = %s)\n", other.symbol.c_str());
    copy_content(ctx, other, *this);
  }

  ciphertext(const stackable_ctx& ctx)
      : ctx(ctx)
  {}

  ciphertext(ciphertext&&)            = default;
  ciphertext& operator=(ciphertext&&) = default;

  static void copy_content(stackable_ctx& ctx, const ciphertext& src, ciphertext& dst)
  {
    dst.ld = ::std::make_unique<stackable_logical_data<slice<uint64_t>>>(ctx.logical_data(src.ld->shape()));
    ctx.parallel_for(src.ld->shape(), src.ld->read(), dst.ld->write()).set_symbol("copy")->*
      [] __device__(size_t i, auto src, auto dst) {
        dst(i) = src(i);
      };
  }

  void set_symbol(std::string s)
  {
    ld->set_symbol(s);
    symbol = s;
  }

  plaintext decrypt() const
  {
    plaintext p(ctx);
    p.ld = ::std::make_unique<stackable_logical_data<slice<char>>>(
      ctx.logical_data(shape_of<slice<char>>(ld->shape().size())));
    ctx.parallel_for(ld->shape(), ld->read(), p.ld->write()).set_symbol("decrypt")->*
      [] __device__(size_t i, auto dctxt, auto dptxt) {
        dptxt(i) = char((dctxt(i) >> 32));
      };
    return p;
  }

  // Copy assignment operator
  ciphertext& operator=(const ciphertext& other)
  {
    if (this != &other)
    {
      fprintf(stderr, "CTX copy assignment (src = %s)\n", other.symbol.c_str());
      ctx = other.ctx;
      copy_content(ctx, other, *this);
    }
    return *this;
  }

  ciphertext operator|(const ciphertext& other) const
  {
    ciphertext result(ctx);
    result.ld = ::std::make_unique<stackable_logical_data<slice<uint64_t>>>(ctx.logical_data(ld->shape()));

    ctx.parallel_for(ld->shape(), ld->read(), other.ld->read(), result.ld->write()).set_symbol("OR")->*
      [] __device__(size_t i, auto d_c1, auto d_c2, auto d_res) {
        d_res(i) = d_c1(i) | d_c2(i);
      };

    return result;
  }

  ciphertext operator&(const ciphertext& other) const
  {
    ciphertext result(ctx);
    result.ld = ::std::make_unique<stackable_logical_data<slice<uint64_t>>>(ctx.logical_data(ld->shape()));

    ctx.parallel_for(ld->shape(), ld->read(), other.ld->read(), result.ld->write()).set_symbol("AND")->*
      [] __device__(size_t i, auto d_c1, auto d_c2, auto d_res) {
        d_res(i) = d_c1(i) & d_c2(i);
      };

    return result;
  }

  ciphertext operator~() const
  {
    ciphertext result(ctx);
    result.ld = ::std::make_unique<stackable_logical_data<slice<uint64_t>>>(ctx.logical_data(ld->shape()));

    ctx.parallel_for(ld->shape(), ld->read(), result.ld->write()).set_symbol("NOT")->*
      [] __device__(size_t i, auto d_c, auto d_res) {
        d_res(i) = ~d_c(i);
      };

    return result;
  }

  template <typename... Pack>
  void push(Pack&&... pack)
  {
    ld->push(::std::forward<Pack>(pack)...);
  }

#if 0
  void pop()
  {
    ld->pop();
  }
#endif

  mutable ::std::unique_ptr<stackable_logical_data<slice<uint64_t>>> ld;

private:
  mutable stackable_ctx ctx;
  ::std::string symbol;
};

ciphertext plaintext::encrypt() const
{
  ciphertext c(ctx);
  c.ld = ::std::make_unique<stackable_logical_data<slice<uint64_t>>>(
    ctx.logical_data(shape_of<slice<uint64_t>>(ld->shape().size())));

  ctx.parallel_for(ld->shape(), ld->read(), c.ld->write()).set_symbol("encrypt")->*
    [] __device__(size_t i, auto dptxt, auto dctxt) {
      // A super safe encryption !
      dctxt(i) = ((uint64_t) (dptxt(i)) << 32 | 0x4);
    };

  return c;
}

template <typename T>
T circuit(const T& a, const T& b)
{
  return (~((a | ~b) & (~a | b)));
}

int main()
{
  stackable_ctx ctx;

  std::vector<char> vA{3, 3, 2, 2, 17};
  plaintext pA(ctx, vA);
  pA.set_symbol("A");

  std::vector<char> vB{1, 7, 7, 7, 49};
  plaintext pB(ctx, vB);
  pB.set_symbol("B");

  auto eA = pA.encrypt();
  auto eB = pB.encrypt();

  eA.set_symbol("A");
  eB.set_symbol("B");

  ctx.push();

  eA.push(access_mode::read);
  eB.push(access_mode::read);

  auto out = circuit(eA, eB);

  ctx.pop();

  std::vector<char> v_out;
  out.decrypt().convert_to_vector(v_out);

  ctx.finalize();

  for (size_t i = 0; i < v_out.size(); i++)
  {
    char expected = circuit(vA[i], vB[i]);
    EXPECT(expected == v_out[i]);
  }
}
