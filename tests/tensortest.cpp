#define BOOST_TEST_MODULE test module name
#include <boost/test/included/unit_test.hpp>

#include "tezma.h"
#include "tensor.h"

BOOST_AUTO_TEST_CASE(broadcastable1)
{
  tz::Tensor<float> t1 = tz::empty<float>({5, 7, 3});
  BOOST_TEST(tz::broadcastable(t1, t1, std::vector<size_t>()));
}

BOOST_AUTO_TEST_CASE(broadcastable2)
{
  tz::Tensor<float> t1 = tz::empty<float>({5, 3, 4, 1});
  tz::Tensor<float> t2 = tz::empty<float>({3, 1, 1});
  BOOST_TEST(tz::broadcastable(t1, t2, std::vector<size_t>()));
}

BOOST_AUTO_TEST_CASE(broadcastable3)
{
  tz::Tensor<float> t1 = tz::empty<float>({5, 2, 4, 1});
  tz::Tensor<float> t2 = tz::empty<float>({3, 1, 1});
  BOOST_TEST(!tz::broadcastable(t1, t2, std::vector<size_t>()));
}