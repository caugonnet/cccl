#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/partition.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__ void
is_partitioned_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::is_partitioned(exec, first, last, pred);
}

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x) const
  {
    return ((int) x % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestIsPartitionedDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  n = ::cuda::std::max<size_t>(n, 2);

  thrust::device_vector<int> v = unittest::random_integers<int>(n);

  thrust::device_vector<bool> result(1);

  v[0] = 1;
  v[1] = 0;

  is_partitioned_kernel<<<1, 1>>>(exec, v.begin(), v.end(), is_even<int>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  thrust::partition(v.begin(), v.end(), is_even<int>());

  is_partitioned_kernel<<<1, 1>>>(exec, v.begin(), v.end(), is_even<int>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
}

void TestIsPartitionedDeviceSeq()
{
  TestIsPartitionedDevice(thrust::seq);
}
DECLARE_UNITTEST(TestIsPartitionedDeviceSeq);

void TestIsPartitionedDeviceDevice()
{
  TestIsPartitionedDevice(thrust::device);
}
DECLARE_UNITTEST(TestIsPartitionedDeviceDevice);
#endif

void TestIsPartitionedCudaStreams()
{
  thrust::device_vector<int> v(4);
  v[0] = 1;
  v[1] = 1;
  v[2] = 1;
  v[3] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);

  // empty partition
  ASSERT_EQUAL_QUIET(true,
                     thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin(), ::cuda::std::identity{}));

  // one element true partition
  ASSERT_EQUAL_QUIET(
    true, thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin() + 1, ::cuda::std::identity{}));

  // just true partition
  ASSERT_EQUAL_QUIET(
    true, thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin() + 2, ::cuda::std::identity{}));

  // both true & false partitions
  ASSERT_EQUAL_QUIET(true,
                     thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.end(), ::cuda::std::identity{}));

  // one element false partition
  ASSERT_EQUAL_QUIET(true,
                     thrust::is_partitioned(thrust::cuda::par.on(s), v.begin() + 3, v.end(), ::cuda::std::identity{}));

  v[0] = 1;
  v[1] = 0;
  v[2] = 1;
  v[3] = 1;

  // not partitioned
  ASSERT_EQUAL_QUIET(false,
                     thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.end(), ::cuda::std::identity{}));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestIsPartitionedCudaStreams);
