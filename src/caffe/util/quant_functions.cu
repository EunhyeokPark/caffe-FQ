#include <cmath>
#include <cfloat>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "caffe/common.hpp"
#include "caffe/util/quant_functions.hpp"

namespace caffe
{
  
// find level difference for applying quantization
template <typename Dtype>
QuantOpt<Dtype> find_quant_opt(const Dtype* input, const int N, const int num_level)
{
  thrust::device_ptr<const Dtype> t_ptr = thrust::device_pointer_cast<const Dtype>(input);
  thrust::pair<thrust::device_ptr<const Dtype>, thrust::device_ptr<const Dtype> > rtn = 
    thrust::minmax_element(t_ptr, t_ptr+N);

  const Dtype min_val = *(rtn.first);
  const Dtype max_val = *(rtn.second);

  // positive only quantization, e.g. after ReLU
  if(min_val > -1 * FLT_EPSILON){
    Dtype diff =  powf(2., ceil( log2(max_val/(num_level-1)) ));
    return QuantOpt<Dtype>(diff, 0, num_level - 1);
  }
  else{// positive-negative all range quantization
    // if num_level is odd, then both sides are symmetry
    // otherwise, negative side has one more level
    int pos_level = (num_level-1) / 2;
    int neg_level = -1 * num_level / 2;

    Dtype diff = max(min_val / neg_level, max_val / pos_level);
    diff = powf(2., ceil(log2(diff)));
    return QuantOpt<Dtype>(diff, neg_level, pos_level);
  }
  // other cases???
}

template
QuantOpt<float> find_quant_opt(const float* input, const int N, const int num_level);

template 
QuantOpt<double> find_quant_opt(const double* input, const int N, const int num_level);

template <typename Dtype>
__global__ void quantize_kernel(const int n, 
  const Dtype* input, Dtype* output, const QuantOpt<Dtype> opt)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    Dtype rounded = round(input[i] / opt.diff);
    rounded = (rounded > opt.max_level) ? opt.max_level :
      (rounded < opt.min_level ? opt.min_level : rounded);
    output[i] = rounded * opt.diff;
  }
}

// apply quantization for input data with N elements.
template <typename Dtype>
void quantize(const Dtype* input, Dtype* output, const int N, const QuantOpt<Dtype>& opt)
{
  quantize_kernel<Dtype><<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >>>
    (N, input, output, opt);
}

template
void quantize(const float* input, float* output, const int N, const QuantOpt<float>& opt);

template
void quantize(const double* input, double* output, const int N, const QuantOpt<double>& opt);

} // namespace caffe