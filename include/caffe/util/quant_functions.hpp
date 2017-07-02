#ifndef CAFFE_QUANT_FUNC_HPP_
#define CAFFE_QUANT_FUNC_HPP_

// fixed point quantization functions
namespace caffe
{

template <typename Dtype>
struct QuantOpt
{
  QuantOpt(Dtype diff, int min_level, int max_level):
    diff(diff), min_level(min_level), max_level(max_level){}

  Dtype diff;
  int min_level;
  int max_level;
};

// find level difference for applying quantization
template <typename Dtype>
QuantOpt<Dtype> find_quant_opt(const Dtype* input, const int N, const int num_level);

// apply quantization for input data with N elements.
template <typename Dtype>
void quantize(const Dtype* input, Dtype* output, const int N, const QuantOpt<Dtype>& opt);

}
#endif
