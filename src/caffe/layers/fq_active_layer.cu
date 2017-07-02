#include <vector>
#include <cmath>
#include "caffe/layers/fq_active_layer.hpp"
#include "caffe/util/quant_functions.hpp"

namespace caffe {

template <typename Dtype>
void FQActiveLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  Dtype* quant_data = this->blobs_[0]->mutable_cpu_data();
  const int num_level = static_cast<int>(quant_data[0]);
  Dtype& diff = quant_data[1];
  const int min_level = static_cast<int>(quant_data[2]);
  const int max_level = static_cast<int>(quant_data[3]);

  QuantOpt<Dtype> opt = QuantOpt<Dtype>(diff, min_level, max_level);
  
  if(this->phase_ == TRAIN || diff == 0 || min_level == max_level)
  {
    opt = find_quant_opt<Dtype>(bottom_data, count, num_level);
    diff = max(diff, opt.diff);
    quant_data[2] = static_cast<Dtype>(opt.min_level);
    quant_data[3] = static_cast<Dtype>(opt.max_level);
  }  

  quantize<Dtype>(bottom_data, top_data, count, opt);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void FQActiveLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    caffe_copy<Dtype>(count, top_diff, bottom_diff);    
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FQActiveLayer);

}  // namespace caffe
