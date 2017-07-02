#include <vector>

#include "caffe/layers/fq_conv_layer.hpp"
#include "caffe/util/quant_functions.hpp"

namespace caffe {

template <typename Dtype>
void FQConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  Dtype* quant = this->quant_.mutable_gpu_data();
  Dtype* quant_data = this->blobs_[this->blobs_.size()-1]->mutable_cpu_data();
  const int num_level = static_cast<int>(quant_data[0]);
  Dtype& diff = quant_data[1];
  const int min_level = static_cast<int>(quant_data[2]);
  const int max_level = static_cast<int>(quant_data[3]);

  QuantOpt<Dtype> opt = QuantOpt<Dtype>(diff, min_level, max_level);
  
  if(this->phase_ == TRAIN || diff == 0 || min_level == max_level)
  {
    opt = find_quant_opt<Dtype>(weight, this->blobs_[0]->count(), num_level);
    diff = max(diff, opt.diff);
    quant_data[2] = static_cast<Dtype>(opt.min_level);
    quant_data[3] = static_cast<Dtype>(opt.max_level);
  }  

  quantize<Dtype>(weight, quant, this->blobs_[0]->count(), opt);
  CUDA_POST_KERNEL_CHECK;

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, quant,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void FQConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* quant = this->quant_.gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, quant,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FQConvolutionLayer);

}  // namespace caffe
