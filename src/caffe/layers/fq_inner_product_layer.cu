#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fq_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/quant_functions.hpp"

namespace caffe {

template <typename Dtype>
void FQInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
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

  if (this->M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->N_, this->K_, (Dtype)1.,
                         quant, bottom_data, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_axpy<Dtype>(this->N_, this->bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          this->transpose_ ? CblasNoTrans : CblasTrans,
                          this->M_, this->N_, this->K_, (Dtype)1.,
                          bottom_data, quant, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1, (Dtype)1.,
                            this->bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void FQInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    // Gradient with respect to weight
    if (this->transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->K_, this->N_, this->M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->N_, this->K_, this->M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype)1., top_diff,
        this->bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (this->transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->quant_.gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->M_, this->K_, this->N_,
         (Dtype)1., top_diff, this->quant_.gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FQInnerProductLayer);

}  // namespace caffe
