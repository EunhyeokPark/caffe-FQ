#ifndef CAFFE_FQ_CONV_LAYER_HPP_
#define CAFFE_FQ_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

/**
 * @brief Fixed-point quantization applied Convolution Layer
 *  This layer does not support CuDNN, because it is implemented based on original Caffe code
 */
template <typename Dtype>
class FQConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit FQConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FQConvolution"; }

 protected:
  // cpu functions are not implemented
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
      { LOG(FATAL) << "FQ cpu functions are not implemented"; }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      { LOG(FATAL) << "FQ cpu functions are not implemented"; }

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> quant_; 
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
