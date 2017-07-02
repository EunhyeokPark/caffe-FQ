#ifndef CAFFE_FQ_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_FQ_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief fixed-point quantization version for "fully-connected" layer
 *
 */
template <typename Dtype>
class FQInnerProductLayer : public InnerProductLayer<Dtype> {
 public:
  explicit FQInnerProductLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FQInnerProduct";}


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

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
