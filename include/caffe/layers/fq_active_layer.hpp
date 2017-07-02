#ifndef CAFFE_FQ_ACTIVE_LAYER_HPP_
#define CAFFE_FQ_ACTIVE_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief fixed point quantization for activation 
 *        
 */
template <typename Dtype>
class FQActiveLayer : public NeuronLayer<Dtype> {
 public:
  explicit FQActiveLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FQActive"; }

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
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
