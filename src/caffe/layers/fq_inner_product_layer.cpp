#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fq_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FQInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);

  this->quant_.ReshapeLike(*this->blobs_[0]);

  int blob_size = this->blobs_.size();
  this->blobs_.resize(blob_size+1);
  vector<int> weight_shape(1);
  weight_shape[0] = 4;
  this->blobs_[blob_size].reset(new Blob<Dtype>(weight_shape));

  Dtype* quant_data = this->blobs_[blob_size]->mutable_gpu_data();
  quant_data[0] = static_cast<Dtype>(8); // default level
  quant_data[1] = 0;  // diff
  quant_data[2] = 0;  // min_level
  quant_data[3] = 0;  // max_level

  this->param_propagate_down_.push_back(false);
}

#ifdef CPU_ONLY
STUB_GPU(FQInnerProductLayer);
#endif

INSTANTIATE_CLASS(FQInnerProductLayer);
REGISTER_LAYER_CLASS(FQInnerProduct);

}  // namespace caffe
