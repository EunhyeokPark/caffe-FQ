#include <vector>

#include "caffe/layers/fq_active_layer.hpp"

namespace caffe {

template <typename Dtype>
void FQActiveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  
  this->blobs_.resize(1);
  vector<int> weight_shape(1);
  weight_shape[0] = 4;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  Dtype* quant_data = this->blobs_[0]->mutable_gpu_data();
  quant_data[0] = static_cast<Dtype>(8); // default level
  quant_data[1] = 0;  // diff
  quant_data[2] = 0;  // min_level
  quant_data[3] = 0;  // max_level
}

#ifdef CPU_ONLY
STUB_GPU(FQActiveLayer);
#endif

INSTANTIATE_CLASS(FQActiveLayer);
REGISTER_LAYER_CLASS(FQActive);

}  // namespace caffe
