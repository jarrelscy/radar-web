# Cut part_b at a tensor: .venv-conv/bin/python tools/cut.py <tensor> -> tools/cut.onnx (debug only)
import sys, onnx
from onnx.utils import Extractor
m = onnx.load('models/part_b.onnx')
m.graph.value_info.append(onnx.helper.make_tensor_value_info(sys.argv[1], onnx.TensorProto.FLOAT, None))
e = Extractor(m).extract_model(['t'], [sys.argv[1]])
onnx.save(e, 'tools/cut.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='cut.data')
