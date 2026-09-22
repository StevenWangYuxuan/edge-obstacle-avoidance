import onnx

m14 = onnx.load('/home/steven/day6-llm/01_models/distilgpt2_onnx/model.onnx')
ops14 = set(n.op_type for n in m14.graph.node)
isnan14 = [n for n in m14.graph.node if 'IsNaN' in n.op_type]
print(f"Opset 14: {len(ops14)} unique ops, IsNaN ops: {len(isnan14)}")

m18 = onnx.load('/home/steven/day6-llm/01_models/distilgpt2_onnx_v18/model.onnx')
ops18 = set(n.op_type for n in m18.graph.node)
isnan18 = [n for n in m18.graph.node if 'IsNaN' in n.op_type]
print(f"Opset 18: {len(ops18)} unique ops, IsNaN ops: {len(isnan18)}")

only14 = ops14 - ops18
only18 = ops18 - ops14
print(f"Only in 14: {sorted(only14)}")
print(f"Only in 18: {sorted(only18)}")
print(f"Common: {len(ops14 & ops18)}")
