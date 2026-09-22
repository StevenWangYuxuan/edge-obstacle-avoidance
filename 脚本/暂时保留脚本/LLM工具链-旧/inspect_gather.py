import onnx

m = onnx.load('/home/steven/day6-llm/01_models/distilgpt2_qnn.onnx')
print("Gather nodes:")
for n in m.graph.node:
    if n.op_type == 'Gather':
        attrs = {}
        for a in n.attribute:
            attrs[a.name] = a.i if a.type == 2 else str(list(a.ints)) if a.type == 7 else '?'
        axis = attrs.get('axis', '?')
        print(f"  {n.name}: axis={axis}, inputs={n.input}")
