"""
Replace the dynamic Gather_6 with a static Slice operation.
Gather_6 extracts last-token logits from the flattened output.

Original: Gather(Flatten_output, Add_1_output) axis=0
Replace: Reshape → Slice last position → Squeeze
"""
import onnx
from onnx import helper
import numpy as np

INPUT_PATH = '/home/steven/day6-llm/01_models/dlc/distilgpt2_fp32.dlc'

# Load the ONNX model, not the DLC
ONNX_PATH = '/home/steven/day6-llm/01_models/distilgpt2_qnn.onnx'
OUTPUT_PATH = '/home/steven/day6-llm/01_models/distilgpt2_fixed.onnx'

print("Loading ONNX model...")
m = onnx.load(ONNX_PATH)
graph = m.graph

# Find Gather_6 (the one with Flatten_output_0 and Add_1_output_0)
gather6_node = None
for n in graph.node:
    if n.name == '/transformer/Gather_6':
        gather6_node = n
        break

if gather6_node is None:
    print("ERROR: Gather_6 not found!")
    exit(1)

print(f"Found: {gather6_node.name}")
print(f"  Inputs: {gather6_node.input}")
print(f"  Outputs: {gather6_node.output}")

# Find consumers of Gather_6 output
gather_output = gather6_node.output[0]
consumers = [n for n in graph.node if gather_output in n.input]
print(f"  Consumers: {[n.name for n in consumers]}")

# The consumer should be a Reshape that feeds into lm_head
# If we know gather is getting the last token (index 31 for seq=32),
# we can replace with Slice

# For a static seq_len=32, the last token index is 31
# Flatten_output shape: [batch*seq, hidden] = [32, 768]
# We want: [batch, hidden] at position seq-1 = [1, 768]

# Strategy: Reshape to [batch, seq, hidden] → Slice[:, -1, :] → squeeze

flatten_output = gather6_node.input[0]  # /transformer/Flatten_output_0
add_output = gather6_node.input[1]      # dynamic index (problem!)

# Create Reshape to [1, 32, 768]
reshape_name = '/transformer/Gather_6_fix/Reshape'
reshape_out = reshape_name + '_output_0'
reshape_node = helper.make_node(
    'Reshape',
    inputs=[flatten_output, reshape_name + '_shape'],
    outputs=[reshape_out],
    name=reshape_name
)

# Create shape constant [1, 32, 768]
shape_tensor = helper.make_tensor(
    reshape_name + '_shape',
    onnx.TensorProto.INT64,
    dims=[3],
    vals=np.array([1, 32, 768], dtype=np.int64)
)
shape_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=[reshape_name + '_shape'],
    name=reshape_name + '_shape_const',
    value=shape_tensor
)

# Create Slice to get last token: slice along axis=1, start=31, end=32
slice_name = '/transformer/Gather_6_fix/Slice'
slice_out = slice_name + '_output_0'
# Slice in opset 18 needs starts, ends, axes, steps as inputs
starts_tensor = helper.make_tensor(
    slice_name + '_starts',
    onnx.TensorProto.INT64,
    dims=[1],
    vals=np.array([31], dtype=np.int64)
)
ends_tensor = helper.make_tensor(
    slice_name + '_ends',
    onnx.TensorProto.INT64,
    dims=[1],
    vals=np.array([32], dtype=np.int64)
)
axes_tensor = helper.make_tensor(
    slice_name + '_axes',
    onnx.TensorProto.INT64,
    dims=[1],
    vals=np.array([1], dtype=np.int64)
)
steps_tensor = helper.make_tensor(
    slice_name + '_steps',
    onnx.TensorProto.INT64,
    dims=[1],
    vals=np.array([1], dtype=np.int64)
)

slice_node = helper.make_node(
    'Slice',
    inputs=[reshape_out,
            slice_name + '_starts',
            slice_name + '_ends',
            slice_name + '_axes',
            slice_name + '_steps'],
    outputs=[slice_out],
    name=slice_name
)

# Create Squeeze to remove seq dim: [1, 1, 768] → [1, 768]
squeeze_name = '/transformer/Gather_6_fix/Squeeze'
squeeze_out = gather6_node.output[0]  # Replace Gather_6 output
squeeze_node = helper.make_node(
    'Squeeze',
    inputs=[slice_out, squeeze_name + '_axes'],
    outputs=[squeeze_out],
    name=squeeze_name
)
squeeze_axes = helper.make_tensor(
    squeeze_name + '_axes',
    onnx.TensorProto.INT64,
    dims=[1],
    vals=np.array([1], dtype=np.int64)
)

# Build new graph
new_nodes = []
for n in graph.node:
    if n.name == '/transformer/Gather_6':
        # Replace with our static equivalent
        new_nodes.append(shape_node)
        new_nodes.append(reshape_node)
        # Add Slice constants as initializers instead of Constant nodes
        new_nodes.append(slice_node)
        new_nodes.append(squeeze_node)
    else:
        new_nodes.append(n)

# Add Slice/Squeeze constants as initializers
graph.initializer.extend([starts_tensor, ends_tensor, axes_tensor, steps_tensor, squeeze_axes])

# Clear and rebuild graph
while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(new_nodes)

# Validate and save
onnx.checker.check_model(m)
onnx.save(m, OUTPUT_PATH)

# Verify
m2 = onnx.load(OUTPUT_PATH)
gathers_left = [n for n in m2.graph.node if n.name == '/transformer/Gather_6']
print(f"\nDONE: Gather_6 removed. Remaining: {len(gathers_left)}")
print(f"Saved: {OUTPUT_PATH}")
