"""
Remove IsNaN ops from ONNX model by replacing with Equal-based NaN detection.
IsNaN(x) ≡ Not(Equal(x, x)) because NaN != NaN per IEEE 754.
Pattern: Softmax → IsNaN → Where(IsNaN, zeros, Softmax)
Replace: Softmax → Equal(x,x) → Where(Equal, x, zeros)
"""
import onnx
from onnx import helper

INPUT_PATH = '/home/steven/day6-llm/01_models/distilgpt2_sim.onnx'
OUTPUT_PATH = '/home/steven/day6-llm/01_models/distilgpt2_qnn.onnx'

def remove_isnan(model_path, output_path):
    m = onnx.load(model_path)
    graph = m.graph

    # Find IsNaN → Where pattern
    nodes_to_remove = set()
    new_nodes = []

    for node in graph.node:
        if node.op_type == 'IsNaN':
            isnan_name = node.name
            isnan_input = node.input[0]  # Softmax output
            isnan_output = node.output[0]

            # Find the Where node that uses this IsNaN output
            where_node = None
            for n in graph.node:
                if n.op_type == 'Where' and isnan_output in n.input:
                    where_node = n
                    break

            if where_node is None:
                print(f"WARNING: IsNaN {isnan_name} has no Where consumer, skipping")
                continue

            # Where(condition=IsNaN_out, X=zeros, Y=softmax_out)
            # Replace with: Where(Equal(x,x), x, zeros) — invert condition

            softmax_out = isnan_input
            equal_name = isnan_name.replace('IsNaN', 'EqualSafe')

            # Create Equal node: Equal(softmax, softmax) — True for non-NaN
            equal_node = helper.make_node(
                'Equal',
                inputs=[softmax_out, softmax_out],
                outputs=[equal_name + '_output_0'],
                name=equal_name
            )

            # Modify Where node: swap X and Y to account for inverted condition
            # Original: Where(IsNaN=NaN_mask, X=zeros, Y=S) → if NaN→0 else S
            # New:      Where(Equal=not_NaN, X=S, Y=zeros) → if not_NaN→S else 0
            where_new_inputs = [
                equal_node.output[0],  # new condition: not_NaN
                softmax_out,            # X: original value (when True/not NaN)
                where_node.input[1]     # Y: zeros (when False/is NaN)
            ]
            where_node.input[:] = where_new_inputs

            nodes_to_remove.add(node.name)
            new_nodes.append(equal_node)
            print(f"  Replaced: {node.name} (IsNaN) → {equal_name} (Equal) + rewrite {where_node.name}")

    # Rebuild graph with Equal nodes placed before their Where consumers
    inserted = set()
    result_nodes = []
    for n in graph.node:
        if n.name in nodes_to_remove:
            continue
        for eq_node in new_nodes:
            if eq_node.output[0] in n.input and eq_node.name not in inserted:
                result_nodes.append(eq_node)
                inserted.add(eq_node.name)
        result_nodes.append(n)
    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(result_nodes)

    onnx.checker.check_model(m)
    onnx.save(m, output_path)

    remaining = sum(1 for n in m.graph.node if n.op_type == 'IsNaN')
    print(f"\nDone: {len(nodes_to_remove)} IsNaN removed, {remaining} remaining")
    print(f"Saved: {output_path}")

remove_isnan(INPUT_PATH, OUTPUT_PATH)
