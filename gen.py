# 2, 5, 1
# generates the following 1bit linear step layers:
# lin(2,5), lin(5,1)

DO_TRAIN = True
DO_TEST = True
DO_DEBUG = False
PERF_MODE = True

# NO_BIT_RECALC = False
NO_BIT_RECALC = True

if NO_BIT_RECALC: print("// WARNING!!! YOU WILL NEED TO RECOMPILE 'bitnn.cpp' WITH '#define UPDATE_IN_BACKW'")

topo = [2, 8, 8, 1]

xs = [[0,0],[0,1],[1,0],[1,1]]

ys = [[0], [1], [1], [0]]
# N_EPOCH = 50_000_000
N_EPOCH = 500

# GENERATOR

init_mat_cpp = ""

defn_mat_cpp = ""
defn_bmat_cpp = ""

runnn_cpp = ""
runnn_signature_cpp = ""
runnn_call_signature_cpp = ""

backw_cpp_items = []
backw_signature_cpp = ""
backw_call_signature_cpp = ""

debug_print_cpp = ""

set_bmat_cpp = ""

for i in range(len(topo) - 1):
    in_size, out_size = topo[i: i+2]

    # random init of weight mats
    init_mat_cpp += f"\trandom2DInt8Mat<{in_size}, {out_size}>(lin{i}_w);\n"
    init_mat_cpp += f"\trandom1DInt8Mat<{out_size}>(lin{i}_b);\n"

    # the actual weights
    # define weight int8
    defn_mat_cpp += f"\tint8_t lin{i}_w[{out_size}][{in_size}];\n"

    # define bias int8
    defn_mat_cpp += f"\tint8_t lin{i}_b[{out_size}];\n"

    # the binary weights
    # weight
    defn_bmat_cpp += f"\tstd::bitset<{in_size}> lin{i}_w_bits[{out_size}];\n"
    set_bmat_cpp += f"\ttoBinMat<{in_size}, {out_size}>(lin{i}_w, lin{i}_w_bits);\n"

    # biases
    defn_bmat_cpp += f"\tstd::bitset<{out_size}> lin{i}_b_bits;\n"
    set_bmat_cpp += f"\tlin{i}_b_bits = posBits<{out_size}>(lin{i}_b);\n"

    # define matmul + bias + step activation (forward pass for a layer)
    layer_in = "nn_input" if i == 0 else f"lin{i-1}_out"
    layer_grad_in = "out_grad" if i == (len(topo)-2) else f"lin{i}_grad"
    layer_grad_out = "in_grad" if i == 0 else f"lin{i-1}_grad"

    runnn_cpp += f"\tstd::bitset<{out_size}> lin{i}_out = matStepBitVecMulBias<{in_size}, {out_size}>(lin{i}_w_bits, lin{i}_b_bits, {layer_in});\n"

    runnn_signature_cpp += f"std::bitset<{in_size}> lin{i}_w_bits[{out_size}], std::bitset<{out_size}> lin{i}_b_bits, "
    runnn_call_signature_cpp += f"lin{i}_w_bits, lin{i}_b_bits, "

    if NO_BIT_RECALC: backw_signature_cpp += f"int8_t lin{i}_w[{out_size}][{in_size}], int8_t lin{i}_b[{out_size}], std::bitset<{in_size}> lin{i}_w_bits[{out_size}], std::bitset<{out_size}>& lin{i}_b_bits, "
    else: backw_signature_cpp += f"int8_t lin{i}_w[{out_size}][{in_size}], int8_t lin{i}_b[{out_size}], "

    if NO_BIT_RECALC: backw_call_signature_cpp += f"lin{i}_w, lin{i}_b, lin{i}_w_bits, lin{i}_b_bits, "
    else: backw_call_signature_cpp += f"lin{i}_w, lin{i}_b, "

    # this will be reversed so do this backwards
    backw_cpp_items.append(f"\tbackwardBitStepMVBias<{in_size}, {out_size}>(lin{i}_w_bits, lin{i}_b_bits, {layer_in}, {layer_grad_in}_zero, {layer_grad_in}_sign, lin{i}_w, lin{i}_b, {layer_grad_out}_zero, {layer_grad_out}_sign);\n")
    backw_cpp_items.append(f"\tstd::bitset<{in_size}> {layer_grad_out}_zero;\n")
    backw_cpp_items.append(f"\tstd::bitset<{in_size}> {layer_grad_out}_sign;\n")

    weight_dump_cpp = "<< ".join(f"static_cast<int16_t>(lin{i}_b[{j}]) << \" \" " for j in range(out_size))
    debug_print_cpp += f"\tstd::cout << \"lin{i}_b: \" << {weight_dump_cpp} << \"\\n\";\n"

    weight_dump_cpp = "<< \"\\n\" << ".join("<<".join(f"static_cast<int16_t>(lin{i}_w[{k}][{j}]) << \" \" " for j in range(in_size)) for k in range(out_size))
    debug_print_cpp += f"\tstd::cout << \"lin{i}_w: \" << {weight_dump_cpp} << \"\\n\";\n"

NLAY = len(topo)-1
IN_SIZE = topo[0]
OUT_SIZE = topo[-1]

# backw_cpp_items.append(f"\tstd::bitset<{OUT_SIZE}> out_grad_zero;\n")
# backw_cpp_items.append(f"\tstd::bitset<{OUT_SIZE}> out_grad_sign;\n")

backw_cpp = "".join(backw_cpp_items[::-1])

# print(f"{backw_cpp}")

# exit()
runnn_signature_cpp = runnn_signature_cpp[:-2]
runnn_call_signature_cpp = runnn_call_signature_cpp[:-2]

backw_signature_cpp = backw_signature_cpp[:-2]
backw_call_signature_cpp = backw_call_signature_cpp[:-2]

# generate a template:

total_file = ""

total_file += "#include \"bitnn.cpp\"\n"
total_file += "#include <iostream> // std::cout\n"
total_file += "#include <bitset>   // std::bitset\n"
total_file += "\n"

total_file += "// run the neural network\n"
total_file += f"inline std::bitset<{OUT_SIZE}> _runNN(std::bitset<{IN_SIZE}> nn_input, {runnn_signature_cpp}) {{\n"
total_file += runnn_cpp
total_file += f"\treturn lin{NLAY-1}_out;\n"
total_file += "}\n"
total_file += "\n"

if DO_TRAIN:
    total_file += "// run the neural network\n"
    total_file += f"inline void _backwNN(std::bitset<{IN_SIZE}> nn_input, std::bitset<{OUT_SIZE}> expected_nn_output, {backw_signature_cpp}) {{\n"
    
    if not NO_BIT_RECALC:
        total_file += "\t// define bin mats to store the binarized matricies\n"
        total_file += f"{defn_bmat_cpp}\n"
        total_file += "\t// store the binarized matricies (this is done because presumably the weights have been updated previously)\n"
        total_file += f"{set_bmat_cpp}\n"
    
    total_file += "\t// run the NN (we still need the layer inputs)\n"
    total_file += runnn_cpp
    total_file += "\n"
    total_file += f"\tstd::bitset<{OUT_SIZE}> out_grad_zero;\n"
    total_file += f"\tstd::bitset<{OUT_SIZE}> out_grad_sign;\n"
    total_file += f"\tstd::bitset<{OUT_SIZE}> errors = lin{NLAY-1}_out ^ expected_nn_output;\n"
    total_file += f"\tif (errors.count() == 0) return;\n"

    if not PERF_MODE: total_file += f"\tstd::cout << \"expected: \" << expected_nn_output << \"\\n\";\n"
    if not PERF_MODE: total_file += f"\tstd::cout << \"got: \" << lin{NLAY-1}_out << \"\\n\";\n"
    if not PERF_MODE: total_file += f"\tstd::cout << \"error: \" << errors << \"\\n\";\n"
    total_file += f"\tout_grad_zero = ~errors;\n"
    # total_file += f"\tout_grad_sign = ~((~lin{NLAY-1}_out) & expected_nn_output);\n"
    total_file += f"\tout_grad_sign = lin{NLAY-1}_out | (~expected_nn_output);\n"
    if not PERF_MODE: total_file += f"\tstd::cout << \"out_grad_zero: \" << out_grad_zero << \"\\n\";\n"
    if not PERF_MODE: total_file += f"\tstd::cout << \"out_grad_sign: \" << out_grad_sign << \"\\n\";\n"
    total_file += f"{backw_cpp}\n"
    total_file += "}\n"
    total_file += "\n"

total_file += "// Autogenned infra for running the NN\n"
total_file += "int main(int argc, char const *argv[]){\n"
total_file += "\t// good srand\n"
total_file += "\tsrand(123);\n"

total_file += "\t// define the necessary int8 buffers for the nn mats\n"
total_file += f"{defn_mat_cpp}\n"
total_file += "\t// fill the nn mats with random int8s\n"
total_file += f"{init_mat_cpp}\n"
total_file += "\t// define bin mats to store the binarized matricies\n"
total_file += f"{defn_bmat_cpp}\n"
# total_file += "\t// store the binarized matricies\n"
# total_file += f"{set_bmat_cpp}\n"
total_file += f"\n"
# total_file += f"\t// TODO: set the input value here!!!\n"
# total_file += f"\tstd::bitset<{IN_SIZE}> nn_input (0b{'0'*IN_SIZE});\n"
# total_file += f"\tstd::bitset<{OUT_SIZE}> nn_output = _runNN(nn_input, {runnn_call_signature_cpp});\n"
# total_file += f"\tstd::cout << \"NN Result:\" << nn_output << \"\\n\";\n"
if DO_DEBUG: total_file += f"\n{debug_print_cpp}"

if DO_TRAIN or DO_TEST:
    total_file += f"\tstd::bitset<{IN_SIZE}> in_vals[] = {{{', '.join(map(lambda x: '0b'+''.join(map(str, map(int, x))), xs))}}};\n"
    total_file += f"\tstd::bitset<{OUT_SIZE}> out_vals[] = {{{', '.join(map(lambda x: '0b'+''.join(map(str, map(int, x))), ys))}}};\n"

if DO_TRAIN:
    if NO_BIT_RECALC:
        total_file += "\t// set the bits initially\n"
        total_file += f"{set_bmat_cpp}"
    total_file += "\t// training\n"
    total_file += f"\n"
    total_file += f"\tsize_t N_EPOCH = {N_EPOCH};\n"
    total_file += f"\tfor (size_t epoch = 0; epoch < N_EPOCH; epoch++){{\n"
    total_file += f"\t\tsize_t i = rand() % {len(xs)};\n"
    if not PERF_MODE: total_file += f"\t\tstd::cout << \"ep: \" << epoch << \" (i=\" << i << \")\" << \"\\n\";\n"
    total_file += f"\t\t_backwNN(in_vals[i], out_vals[i], {backw_call_signature_cpp});\n"
    total_file += "\t}\n"

    total_file += "\t// store the binarized matricies\n"
    total_file += f"{set_bmat_cpp}\n"
    total_file += "\n\n"

if DO_TEST:
    total_file += "\t// test the nn\n"
    total_file += f"\tfor (size_t i = 0; i < {len(xs)}; i++){{\n"
    total_file += f"\t\tstd::bitset<{OUT_SIZE}> res = _runNN(in_vals[i], {runnn_call_signature_cpp});\n"
    total_file += f"\t\tstd::cout << \"======================\" << \"\\n\";\n"
    total_file += f"\t\tstd::cout << \"IN: \" << in_vals[i] << \"\\n\";\n"
    total_file += f"\t\tstd::cout << \"OUT: \" << res << \"\\n\";\n"
    total_file += f"\t\tstd::cout << \"EXPECTED: \" << out_vals[i] << \"\\n\";\n"
    total_file += f"\t\tstd::cout << \"ERROR: \" << (out_vals[i] ^ res) << \"\\n\";\n"
    total_file += "\t}\n"

if DO_DEBUG: total_file += f"\n{debug_print_cpp}"

total_file += "}"


print(f"{total_file}")