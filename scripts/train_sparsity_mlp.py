import torch
import nanopq
import statistics
import numpy as np
import io
import argparse
from time import time

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

import whisper

parser = argparse.ArgumentParser(description='Your script description here')
parser.add_argument('--layer', default=0, type=int, help='An integer parameter')
args = parser.parse_args()

# --- res: cf end of file --- 
in_model_file=None
NLAYERS=24
outpath="/data/whisper-sparsity"
# save mlp to...  
#in_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/3b-official-sparsity/official.pth'
#out_model_file = '/home/bfr4xr/RWKV-LM/RWKV-v5/out/3b-official-sparsity/official-mlp.pth'
out_model_file = "whisper_medium_decoder_ffn_mlp.pth"

###############################################
TEST_LAYERS = range(0, NLAYERS)
#TEST_LAYERS = [0, NLAYERS//2, NLAYERS-1]   # sample
#TEST_LAYERS = [args.layer]   # sample
###############################################
# TEST_THR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#TEST_THR = [0.5, 0.6, 0.7, 0.8]
TEST_THR = [0.8]
###############################################

mlp_map = [0.6] * 5 + [0.7] * 5 + [0.7] * 5 + [0.7] * 5 + [0.8] * 4
mlp_map = [0.5] * 24

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

# Define a simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        # hidden dim
        self.ddd = 64   # no much better
        #self.ddd = 256  # balanced
        # self.ddd = 32       # no much worse than 32
        # Two fully connected layers, bias=None as in dejavu
        self.fc1 = nn.Linear(input_dim, self.ddd, bias=None)  
        self.fc2 = nn.Linear(self.ddd, output_dim, bias=None) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation after first layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Define a weight initialization function
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero

# return: pq, X_code
def train_layer_pq(layer_id): 
    global weights, inputs, labels, model_dict

    # XXXX only takes 80% of it 
    X = weights[layer_id].cpu().numpy().T.astype(np.float32)         # (3.5D, D)
    Xt = X  # training data
    pq = nanopq.PQ(M=8, # sub-spaces
                   Ks=256, verbose=False, metric='dot')    
    # Train codewords
    pq.fit(Xt)
    # Encode to PQ-codes
    X_code = pq.encode(X)
    return pq, X_code

# Custom n-bit quantization function
def quantize(tensor, bit):
    factor = pow(2, bit) - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / factor  # 2^2 - 1 = 3
    zero_point = min_val
    quantized = ((tensor - zero_point) / scale).round().clamp(0, factor).to(torch.int8)
    return quantized, scale, zero_point

# Custom n-bit dequantization function
def dequantize(quantized, scale, zero_point):
    return quantized.to(torch.half) * scale + zero_point

    
def train_layer(layer_id):
    global weights, inputs, labels, model_dict, outputs_gt, mlp_lns

    # weights[0] (D,4xD)
    D1 = weights[layer_id].shape[0]
    D2 = weights[layer_id].shape[1]   # # of cols -- # of vectors to be indexed
    batch_size = 16

    # train/val split 
    n_batches = labels[layer_id].shape[0] // batch_size
    n_batches_val = n_batches // 5    #20% for validation
    n_batches_train = n_batches - n_batches_val

    N_TRAIN = n_batches_train * batch_size

    # training data  ... T/F labels    
    mlpinput = inputs[layer_id][:N_TRAIN].view(-1,batch_size,D1).to(torch.float32)
    mlplabel = labels[layer_id][:N_TRAIN].view(-1,batch_size,D2).to(torch.float32)

    # val data ... 
    val_inputs =         inputs[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D1).to(torch.float32)
    val_labels =         labels[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D2).to(torch.float32)
    val_outputs_gt = outputs_gt[layer_id][N_TRAIN:N_TRAIN+n_batches_val*batch_size].view(-1,batch_size,D2).to(torch.float32)

    www = torch.numel(mlplabel) / torch.sum(mlplabel) - 1  # == #false/#true

    model = MLP(D1, D2).to(torch.float32).to('cuda') 
    class_weight = torch.tensor([1.0, www])  # Higher weight for the minority class
    # class_weight = torch.tensor([1.0, 1.0])  # no weight -- balanced classes, should be BAD (skewed classes)
    # class_weight = torch.tensor([www, 1.0])  # sanity check -- should be VERY bad
    # class_weight = torch.tensor([1.0, 5.0])  # fixed weight
    # loss_fn = nn.BCELoss(weight=class_weights.to('cuda'))  # Binary Cross-Entropy Loss for binary classification
    loss_fn = nn.BCELoss(reduction='none')  # Use no reduction initially to apply per-sample weights
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.apply(weights_init)

    # --------- Validation w/o training. acc~=50%----------
    # model.eval()  # Set model to evaluation mode
    # with torch.no_grad():  # Disable gradient computation for validation
    #     val_outputs = model(val_inputs)
    #     val_loss = loss_fn(val_outputs, val_labels)

    #     # Compute accuracy (considering outputs > 0.5 as True, else False)
    #     predicted = (val_outputs > 0.5).float()
    #     correct = (predicted == val_labels).float().sum()
    #     val_accuracy = correct / (val_labels.numel())

    #     print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%")

    epochs = 100
    for epoch in range(epochs):
        model.train()  # Set the model in training mode
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(mlpinput)
        
        # breakpoint()
        # loss = loss_fn(outputs, mlplabel)
        
        per_sample_loss = loss_fn(outputs, mlplabel)
        weighted_loss = per_sample_loss * (mlplabel * class_weight[1] + (1 - mlplabel) * class_weight[0])
        loss = weighted_loss.mean()
        
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # check progress
        if False: 
            if epoch % 5 ==0: 
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():  # Disable gradient computation for validation
                    val_outputs = model(val_inputs)

                    # Compute accuracy (considering outputs > 0.5 as True, else False)
                    predicted = (val_outputs.abs() > 0.5).float()
                    # predicted = (val_outputs > 0.35).float()          # xzl: can play with this 

                    # Compute recall
                    true_positives = (predicted * val_labels).sum()  # Count of TP
                    false_negatives = ((1 - predicted) * val_labels).sum()  # Count of FN
                    recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add epsilon to avoid division by zero

                    print(f'epoch {epoch:03d}: layer {layer_id} sparsity: true {1-torch.sum(val_labels)/torch.numel(val_labels):.3f} pred {1-torch.sum(predicted)/torch.numel(predicted):.3f}')
                    print(f"    Validation Recall: {recall.item() * 100:.2f}%")

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # print("Training complete.")

    # ---  save this MLP to an existing model file ---- # 
    model_dict = None
    if model_dict is not None:          
        # model_dict[f'blocks.{layer_id}.mlp'] = model.state_dict()
        model_dict[f'decoder.blocks.{layer_id}.mlp_fc1.weight'] = model.state_dict()['fc1.weight']
        model_dict[f'decoder.blocks.{layer_id}.mlp_fc2.weight'] = model.state_dict()['fc2.weight']

        if layer_id == NLAYERS-1:
            #print(model_dict)
            torch.save(model_dict, out_model_file)   # OK to overwrite
            print(f'>>>>>>>>>> saved blocks.0-{layer_id}.mlp to:' + out_model_file)

    # --- final ----- #
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        val_outputs = model(val_inputs)
        # val_loss = loss_fn(val_outputs, val_labels)
        val_loss = loss_fn(val_outputs, val_labels).mean()

        # Compute accuracy (considering outputs > 0.5 as True, else False)
        thr = mlp_map[layer_id]
        predicted = (val_outputs > thr).float()         # xzl: can play with this 

        # predicted: tensor shape (#batches, batch_size, D2)
        correct = (predicted == val_labels).float().sum()
        val_accuracy = correct / (val_labels.numel())

        # Compute recall
        true_positives = (predicted * val_labels).sum()  # Count of TP
        false_negatives_mask = ((1 - predicted) * val_labels)  # 0/1 masks, (#batches, batch_size, D2)
        
        # -- debug -- 
        false_neg_vals = val_outputs_gt[false_negatives_mask.bool()]  # Collect values from outputs_gt where mask is 1
        false_neg_vals1 = val_outputs_gt * false_negatives_mask # tensor version, not collecting values
        true_pos_vals = val_outputs_gt[(predicted * val_labels).bool()]  # Collect values from outputs_gt where mask is 1

        # Find top K values in false_neg_vals1 and their locations
        K = 20
        # topK logits across all inputs
        top_k_false_neg_vals1_all, top_k_indices1_all = torch.topk(false_neg_vals1.view(-1), K)  
        # topK logits per input, their indices
        top_k_false_neg_vals1, top_k_indices1 = torch.topk(false_neg_vals1, K)

        # Print top K values
        top_k_false_neg_vals, top_k_indices = torch.topk(false_neg_vals, K)
        top_k_true_pos_vals = torch.topk(true_pos_vals, K).values

        #print(f'Top {K} false negative values: {top_k_false_neg_vals}')
        #print(f'Top {K} true positive values: {top_k_true_pos_vals}')

        #print(f'true_pos_vals: min {true_pos_vals.min()} max {true_pos_vals.max()} mean {true_pos_vals.mean()} median {torch.median(true_pos_vals)}')
        #print(f'false_neg_vals: min {false_neg_vals.min()} max {false_neg_vals.max()} mean {false_neg_vals.mean()} median {torch.median(false_neg_vals)}')
        
        # breakpoint()

        false_negatives = false_negatives_mask.sum()  # Count of FN
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add epsilon to avoid division by zero

        #print(f'layer {layer_id} thr {thr} sparsity: true {1-torch.sum(val_labels)/torch.numel(val_labels):.3f} pred {1-torch.sum(predicted)/torch.numel(predicted):.3f}')
        # breakpoint()

        #print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item() * 100:.2f}%, Recall: {recall.item() * 100:.2f}%")

        # --- add on: try pq ---- #
        if True: 
            pq, X_code = train_layer_pq(layer_id)
            query = val_inputs.cpu().numpy().astype(np.float32) # all val inputs. can only do float32
            dists = pq.dtable(query[0][0]).adist(X_code) # can only query with a single input 

            # --sanity check: run the orig model to get the ground truth output
            kw = weights[layer_id]
            kx = val_inputs[0][0].half()
            kx = mlp_lns[layer_id](kx)
            k = kx @ kw
            vx = F.gelu(k)
            #vx = torch.relu(k) ** 2
            #assert torch.equal(vx, val_outputs_gt[0][0].half())

            gt_output = val_outputs_gt[0][0].cpu().numpy()
            negs_idx = top_k_indices1[0][0].cpu().numpy()
            # print(f"gt_output at false neg -- {gt_output[negs_idx]}")
            # print(f"distance at false neg -- {dists[negs_idx]}")
            
            # --- run the orig model quantized, 8 bit
            # Define the quantization configuration  (quant only works float
            kw = kw.cpu().float()
            kx = kx.cpu().float()
            observer = torch.quantization.default_observer()
            observer(kw)
            scale_kw, zero_point_kw = observer.calculate_qparams()
            observer(kx)
            scale_kx, zero_point_kx = observer.calculate_qparams()
            # Quantize the weights and input tensors
            kw_quantized = torch.quantize_per_tensor(kw, scale_kw.item(), zero_point_kw.item(), dtype=torch.qint8)
            kx_quantized = torch.quantize_per_tensor(kx, scale_kx.item(), zero_point_kx.item(), dtype=torch.qint8)
            # Calculate the output scale and zero_point
            output_scale = scale_kx.item() * scale_kw.item()
            output_zero_point = 0  # Typically set to 0 for simplicity
            # Perform quantized matrix multiplication
            result_quantized = torch.ops.quantized.matmul(kx_quantized, kw_quantized, output_scale, output_zero_point)
            # Dequantize the result to get it back to floating-point
            result = result_quantized.dequantize()
            # print(f"quantgized res at false neg -- {result[negs_idx]}")
            # breakpoint()            
            '''
            ex: GT: 209 activated; quant: 169 activated (95% percentile), only 1 false positive

            percentile = torch.quantile(result, 0.95).item()
            msk=(result>percentile)
            masked=val_outputs_gt[0][0].cpu()[msk]
            (result>percentile_90).sum()  # num activated -- according to quantization 
            (val_outputs_gt[0][0].cpu()>0).sum()  # GT num activated
            (masked==0).sum()    # false positive
            '''

            #  --- 4 bit quantization
            start = time()
            kw_4bit, scale_kw_4bit, zero_kw_4bit = quantize(kw, 4)
            end = time()
            #print(f'4bit kw quantization time: {end-start}')
            kx_4bit, scale_kx_4bit, zero_kx_4bit = quantize(kx, 4)
           
            start = time()
            kw_4bit_dequant = dequantize(kw_4bit, scale_kw_4bit, zero_kw_4bit)
            end = time()
            #print(f'4bit kw dequantization time: {end-start}')
            kx_4bit_dequant = dequantize(kx_4bit, scale_kx_4bit, zero_kx_4bit)
            result_4bit = kx_4bit_dequant @ kw_4bit_dequant

            # --- 2-bit quantization
            start = time()
            kw_2bit, scale_kw_2bit, zero_kw_2bit = quantize(kw, 2)
            end = time()
            #print(f'2bit kw quantization time: {end-start}')
            kx_2bit, scale_kx_2bit, zero_kx_2bit = quantize(kx, 2)

            start = time()
            kw_2bit_dequant = dequantize(kw_2bit, scale_kw_2bit, zero_kw_2bit)
            end = time()
            #print(f'2bit kw dequantization time: {end-start}')
            kx_2bit_dequant = dequantize(kx_2bit, scale_kx_2bit, zero_kx_2bit)
            result_2bit = kx_2bit_dequant @ kw_2bit_dequant

            start = time()
            kw_1bit, scale_kw_1bit, zero_kw_1bit = quantize(kw, 1)
            end = time()
            #print(f'1bit kw quantization time: {end-start}')
            kx_1bit, scale_kx_1bit, zero_kx_1bit = quantize(kx, 1)

            start = time()
            kw_1bit_dequant = dequantize(kw_1bit, scale_kw_1bit, zero_kw_1bit)
            end = time()
            #print(f'1bit kw dequantization time: {end-start}')
            kx_1bit_dequant = dequantize(kx_1bit, scale_kx_1bit, zero_kx_1bit)
            result_1bit = kx_1bit_dequant @ kw_1bit_dequant

            '''
            ex: GT: 209 activated; quant: 139 activated (95% percentile), 24 false positive
            '''

            per = 0.8  # percetile we use to take quant as activated            
            result=result_2bit.float()
            percentile = torch.quantile(result, per).item()
            # msk=(result>percentile)
            # masked=val_outputs_gt[0][0].cpu()[msk]
            
            mlp_pred = predicted[0][0].cpu().int()
            quant_pred = (result > percentile).int()
            ensemble_pred = mlp_pred | quant_pred
            gt_labels = val_labels[0][0].cpu()  # GT labels
            gt_sparsity = 1 - torch.sum(gt_labels) / torch.numel(gt_labels)
            #print(f'\033[91mGT sparsity {gt_sparsity:.2f}\033[0m')

            # -- mlp perf
            tp = (mlp_pred * gt_labels).sum()
            fn = ((1 - mlp_pred) * gt_labels).sum()
            recall = tp / (tp + fn)
            sparsity = 1 - torch.sum(mlp_pred) / torch.numel(mlp_pred)
            mlp_recall = recall
            mlp_sparsity = sparsity
            #print(f'\033[91mMLP thr {thr} tp {tp} fn {fn} recall {recall} sparsity {sparsity:.2f}\033[0m')

            # -- quant perf
            tp = (quant_pred * gt_labels).sum()
            fn = ((1 - quant_pred) * gt_labels).sum()
            recall = tp / (tp + fn)
            sparsity = 1 - torch.sum(quant_pred) / torch.numel(result)
            quant_recall = recall
            quant_sparsity = sparsity
            #print(f'\033[91mQUANT per {per} tp {tp} fn {fn} recall {recall} sparse {sparsity:.2f}\033[0m')

            # -- ensemble perf
            tp = (ensemble_pred * gt_labels).sum()
            fn = ((1 - ensemble_pred) * gt_labels).sum()
            recall = tp / (tp + fn)
            sparsity = 1 - torch.sum(ensemble_pred) / torch.numel(ensemble_pred)
            ensemble_recall = recall
            ensemble_sparsity = sparsity
            #print(f'\033[91mENSENBLE tp {tp} fn {fn} recall {recall} sparse {sparsity:.2f}\033[0m')

            summary = True
            if summary:
                print(f"{gt_sparsity:.2f} & {mlp_sparsity:.2f} & {mlp_recall:.2f} & {quant_sparsity:.2f} & {quant_recall:.2f} & {ensemble_sparsity:.2f} & {ensemble_recall:.2f} \\\\")
                

            # for K in [50]:
            #     largest_k_indices = np.argpartition(dists, -K)[-K:]
            #     largest_k_items = dists[largest_k_indices]            

        #breakpoint()
        if False: 
            pq, X_code = train_layer_pq(layer_id)
            # query = val_inputs.cpu().numpy().astype(np.float32) # can only do float32
            query = val_inputs.cpu().numpy().astype(np.float32)     
            dists = pq.dtable(query[0][0]).adist(X_code) # can only query single vector
            predicted = predicted[0][0]
            val_labels = val_labels[0][0]

            for K in [50, 100, 200, 400, 800]:
                largest_k_indices = np.argpartition(dists, -K)[-K:]
                # largest_k_items = dists[largest_k_indices]
                largest = torch.from_numpy(largest_k_indices)
                over = predicted[largest]
                comp = ((1 - predicted) * val_labels)[largest]   # select from false negative
                print(f'K {K} overlapped {over.sum()} complementary {comp.sum()}')


def load_a_tensor(file_path):
    """
    Load the a  tensor from the file.
    """
    try:
        data = torch.from_numpy(np.load(file_path))
        #data = torch.load(file_path, map_location=torch.device('cuda'), weights_only=True)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    
def load_tensors(file_path):
    """
    Load the list of tensors from the file.
    """
    try:
        data = []
        temp = torch.from_numpy(np.load(file_path))
        for row in temp:
            data.append(row.to("cuda"))

        if isinstance(data, list):
            return data
    except FileNotFoundError:
        print("File not found.")
        return []
    
if __name__ == '__main__':
    # ---------- load from file 
    weights={}  # dict:layer_id->ffnkey (D,3.5xD)
    inputs={}   # dict: layer_id -> 2D tensors, (# inputs, D)]
    outputs_gt={}   # dict: layer_id -> 2D tensors (# inputs, 3.5D)  as from the orig model
    labels={}   # dict: layer_id -> 2D tensors (# inputs, 3.5D) True/False
    mlp_lns = {}

    #model_dict = whisper.load_model("medium").state_dict()
    model_dict = {}

    #if in_model_file != None:
    #    model_dict = torch.load(in_model_file)

    print("Ground-truth")
    for layer_id in TEST_LAYERS:    
        outpath_query=f'{outpath}/whisper_medium_decoder_ffn_query/FFN.key-layer{layer_id}-query.npy'
        outpath_weights=f'{outpath}/whisper_medium_decoder_ffn_weight/FFN.key-layer{layer_id}-weights.npy'
        outpath_ln=f'{outpath}/whisper_medium_decoder_ffn_ln_recording/FFN.key-layer{layer_id}-ln.npy'
        #outpath_query=f'{outpath}/FFN.key-layer{layer_id}-query.npy'

        w=load_a_tensor(outpath_weights).half().to("cuda")
        weights[layer_id]=w.t()

        q=load_tensors(outpath_query)
        ln_w, ln_b = np.load(outpath_ln)
        mlp_lns[layer_id] = LayerNorm(w.shape[1]).to("cuda")
        mlp_lns[layer_id].weight.data = torch.from_numpy(ln_w).to("cuda")
        mlp_lns[layer_id].bias.data = torch.from_numpy(ln_b).to("cuda")

        inputs[layer_id]=torch.stack(q)

        ## gen T/F labels by running the actual matmul
        kw = weights[layer_id]
        kx = inputs[layer_id]
        kx = mlp_lns[layer_id](kx)
        k = kx @ kw
        vx = F.gelu(k)
        #vx = torch.relu(k) ** 2
        # nz_mask = ~torch.eq(vx, 0)
        # num_nzeros = torch.sum(nz_mask).item()
        # num_zeros = nz_mask.shape[-1] - num_nzeros
        outputs_gt[layer_id] = vx
        neg_mask = (vx <= -0.01) & (vx > -0.169)
        pos_mask = (vx > 0)
        mask = neg_mask | pos_mask
        temp = torch.where(mask, vx, torch.tensor(0.0)).float() # abs() to make it binary
        labels[layer_id] = (temp > 0).int()

        # ------------ check for sparsity ....
        #onehot = (vx>0)
        onehot = labels[layer_id]
        #print(f'layer {layer_id} avg sparsity {1-torch.count_nonzero(onehot)/torch.numel(onehot)}')

        # train mlp    
        train_layer(layer_id)
        

    if model_dict:
        print(f'saved to:' + out_model_file)
