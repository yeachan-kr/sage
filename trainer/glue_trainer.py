
from tqdm.auto import tqdm

import torch
import numpy as np
from transformers import Seq2SeqTrainer, Trainer
from utils.gen_data_module import cls_rev_label_map
from evaluate import load

import lovelyplots
import matplotlib.pyplot as plt

from utils.CKA import CudaCKA

plt.style.use('ipynb')

# colors = ColorHunt02().cmap
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def heatmap(data, row_labels, col_labels, ax = None,
            cbar_kw = {}, cbarlabel = "", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
      ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = "bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels = col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels = row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top = True, bottom = False,
                   labeltop = True, labelbottom = False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = -30, ha = "right",
             rotation_mode = "anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor = True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which =  "minor", bottom = False, left = False)

    return im

class GLUESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, tokenizer, dataset, max_input_len, max_output_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        # self.base_model = copy.deepcopy(self.model)
        self.num_row_ranks = 64
        self.diff_ranks = []

        self.base_model = None
            

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs):
    
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.dataset in ['stanfordnlp/imdb', 'SetFit/20_newsgroups', 'mrpc', 'yelp', 'piqa']:
            glue_metric = load('glue', 'sst2')
        else:
            glue_metric = load('glue', self.dataset)

        refs, preds = [], []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
            src_len = batch['source_len']
            batch['input_ids'] = batch['input_ids'][:, :src_len]
            batch['attention_mask'] = batch['attention_mask'][:, :src_len]

            # print(self.tokenizer.decode(batch['input_ids'][0]))
            self.model.set_source_len(src_len)
            output = self.model.generate(input_ids=batch['input_ids'],
                                         attention_mask=batch['attention_mask'],
                                         max_length=int(self.max_input_len + self.max_output_len), pad_token_id=self.tokenizer.pad_token_id)
            p = list(self.tokenizer.batch_decode(output[:, batch['input_ids'].size(1):batch['input_ids'].size(1)+2], skip_special_tokens=True))
            t = list(self.tokenizer.batch_decode(batch['labels'][:, batch['input_ids'].size(1):], skip_special_tokens=True))
            preds.extend(p)
            refs.extend(t)


        # formatting
        if self.dataset in cls_rev_label_map:
            # if preds in mapping dict, convert them back to original labels otherwise keep them as they are
            for i, pred in enumerate(preds):
                if pred in cls_rev_label_map[self.dataset]:
                    preds[i] = cls_rev_label_map[self.dataset][pred]
                else:
                    preds[i] = -1

            for i, ref in enumerate(refs):
                if ref in cls_rev_label_map[self.dataset]:
                    refs[i] = cls_rev_label_map[self.dataset][ref]
            
        results = glue_metric.compute(predictions=preds, references=refs)
        self.log(results)
        # print(results)
        # self.analysis(eval_dataset=eval_dataset, base_model=self.base_model)
        return results
    
class Seq2SeqTrainer(Trainer):  
    def __init__(self, method, dataset, max_input_len, max_output_len, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer
        self.best_acc = -1
        self.iter = 0
        self.tracking_layers = {}
        target_layers = [0, 6, 13, 20, 27]
        for i in target_layers:
            self.tracking_layers[i] = []

        self.cumulative_accs = []        
    
    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     ignore_keys=None,
    #     metric_key_prefix: str = "eval",
    #     **gen_kwargs):
        
    #     self.model.eval()
    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)


    #     with torch.no_grad():
    #         total_acc = 0
    #         total_count = 0
    #         refs, preds = [], []
    #         for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):    
    #             inputs = {
    #                     "input_ids": batch['input_ids'],
    #                     "attention_mask": batch['attention_mask'],
    #                     "max_new_tokens": 128
    #             }
            
    #             preds = self.model.generate(**inputs)
    #             preds = list(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
    #             targets = list(self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
    #             print(preds[0])
    #             print(targets[0])
    #             exit()
                
                
class GLUECLSTrainer(Trainer):
    def __init__(self, method, dataset, max_input_len, max_output_len, base_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.base_model = base_model
        self.best_acc = -1
        self.iter = 0
        self.tracking_layers = {}
        target_layers = [0, 6, 13, 20, 27]
        for i in target_layers:
            self.tracking_layers[i] = []

        self.cumulative_accs = []
        self.cumulative__acts = []

    def calculate_attention_sparsity(self, eval_dataset):
        self.model.eval()
        # 1. attention map
        # 2. sorting and cumulative attention
        # 3. plot (x: token index, y: cumulative attention)
        # 4. result (skewness? kurtosis?)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        target_layers = [0, 6, 13, 20, 27]
        layer_attns = {}
        for i in target_layers:
            layer_attns[i] = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
                main_output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_attentions=True)
                attentions = torch.stack(main_output.attentions)

                if len(layer_attns[0]) > 20:
                    break

                for layer in target_layers:
                    attn = attentions[layer]
                    attn = attn.sum(dim=1)
                    attn = attn[:, -1]
                    attn = attn / (attn.sum()+1e-6) # (batch_size, seq_len)

                    # sort attention
                    attn, _ = torch.sort(attn, dim=1, descending=True)
                    attn = attn.cumsum(dim=1) / attn.sum(dim=1, keepdim=True)
                    layer_attns[layer].append(attn.mean(dim=0).cpu())

                hist = self.model.model.layer_token_hist
                print(hist)
                exit()

            plt.figure(figsize=(7, 3))
            lidx = 0
            for layer, attns in layer_attns.items():
                attns = torch.stack(attns)
                attns = attns.mean(dim=0)
                # attns = attns / attns.sum()
                plt.plot(attns, label=f'Layer {layer+1}',marker='None')

                # set 90% energy
                # avg_energy = attns.sum()
                avg_energy = (attns > 0.90).nonzero(as_tuple=False)[0].item()
                if layer == 0 or layer == 27: # only plot for first and last layer
                    plt.axvline(x=avg_energy, color=colors[lidx], linestyle='--')
                lidx += 1 

                # tracking 95% energy token index
                self.tracking_layers[layer].append(avg_energy)

                
            num_points = 10
            plt.xticks(range(0, len(attns), int(len(attns)/num_points)), range(0, len(attns), int(len(attns)/num_points)))
            plt.xlabel('Sorted token index')
            plt.ylabel('Cumulative attention')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(f'./figs/attn_sparsity_layer_{self.iter}.pdf')
            plt.close()
            
            # magnified plot (first 50 tokens)
            plt.figure(figsize=(7, 3))
            lidx = 0
            for layer, attns in layer_attns.items():
                attns = torch.stack(attns)
                attns = attns.mean(dim=0)
                # attns = attns / attns.sum()
                plt.plot(attns[:50], label=f'Layer {layer+1}',marker='None')
                
                # set 90% energy
                # avg_energy = attns.sum()
                avg_energy = (attns > 0.90).nonzero(as_tuple=False)[0].item()
                if layer == 0 or layer == 27:
                    plt.axvline(x=avg_energy, color=colors[lidx], linestyle='--')
                lidx += 1
                
                
            num_points = 10
            plt.xticks(range(0, 50, int(50/num_points)), range(0, 50, int(50/num_points)))
            plt.xlabel('Sorted token index')
            plt.ylabel('Cumulative attention')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(f'./figs/attn_sparsity_layer_{self.iter}_magnified.pdf')
            # set 90% energy

            # plot tracking layers
            plt.figure(figsize=(7, 3))
            for layer, values in self.tracking_layers.items():
                plt.plot(values, label=f'Layer {layer+1}',marker='None')
            # plt.legend()
            max_tokens = 36
            num_points = 5
            num_steps = max(len(self.tracking_layers[0]), 10)
            plt.yticks(range(0, max_tokens, int(max_tokens/num_points)), range(0, max_tokens, int(max_tokens/num_points)))
            plt.xticks(range(0, num_steps, int(num_steps/num_points)), range(0, num_steps, int(num_steps/num_points)))
            plt.xlabel('Training iteration')
            plt.ylabel('Dominant tokens')
            plt.savefig(f'./figs/tracking_layers_{self.iter}.pdf')
            plt.close()
            # exit()

            self.iter += 1
                



    def calculate_convergence_ratio(self, eval_dataset):
        # layer-wise convergence ratio based on CKA similarity
        # original model - self.base_model
        self.model.eval()
        # self.base_model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        cuda_cka = CudaCKA(device=self.model.device)

        with torch.no_grad():
            layer_cka_similarities = {}
            for i in range(len(self.model.model.model.layers)):
                layer_cka_similarities[i] = []

            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):

                if len(layer_cka_similarities[0]) > 10:
                    break

                main_output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                with self.model.disable_adapter():
                    base_output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                
                main_hidden_states = main_output.hidden_states[1:]
                base_hidden_states = base_output.hidden_states[1:]

                for i, (mh, bh) in enumerate(zip(main_hidden_states, base_hidden_states)): # layer
                    sims = 0
                    ndata = len(mh)
                    for j in range(len(mh)): # batch
                        # cka = cuda_cka.linear_CKA(mh[j].view(-1, mh[j].shape[-1]).to(torch.float32), bh[j].view(-1, bh[j].shape[-1]).to(torch.float32))
                        # cosine similarity
                        seq_length = batch['attention_mask'][j].sum()
                        h1 = mh[j][-seq_length:]
                        h2 = bh[j][-seq_length:]
                        # print(h1.shape, h2.shape)
                        # sims += torch.nn.functional.cosine_similarity(h1, h2).mean().item()
                        sims += cuda_cka.linear_CKA(h1.view(-1, h1.shape[-1]).to(torch.float32), h2.view(-1, h2.shape[-1]).to(torch.float32))
                    layer_cka_similarities[i].append(sims/ndata)

            diff = []
            for i, cka_similarities in layer_cka_similarities.items():
                # print(f'Layer {i}: {sum(cka_similarities) / len(cka_similarities)}')
                scores = sum(cka_similarities) / len(cka_similarities)
                diff.append(1-scores)
            diff = torch.tensor(diff)
            # self.model.model.layer_probs = diff.to(self.model.device)
            # print('Layer probs:', self.model.model.layer_probs)

    def analysis(self, eval_dataset, acc):
        self.model.train()
        eval_dataloader = self.get_train_dataloader()

        cka = CudaCKA(device=self.model.device)

        target_layers = [3, 9, 15, 21, 27]
        # target_layers = [1, 7, 14, 21, 28]
        layer_singular_values_act = {}
        layer_singular_values_grad = {}
        for i in target_layers:
            layer_singular_values_act[i] = []
            layer_singular_values_grad[i] = []

        # with torch.no_grad():
        for bidx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)):
            if bidx == 50:
                break
            output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # 1. get activations
            loss = output.loss
            loss.backward()

            gradients = []
            activations = []
            for i in range(len(self.model.model.model.layers)):
                activations.append(self.model.model.model.layers[i].self_attn.q_activations)
                gradients.append(self.model.model.model.layers[i].self_attn.q_gradients)
            activations = torch.stack(activations)
            gradients = torch.stack(gradients)

            # 2. perform svd and get singular values
            # for j in range(batch['input_ids'].shape[0]): # iterate over samples
            # seq_len = activations.size(-2)
            for i in target_layers: # iterate over layers
                activation = activations[i].to(torch.float32).view(-1, activations.size(-1)).transpose(0, 1)
                gradient = gradients[i].to(torch.float32).view(-1, gradients.size(-1)).transpose(0, 1)

                u, s, v = torch.svd(activation)
                # cumulative energy of singular values
                s = s ** 2
                s = s.cumsum(dim=0) / s.sum()
                layer_singular_values_act[i].append(s[:128].detach().cpu())

                u, s, v = torch.svd(gradient)
                # cumulative energy of singular values
                s = s ** 2
                s = s.cumsum(dim=0) / s.sum()
                layer_singular_values_grad[i].append(s[:128].detach().cpu())

        # 3. average singular values across samples and layers
        avg_singular_values_act = {}
        avg_singular_values_grad = {}
        for k in layer_singular_values_act:
            avg_singular_values_act[k] = torch.stack(layer_singular_values_act[k]).mean(dim=0)
            avg_singular_values_grad[k] = torch.stack(layer_singular_values_grad[k]).mean(dim=0)

        # plot singular values for each layer
        plt.figure(figsize=(7, 3))
        for i, (k, v) in enumerate(avg_singular_values_act.items()):
            # add zero to v
            # v = torch.cat([torch.zeros(1), v])
            # line without marker
            plt.plot(v, label=f'Layer {k}', linestyle='-', marker='o')
            plt.xticks(range(0, len(v), 10), range(0, len(v), 10))

        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('position of singular value')
        plt.ylabel('cumulative singular value')
        plt.savefig(f'./singular_values_act_{self.dataset}_{acc}.pdf')
        plt.close()

        plt.figure(figsize=(7, 3))
        for i, (k, v) in enumerate(avg_singular_values_grad.items()):
            # add zero to v
            # v = torch.cat([torch.zeros(1), v])
            plt.plot(v, label=f'Layer {k}', linestyle='-', marker=None)
            # plt.xticks(range(len(v)), range(len(v)))
            # ticks every 10
            plt.xticks(range(0, len(v), 10), range(0, len(v), 10))
        plt.legend()

        # mark average 98% energy across layers
        avg_energy = torch.stack([v for v in avg_singular_values_grad.values()]).mean(dim=0)
        # avg_energy = avg_energy / avg_energy.sum()
        print(avg_energy)
        avg_engery = (avg_energy > 0.95).nonzero(as_tuple=False)[0].item()
        

        plt.axvline(x=avg_engery, color='red', linestyle='--', label='98% energy')


        plt.ylim(0, 1)
        plt.xlabel('position of singular value')
        plt.ylabel('cumulative singular value')
        plt.savefig(f'./singular_values_grad_{self.dataset}_{acc}.pdf')
        plt.close()

                
    def check_act_nums(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs):
    
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.dataset in ['stanfordnlp/imdb', '20news', 'mrpc', 'piqa', 'sqa', 'csqa', 'obqa','arc_easy', 'arc_chall', 'winogrande']:
            glue_metric = load('glue', 'sst2')
        else:
            glue_metric = load('glue', self.dataset)

        total_acc = 0
        total_count = 0
        all_hist = []
        self.model.train()
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
            with torch.no_grad():
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # logits = output.logits
            # preds = torch.argmax(logits, dim=1)
            # total_acc += torch.sum(preds == batch['labels']).item()
            # total_count += len(batch['labels'])

            hist = self.model.model.model.layer_token_hist
            hist = torch.tensor(hist).float()
            all_hist.append(hist)
            # print(hist)
        all_hist = torch.stack(all_hist)
        all_hist = all_hist.mean(dim=0)
        avg_acts = torch.mean(all_hist)
        all_hist = all_hist.cpu().numpy().tolist()
        all_hist = ['{0:.2f}'.format(all_hist[i]) for i in range(len(all_hist))]
        print(all_hist, avg_acts.item())
        self.cumulative__acts.append(avg_acts.item())
        print(self.cumulative__acts)
  

    def check_selected_tokens(self, eval_dataset):
        # target_layers = [0, 6, 13, 20, 27]
        target_layers = [27, 20, 13, 6, 0]
        self.iter += 1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.train()
        item_idx = 0
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
            with torch.no_grad():
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                tokens = self.model.model.model.selected_tokens
                num_tokens = self.model.model.model.layer_token_hist
                
                if item_idx > 50:
                    break

                for i in range(len(tokens)):
                    try:
                        data_len = batch['attention_mask'][i].sum().item()
                        if data_len > 24 and data_len < 64:
                            token_ids = batch['input_ids'][i][-data_len:]
                            target_token_dist = []
                            for t in target_layers:
                                token_dist = tokens[t][i][-data_len:]
                                num_t = num_tokens[t]
                                num_t = min(num_t, data_len)
                                
                                # set top num_t value to 1 otherwise 0 in token_dist
                                token_dist = torch.tensor(token_dist).detach().cpu()
                                top_token_dist = torch.topk(token_dist, num_t, dim=0).indices
                                token_dist = torch.zeros(len(token_dist)).scatter_(0, top_token_dist, 1)

                                target_token_dist.append(token_dist)
                            target_token_dist = torch.stack(target_token_dist).detach().cpu().numpy()

                            # plot attention maps
                            # y: layer index
                            # x: tokens (decoded text)
                            # Heat map
                            print(target_token_dist.shape)
                            col_labels = [self.tokenizer.decode(token_ids[i]) for i in range(len(token_ids))]

                            fig, ax = plt.subplots(figsize=(20, 10))
                            heatmap(target_token_dist, row_labels = [i for i in target_layers], col_labels =col_labels,
                                    ax = ax, cmap = "BuGn", cbarlabel = "Label")
                            # plt.colorbar()
                            plt.xlabel('Tokens')
                            plt.ylabel('Layers')
                            # plt.xticks(range(len(target_token_dist[0])), range(len(target_token_dist[0])))
                            # plt.yticks(range(len(target_token_dist)), range(len(target_token_dist)))

                            # add text to the below of the plot (xtick position)
                            # for i in range(len(target_token_dist[0])):
                            #     plt.text(i, -1, self.tokenizer.decode(token_ids[i]), ha='center', va='bottom', rotation=45)

                            plt.savefig(f'./attns/selected_tokens_{self.dataset}_{item_idx}_{self.iter}.pdf')
                            plt.close()

                            item_idx += 1
                    except Exception as e:
                        print(e)
                        pass
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs):
    
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.dataset in ['stanfordnlp/imdb', '20news', 'mrpc', 'piqa', 'sqa', 'csqa', 'obqa','arc_easy', 'arc_chall', 'winogrande', 'bbc', 'ledgar']:
            glue_metric = load('glue', 'sst2')
        else:
            glue_metric = load('glue', self.dataset)

        # self.check_act_nums(eval_dataset=eval_dataset)
        self.check_selected_tokens(eval_dataset=eval_dataset)

        total_acc = 0
        total_count = 0
        refs, preds = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
                output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                logits = output.logits
                preds = torch.argmax(logits, dim=1)
                total_acc += torch.sum(preds == batch['labels']).item()
                total_count += len(batch['labels'])
        acc = total_acc / total_count

        self.best_acc = max(acc, self.best_acc)
        results = {'acc':  total_acc / total_count, 'best acc': self.best_acc, 'task': self.dataset}
        self.log(results)
        
        self.cumulative_accs.append(acc)
        # write files 
        with open(f'./figs/{self.dataset}_{self.method}_accs.txt', 'w') as f:
            for acc in self.cumulative_accs:
                f.write(f'{acc}\t')
            f.write('\n')
        # self.analysis(eval_dataset=eval_dataset, acc=acc)
        return results
    