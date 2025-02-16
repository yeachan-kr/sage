
import numpy as np
from tqdm.auto import tqdm

import torch
# from data import QADataset


# from modeling_opt_qst import QSTOPTForCausalLM, OPTForCausalLM
from transformers import Seq2SeqTrainer
import evaluate
from datasets import load_dataset

IGNORE_INDEX = -100

class AlpacaSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, dataset, max_input_len, max_output_len, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tokenizer = tokenizer

        self.mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/five_shot_mmlu_val.json',
            'test': 'data/mmlu/five_shot_mmlu_test.json',
        })

        self.mmlu_dataset = self.mmlu_dataset['eval']			
        self.abcd_idx = [
                    tokenizer("A", add_special_tokens=False).input_ids[0],
                    tokenizer("B", add_special_tokens=False).input_ids[0],
                    tokenizer("C", add_special_tokens=False).input_ids[0],
                    tokenizer("D", add_special_tokens=False).input_ids[0],
                ]
        self.accuracy = evaluate.load("accuracy")


    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs):

        eval_dataloader = self.get_eval_dataloader(self.mmlu_dataset)
        source_max_len = self.data_collator.source_max_len
        self.data_collator.source_max_len = 2048 # mmlu dataset has long inputs
        self.model.eval()
        preds, refs = [], []
        loss_mmlu = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                output = self.model(input_ids=batch['input_ids'], 
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
                loss = output.loss
                logits = output.logits
                labels = batch['labels']
                for i, logit in enumerate(logits):
                    label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                    logit_abcd = logit[label_non_zero_id - 1][self.abcd_idx]
                    preds.append(torch.argmax(logit_abcd).item())
                labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                refs += [self.abcd_idx.index(label) for label in labels.tolist()]
                loss_mmlu += loss.item()

        # Extract results by subject.
        results = {}
        subject = self.mmlu_dataset['subject']
        subjects = {s: {'refs': [], 'preds': []} for s in set(subject)}
        for s, p, r in zip(subject, preds, refs):
            subjects[s]['preds'].append(p)
            subjects[s]['refs'].append(r)
        subject_scores = []
        for subject in subjects:
            subject_score = self.accuracy.compute(
                references=subjects[subject]['refs'],
                predictions=subjects[subject]['preds']
            )['accuracy']
            results[f'mmlu_eval_accuracy_{subject}'] = subject_score
            subject_scores.append(subject_score)
        results['mmlu_eval_accuracy'] = np.mean(subject_scores)
        self.log(results)
        self.data_collator.source_max_len = source_max_len
        return results

        # glue_metric = load('glue', 'sst2')
        # refs, preds = [], []
        # for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
        #     # print(batch['input_ids'].size())
        #     # exit()
        #     # print(self.tokenizer.decode(batch['input_ids'][0]))
        #     batch['input_ids'] = batch['input_ids'][:, :self.max_input_len]
        #     batch['attention_mask'] = batch['attention_mask'][:, :self.max_input_len]

        #     # print(self.tokenizer.decode(batch['input_ids'][0]))
        #     # exit()


        #     output = self.model.generate(input_ids=batch['input_ids'],
        #                                     attention_mask=batch['attention_mask'],
        #                                     max_length=int(self.max_input_len + 1))
        #     p = list(self.tokenizer.batch_decode(output[:, batch['input_ids'].size(1):], skip_special_tokens=True))
        #     t = list(self.tokenizer.batch_decode(batch['labels'][:, batch['input_ids'].size(1):], skip_special_tokens=True))
        #     p = [x.strip() for x in p]

        #     preds.extend(p)
        #     refs.extend(t)

        #     print(p[:5])
        #     print(t[:5])

        # answer_list = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # for i, (pred, ref) in enumerate(zip(preds, refs)):
        #     if pred in answer_list:
        #         preds[i] = answer_list[pred]
        #     else:
        #         preds[i] = -1
        #     if ref in answer_list:
        #         refs[i] = answer_list[ref]
            
        # results = glue_metric.compute(predictions=preds, references=refs)
        # self.log(results)
        # return results
    