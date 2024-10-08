
import json

import torch
from torch.utils.data import Dataset
from src.prompt import Prompts

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, template, custom_template='default'):
        IGNORE_INDEX=-100
        self.inp = []
        self.trg = []
        self.label = []
        
        PROMPT = Prompts.get_prompt(template)
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def question_template(inp,custom_template):
            if custom_template=='default':
                question = f"[Question]\n위 대화의 {inp['category']}"
                if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                    question += "으로"
                else:
                    question += "로"
                question += " 올바른 지문은?"
                return question
            
            elif custom_template=='skip_category':
                question = f"[Question]\n"
                question += "위 대화에서 알 수 있는 사실로 올바른 것은?"
                return question
            
            elif custom_template=='chain_of_thought':
                question = f"[Question]\n"
                question += "다음 단계를 따라 문제를 해결해 주세요:\n"
                question += "1. 대화의 주요 내용을 간단히 요약하세요.\n"
                question += "2. 대화에서 언급된 중요한 사실들을 나열하세요.\n"
                question += "3. 각 사실의 신뢰성과 관련성을 평가하세요.\n"
                question += "4. 가장 신뢰할 수 있고 관련성 높은 사실을 선택하세요.\n"
                question += "5. 선택한 사실이 왜 가장 적절한 답변인지 설명하세요.\n"
                question += "위의 단계를 따라 분석한 후, 최종적으로 다음 질문에 답하세요: 위 대화에서 알 수 있는 사실로 가장 올바른 것은 무엇인가요?"
                return question

            else :
                raise KeyError

        def make_chat(inp, custom_template):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = question_template(inp, custom_template)
                
            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"], custom_template)
            
            if PROMPT is not None:
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat},
                ]
        
                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else : #no system prompt or use pre-defined system template
                message = [
                    {"role": "user", "content": chat},
                ]
        
                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
            else:
                target = ""
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
