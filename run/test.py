import argparse
import json
import tqdm
import os
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from src.data import CustomDataset

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--custom_template", default='default', type=str, help='custom question template')
g.add_argument("--prompt_template", type=str, default="default", help="pre-defined prompt template type, check src/prompt.py")
g.add_argument("--load_in_4bit",action="store_true")
g.add_argument("--load_in_8bit",action="store_true")


def main(args):
    try:
        peft_config = PeftConfig.from_pretrained(args.model_id)
        is_lora_model = True
    except:
        is_lora_model = False
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
        
    if is_lora_model:
        print("Loading LoRa-Trained Model")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(base_model, args.model_id)
    else:
        print("Loading Normal Model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=True)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<user>: ' if message['role'] == 'user' else '<assistant>: '}}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}"
        print("해당 모델은 Chat_template이 없기 때문에 Default로 설정됩니다.")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = CustomDataset("resource/data/대화맥락추론_test.json",
                            tokenizer,
                            template = args.prompt_template,
                            custom_template=args.custom_template)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _ = dataset[idx]
        outputs = model(
            inp.to(args.device).unsqueeze(0)
        )
        logits = outputs.logits[:,-1].flatten()

        if 'glm' in args.model_id.lower():
            vocab = tokenizer.get_vocab()
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[vocab[b'A']],
                            logits[vocab[b'B']],
                            logits[vocab[b'C']],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
        else:
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer.vocab['A']],
                            logits[tokenizer.vocab['B']],
                            logits[tokenizer.vocab['C']],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

        result[idx]["output"] = answer_dict[numpy.argmax(probs)]

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    exit(main(parser.parse_args()))