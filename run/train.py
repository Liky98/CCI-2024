import sys
import os
import argparse
import torch
from datasets import Dataset,concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType
from src.data import CustomDataset, DataCollatorForSupervisedDataset

os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(prog="train", description="welcome to overfittingzone")

#모델 로드 및 세이브 인자값
parser.add_argument("--model_id", type=str, required=True, help="model file path")
parser.add_argument("--save_local_dir_path", type=str, default="resource/results", help="model save path")
parser.add_argument("--push_to_hub",action="store_true")
parser.add_argument("--hub_model_name", type=str, help="huggingface model name, please upload at 'overfit-brothers/model_name' ")
parser.add_argument("--HF_TOKEN", required=True, type=str, help="your huggingface token")

parser.add_argument("--custom_template", default='default', type=str, help='custom question template')
parser.add_argument("--prompt_template", type=str, default="default", help="pre-defined prompt template type, check src/prompt.py")

#학습 인자값
parser.add_argument("--use_unsloth", action="store_true")
parser.add_argument("--use_peft", action="store_true")
parser.add_argument("--use_mora", action="store_true")
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--use_validation_data",action="store_true")
parser.add_argument("--use_additional_data",action="store_true")

parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
parser.add_argument("--epoch", type=int, default=5, help="training epoch")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--max_seq_length", default=1024, type=int)

args = parser.parse_args()
print(f"\033[94m{args}\033[0m")
def unsloth(args):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    
    max_seq_length = args.max_seq_length
    dtype = None 
    load_in_4bit = True 
    
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = args.model_id,
                                        max_seq_length = max_seq_length,
                                        dtype = dtype,
                                        load_in_4bit = load_in_4bit,
                                        )
    model = FastLanguageModel.get_peft_model(model,
                                            r = 8, 
                                            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                            "gate_proj", "up_proj", "down_proj",],
                                            lora_alpha = 16,
                                            lora_dropout = 0,
                                            bias = "none",
                                            use_gradient_checkpointing = "unsloth", 
                                            random_state = 3407,
                                            use_rslora = False, 
                                            loftq_config = None, 
                                        )

    prompt_template = args.prompt_template
    custom_template = args.custom_template
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<user>: ' if message['role'] == 'user' else '<assistant>: '}}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}"
        print("해당 모델은 Chat_template이 없기 때문에 Default로 설정됩니다.")
    
    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json",
                                    tokenizer,
                                    template = prompt_template,
                                    custom_template=custom_template)
    valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json",
                                    tokenizer,
                                    template = prompt_template,
                                    custom_template=custom_template)
    additional_dataset = CustomDataset("resource/data/additional_data_train.json",
                                    tokenizer,
                                    template = prompt_template,
                                    custom_template=custom_template)


    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    }).shuffle(seed=42)
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })
    additional_dataset = Dataset.from_dict({
        'input_ids': additional_dataset.inp,
        "labels": additional_dataset.label,
    })

    if args.use_validation_data:
        train_dataset = concatenate_datasets([train_dataset, valid_dataset]).shuffle(seed=42)
        if args.use_additional_data:
            train_dataset = concatenate_datasets([train_dataset,valid_dataset, additional_dataset]).shuffle(seed=42)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        data_collator=data_collator,
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            output_dir=args.save_local_dir_path,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio=0.01,
            num_train_epochs=args.epoch,
            learning_rate = args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
        ),
    )

    trainer_stats = trainer.train()
    
    return model, tokenizer
    
def default(args):
    bnb_config = None
    if 'gemma' in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto", 
            attn_implementation='eager'
        )
    else:
        if args.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
                model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                )
        elif args.load_in_8bit:
                bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
                model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=True)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<user>: ' if message['role'] == 'user' else '<assistant>: '}}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}"
        print("해당 모델은 Chat_template이 없기 때문에 Default로 설정됩니다.")
    tokenizer.pad_token = tokenizer.eos_token
    if args.use_peft == True:
        if 'glm' in args.model_id.lower():
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query_key_value"],
            )
            model = get_peft_model(model, peft_config)
            print(f"\033[93m ##Use LoRA## \033[0m")
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],            )
            model = get_peft_model(model, peft_config)
            print(f"\033[93m ##Use PEFT## \033[0m")
    elif args.use_mora==True:
        config = LoraConfig(
        use_mora=True,
        mora_type=1,

        r=128,

        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print(f"\033[93m ##Use MoRA## \033[0m")
    else: print(f"\033[93m ##Use SFT## \033[0m")

    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json",
                                tokenizer,
                                template = args.prompt_template,
                                custom_template=args.custom_template)
    valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json",
                                tokenizer,
                                template = args.prompt_template,
                                custom_template=args.custom_template)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })

    if args.use_validation_data:
        train_dataset = concatenate_datasets([train_dataset, valid_dataset])
        train_dataset = train_dataset.shuffle(seed=42)

    if args.use_additional_data:
        additional_dataset = CustomDataset("resource/data/additional_data_train.json", tokenizer, template = args.prompt_template)
        additional_dataset = Dataset.from_dict({
        'input_ids': additional_dataset.inp,
        "labels": additional_dataset.label,
        })
        
        train_dataset = concatenate_datasets([train_dataset, additional_dataset])
        
        train_dataset = train_dataset.shuffle(seed=42)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.save_local_dir_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=not args.use_validation_data,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
        packing=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=42,
    )

    if args.use_validation_data:
        print(f"\033[93m ##USING TRAIN+VALIDATION DATA FOR TRAIN ## \033[0m")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            args=training_args,
        )
    else:
        print(f"\033[93m ##USING ONLY TRAIN DATA FOR TRAIN ## \033[0m")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=data_collator,
            args=training_args,
        )

    trainer.train()

    return model, tokenizer

    
def main(args):
    if args.use_unsloth:
        print("## Using unsloth ##")
        model, tokenizer = unsloth(args)
    else:
        print("## Default Training ##")
        model, tokenizer = default(args)
    
    if args.push_to_hub:
        print("## Push to HUB ##")
        model.push_to_hub(args.hub_model_name, private=True, token=args.HF_TOKEN)
        tokenizer.push_to_hub(args.hub_model_name, private=True, token=args.HF_TOKEN)


if __name__ == "__main__":
    exit(main(args))