from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
    )

from datasets import load_dataset
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


checkpoint = "huggyllama/Llama-7b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, 
#                                             quantization_config=bnb_config,
                                             torch_dtype=torch.float16,
                                             device_map="auto")


# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)
# model.print_trainable_parameters()


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
dataset = dataset.map(lambda example: tokenizer(example["prompt"], max_length=256, truncation=True), batched=True)
dataset = dataset["train"].train_test_split(0.1, 0.9)
tokenizer.pad_token_id = tokenizer.eos_token_id

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="llama",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=500,
    warmup_steps=2,
    logging_steps=20,
    save_steps=2000,
    learning_rate=2e-4,
    fp16=True,
    #optim="paged_adamw_8bit",
    ddp_find_unused_parameters=False,
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# merged_model = model.merge_and_unload()

# trainer.model.save_pretrained("rugpt_merged_model")

model.push_to_hub("lab4_code", token="xxx")