import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
from collections import namedtuple
import enum


class LLMAgent(pl.LightningModule):
    """
    LLMAgent class create agent with specific PEFT config
    """

    def __init__(self, args):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
       
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        self._build_agent()
        self._build_tokenizer()

    def _build_agent(self):
        # Можно прикреплять разные адаптеры
        if self.args.agent_type == "code":
            peft_config = PeftConfig.from_pretrained("AlanRobotics/lab4_code")
            self.peft_model = PeftModel(self.model, peft_config)
        else:
            peft_config = PeftConfig.from_pretrained("AlanRobotics/lab4_chat")
            self.peft_model = PeftModel(self.model,peft_config)
        
    def _build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def sample(
            self,
            prompt,
            num_return_sequences=1,
            temp=0.1,
            max_new_tokens=128,
            max_length=1024
        ):
        
        input_ids = self.tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids

        tokens = self.model.generate(
            input_ids=input_ids,
            num_beams=num_return_sequences,
            temperature=temp,
            
        )[-max_new_tokens:]

        decoded_tokens = self.tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
        )

        return decoded_tokens