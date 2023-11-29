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
       
        self.args = args
        self._build_agent()
        # self._build_model()
        self._build_tokenizer()

    def _build_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        peft_config = PeftConfig.from_pretrained("AlanRobotics/lab4_code")
        self.model = PeftModel.from_pretrained(self.model, peft_config)


    def _build_agent(self):
        if self.args.quantized == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name, 
                quantization_config=bnb_config,
                device_map="auto"
            )
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        if self.args.agent_type == "code":
            print("TYPE: CODE")
            peft_config = PeftConfig.from_pretrained("AlanRobotics/lab4_code")
            self.model = PeftModel.from_pretrained(self.model, "AlanRobotics/lab4_code")
        else:
            print("TYPE: CHAT")
            peft_config = PeftConfig.from_pretrained("AlanRobotics/lab4_chat")
            self.model = PeftModel.from_pretrained(self.model, "AlanRobotics/lab4_chat")

        
        
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
            max_new_tokens=max_new_tokens,
            num_beams=num_return_sequences
            
        )

        decoded_tokens = self.tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
        )

        return decoded_tokens