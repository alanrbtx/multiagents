from collections import namedtuple
from transformers import BitsAndBytesConfig
from dataclasses import dataclass


@dataclass
class AgentConfig:
    def __init__(
            self,
            model_name,
            agent_type,
            quantized
    ):
        
        self.model_name = model_name
        self.agent_type = agent_type
        self.quantized = quantized