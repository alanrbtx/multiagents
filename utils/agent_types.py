import enum


class AgentTaskType(str, enum.Enum):
    
    CHAT_COMPLETION = "chat"
    CODE_COMPLETION = "code"