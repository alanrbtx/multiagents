from .agent import LLMAgent

class TwoAgentsCoversation:
    def __init__(
            self,
            agent1: LLMAgent,
            agent2: LLMAgent,
            max_rounds: int = 2
    ):
        
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_rounds = max_rounds
        self.context = list()
    
    
    def start_conversation(self, user_prompt):
        system_prompt = (
            "Чат между двумя агентами"
            "ENGINEER: ты генерируешь код шаг за щагом"
            "CRITIC: ты определяешь насколько правильно сгенерирован код"
            "CRITIC, если ты считаешь, что задача решена правильно сгенерируй SOLVED"
            "Задача: "
        )

        self.context.append(system_prompt + user_prompt)
        for i in range(self.max_rounds):
            joined_context = "\n".join(self.context)
            agent1_responce = self.agent1.sample(joined_context + "\n" + "CRITIC:")
            self.context.append(agent1_responce)
            agent2_responce = self.agent2.sample(joined_context + "\n" + agent1_responce + "\n" + "ENGINEER:")
            self.context.append(agent2_responce)
            if "SOLVED" in joined_context:
                break
        print(agent2_responce)
    

    def show_history(self):
        return f"CONVERSATION HISTORY: {self.context}"   
            
