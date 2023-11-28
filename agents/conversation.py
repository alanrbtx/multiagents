from .agent import LLMAgent

class TwoAgentsCoversation:
    def __init__(
            self,
            agent1: LLMAgent,
            agent2: LLMAgent,
            max_new_messages: int = 4
    ):
        
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_new_messages = max_new_messages
        self.context = list()
    
    
    def start_conversation(self, user_prompt):
        system_prompt = (
            "Чат между двумя агентами"
            "Вы должны решить одну задачу"
            "ENGINEER: ты генерируешь код шаг за щагом"
            "CRITIC: ты определяешь насколько правильный сгенерирован код"
            "Если вы считаете, что задача решена сгенерируйте SOLVED"
        )

        self.context.append(system_prompt + user_prompt)
        for i in range(self.max_new_messages + 1):
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
            
