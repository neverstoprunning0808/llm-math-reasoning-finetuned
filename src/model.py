import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import load_config, AppConfig
from tqdm import tqdm


class LLMModel:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_checkpoint, )
        self.model = AutoModelForCausalLM.from_pretrained(config.model.model_checkpoint, dtype=torch.bfloat16).to(self.device)

    def format_cot_template(self, question) -> str:
        system_prompt = ("You are a math teacher at a grade school. You are helping a student to do a math homework."
                         "Show your reasoning step-by-step before providing the final answer."
                         "Be concise and simple. The final answer is a round-up integer."
                         "Please put the final answer after #### ")
        
        examples = [
            {
                "user": "Natalia sold clips to 48 of her friends in April, "\
                "and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "assistant": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
            },
            {
                "user": "Weng earns $12 an hour for babysitting. "
                "Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                "assistant": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
            },
            {
                "user": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
                "assistant": """
                        In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
                        Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
                        This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
                        #### 5
                            """
            }
        ]

        chat_template = []
        chat_template.append({"role": "system", "content": system_prompt})
        for e in examples:
            chat_template.append({"role": "user", "content": e["user"]})
            chat_template.append({"role": "assistant", "content": e["assistant"]})

        chat_template.append({"role": "user", "content": question}), 
    
        return self.tokenizer.apply_chat_template(chat_template, add_generation_prompt=True, tokenize=False)
    

    def generate(self, prompt: str, num_return_sequences: int | None = None, temperature: float | None = None) -> list[str] | list[list[str]]:
        return self.batch_generate([prompt], num_return_sequences, temperature)
    

    def batch_generate(self, prompts: list[str], num_return_sequences: int | None = None, temperature: float | None = None) -> list[str] | list[list[str]]:
        num_return_sequences = num_return_sequences or self.config.model.num_return_sequences
        temperature = temperature or self.config.model.temperature
        batch_size = self.config.model.batch_size
        
        results = []

        if len(prompts) > self.config.model.batch_size:
            for idx in tqdm(range(0, len(prompts), batch_size), desc = f"Runing on mini batches: {batch_size=}"):
                outs = self.batch_generate(prompts[idx:idx+batch_size], num_return_sequences, temperature)
                results.extend(outs)
            
            return results


        # tokenize the prompts
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            prompts,
            padding = True,
            return_tensors = "pt"
        ).to(self.device)

        # generate text
        outs = self.model.generate(**inputs,
                                   max_new_tokens=400,
                                   do_sample = temperature>0,
                                   temperature = temperature,
                                   num_return_sequences = num_return_sequences,
                                   eos_token_id = self.tokenizer.eos_token_id,
                                   pad_token_id = self.tokenizer.pad_token_id
                                   )
        
        new_text_start = inputs.input_ids.shape[1]

        trimmed_output = outs[:, new_text_start:]

        decoded_outs = self.tokenizer.batch_decode(trimmed_output, skip_special_tokens=True)

        if num_return_sequences==1:
            return decoded_outs
        else:
            return [decoded_outs[i:i+num_return_sequences] for i in range(0, len(decoded_outs), num_return_sequences)]


# def extract_answer(reasonings: list[str]) -> int:
#     results = []
#     for text in reasonings:
#         try:
#             value = int(text.split("####")[1].strip())
#         except(IndexError, ValueError):
#             value = float('inf')
#         results.append(value)
#     return results


# def accuracy(truth: list[int|float], preds: list[int|float]) -> float:
#     n = len(truth)
#     is_correct = sum((torch.tensor(truth) == torch.tensor(preds)).int()).item()

#     return is_correct / n


# if __name__ == "__main__":
#     config = load_config()
#     device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

#     model = LLMModel(config)

#     questions = ['Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?',
#                 'James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?']
    
    
#     prompt = [model.format_cot_template(q) for q in questions]


#     outs = model.batch_generate(prompt)

    # results = extract_answer(outs)

    # gold_truth = [60, 10]

    # print(outs)
    # print(results)
    # print(accuracy(results, gold_truth))


            




