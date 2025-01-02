import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse




def chat_template(question,feature):    
    user_prompt = question+"The response should have the following features: " + feature
    template =  [
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(template, 
                                              add_special_tokens=False,
                                              tokenize=False,
                                              add_generation_prompt=True)
    return user_prompt,input_str

def generate(model, tokenizer, input_str):
    input_ids = tokenizer(input_str, return_tensors="pt").to(device).input_ids
    output = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id, max_length=512)
    return tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)



def get_input_combinations(candidate_features, questions):
    input_str_combinations = []
    question_combinations = []
    feature_combinations = []
    for question in questions:
        for i in range(len(candidate_features)):
            for j in range(i, len(candidate_features)):
                if i == j:
                    feature_combinations.append(candidate_features[i])
                    user_prompt,input_str = chat_template(question, candidate_features[i])
                    question_combinations.append(user_prompt)
                    input_str_combinations.append(input_str)
                    continue           
                feature = candidate_features[i] + "," + candidate_features[j]
                feature_combinations.append(feature)
                user_prompt, input_str = chat_template(question, feature)
                question_combinations.append(user_prompt)
                input_str_combinations.append(input_str)
    return input_str_combinations, question_combinations, feature_combinations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', dest='model_name', default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--questions', dest='questions',type=list, default=["What is capital of France?","How to make a cake?","What is the No.1 college in China?","Why we can't drink too much?","Tell us about President Washington's life"])
    parser.add_argument('--styles', dest='styles', type=list, default=["helpful","talk nonsense","answer in reverse order","use quotes every sentence","use dialogues","stutter", "use lots of emojis","humorous","brief","pretend to be Mr. Bean","only use French to answer", "very angry","add a Ah! to every sentence's end"])
    parser.add_argument('--seedindex', dest='seedindex', type=int, default=0, help='seedIndex')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Open the file in write mode
    input_str_combinations, question_combinations, feature_combinations = get_input_combinations(args.styles, args.questions)

    with open("llm_data/arm_suparm_relation.txt", "w") as file:
        # Write the index relationships to the file
        for idx, combination in enumerate(feature_combinations):
            elements = combination.split(',')
            indexes = [str(args.styles.index(element)) for element in elements]
            file.write(f"{idx}\t{','.join(indexes)}\n")
    print("Index relationships saved to arm_suparm_relation.txt")
    model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map=device, torch_dtype=torch.float16)
    print("Generating responses...")
    responses = []
    for input_str in tqdm(input_str_combinations):
        responses.append(generate(model, tokenizer, input_str))
    df = pd.DataFrame({"question": question_combinations, "response": responses})
    df.to_csv("llm_data/responses.csv", index=False)
