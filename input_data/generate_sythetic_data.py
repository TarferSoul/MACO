import itertools
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--styles', dest='styles',type=list, default=["Empathy", "helpful","humorous","creative","high quality", "Philosophical depth","Professionalism","Brevity","Curiosity"])
    args = parser.parse_args()
    key_word_set = args.styles

    # Generate combinations of different lengths and store them in a single list
    all_combinations = []
    for length in range(1, len(key_word_set) + 1):
        combinations = list(itertools.combinations(key_word_set, length))
        for combo in combinations:
            all_combinations.append(','.join(combo))
    with open("synthetic_data/arm_suparm_relation.txt", "w") as file:
    # Write the index relationships to the file
        for idx, combination in enumerate(all_combinations):
            elements = combination.split(',')
            indexes = [str(key_word_set.index(element)) for element in elements]
            file.write(f"{idx}\t{','.join(indexes)}\n")
    print("Index relationships saved to synthetic_data/arm_suparm_relation.txt")
    df = pd.DataFrame({"combination": all_combinations})
    df.to_csv('synthetic_data/all_combinations.csv',index=False)
    print("All combinations saved to synthetic_data/all_combinations.csv")