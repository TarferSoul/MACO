<h1 align="center"> MACO </h1>
Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification
<br>   <br>
<div align="center">
  <img src="pic/design (1).jpg" alt="Logo" width="750">
</div>

# File Structure
```
MACO/
    ├── algorithms/
    │   ├── MACO.py
    │   ├── config_parameters.py
    ├── input_data/
    │   ├── generate_llm_data.py
    │   ├── get_embed_syn_openai.py
    │   ├── get_embed_llm_openai.py
    │   ├── get_embed_llm_google.py
    │   ├── generate_sythetic_data.py
    │   ├── get_embed_syn_google.py
    ├── env/
    │   ├── SupArm.py
    │   ├── User.py
    │   ├── Arm.py
    ├── utils/
    │   ├── utils.py
    │   ├── normalize_dataset.py
    │   ├── compute_spanner.py
    ├── pic/
    │   ├── design.pdf
    ├── scripts/
    │   ├── run_syn_comp_diff_agents.sh
    │   ├── run_syn_diff_poolsize.sh
    │   ├── run_llm_diff_agents.sh
    ├── README.md
    ├── LICENSE
    ├── main.py
    ├── simulateExp.py
    ├── .gitignore
```
# How to use our code
## Step 1. Generate Data
### Generate synthetic scenarios
```bash
cd input_data
mkdir synthetic_data
python generate_synthetic_data.py
```
### Generate real LLM's responses scenarios
```bash
cd input_data
mkdir llm_data
python generate_llm_data.py
```
Then you can get to file `response.csv`, `all_combinations.csv` and `arm_suparm_relation.txt` in each dir
### Get 2 Scenarios's Embedding
#### Emebedding 1: OpenAI text-embedding-3-large
For real LLM's responses scenarios:
```bash
cd input_data 
mkdir OpenAI_llm
cp llm_data/arm_suparm_relation.txt OpenAI_llm
python get_embed_llm_openai.py --OPENAI_API_KEY <YOUR_API_KEY>
```
For synthetic scenarios:
```bash
cd input_data 
mkdir OpenAI_syn
cp synthetic_data/arm_suparm_relation.txt OpenAI_syn
python get_embed_syn_openai.py --OPENAI_API_KEY <YOUR_API_KEY>
```
#### Embedding 2: Google Gecko
For Googel Gecko, you may need to use Google Cloud Vertex-ai to access it.

For real LLM's responses scenarios:
```bash
cd input_data 
mkdir Google_llm
cp llm_data/arm_suparm_relation.txt Google_llm
python get_embed_llm_googel.py  --google_cloud_projectid <YOUR_PROJECT_ID> --google_cloud_location <YOUR_CLOUD_LOCATION> 
```
For synthetic scenarios:
```bash
cd input_data 
mkdir Google_syn
cp synthetic_data/arm_suparm_relation.txt Google_syn
python get_embed_llm_googel.py  --google_cloud_projectid <YOUR_PROJECT_ID> --google_cloud_location <YOUR_CLOUD_LOCATION> 
```

## Step 2. Run Experiment
### Synthetic scenarios for different poolsizes
```bash
bash scripts/run_syn_diff_poolsize.sh
```
### Real LLM's scenarios for different numbers of agents
```bash
bash scripts/run_syn_diff_agents.sh
```
### Compare regrets for different numbers of agents
```bash
bash scripts/run_syn_comp_diff_agents.sh
```