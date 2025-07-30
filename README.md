# LLM4Rail

LLM4Rail is a novel LLM-augmented railway service consulting platform. Empowered by LLM, LLM4Rail can provide custom modules for ticketing, railway food & drink recommendations, weather information, and chit-chat.

<!-- Illustration of LLM4Rail -->
<img src="asset/website_en2.png">

LLM4Rail utilizes the proposed iterative “Question-Thought-Action-Observation(QTAO)” prompting framework, which meticulously integrates verbal reasoning with task-oriented actions.

<!-- Illustration of QTAO -->
<div align="center"><img src="asset/illustration_of_ticket_inquiry.jpg" width=90% height=90%></div>

We also introduce the **C**hinese **R**ailway **F**ood and **D**rink ([CRFD-25](https://anonymous.4open.science/r/CRFD25)) dataset. Based on the CRFD-25, we develop a
novel LLM-based algorithm for zero-shot conversational recommendation. This approach leverages feature-based recommendation alignment to ensure all suggested items are grounded in the provided dataset.

<!-- Illustration of Food & Drink Recommendation -->
<div align="center"><img src="asset/recommendation.jpg" width=70% height=70%></div>


# 📌 Website

You can get started with LLM4Rail on its official website: http://111.170.34.203/

# 🚀 Requirements

- Python == 3.12
- openai == 1.97.0
- numpy == 1.26.4
- pandas  == 2.2.3
- levenshtein == 0.27.1
- scikit-learn == 1.6.1
- tqdm == 4.67.1
- transformers == 4.49.0

# 🌟 Evaluation

## 1. Set Working Directory
Set the working directory to the project root.
```bash
cd <your project path>
```

## 2. Set Environment Variable
Set PYTHONPATH environment variable to the project's root directory.
```bash
export PYTHONPATH=<your project path>
```

## 3. Generate Test Queries
Run the script `evaluation/<task>/query_generator.py`.
```bash
python evaluation/<task>/query_generator.py
```
`<task>`: `weather` or `ticket` <br>
If you want to evaluate recommendation module, skip this step.

## 4. Evaluate Weather/Ticket Inquiry Module 🌧🚅
Run the script `evaluation/<task>/evaluate.py`.
```bash
python evaluation/<task>/evaluate.py \
--model <model> \
--userFile dataset/user.csv \
--config evaluation/<task>/config.json \
--saveDir evaluation/<task>/save/ \
--GPTKey <your GPT key> \
-QwenKey <your Qwen key> \
-amapKey <your amap Key>
```
`<model>`: `qwen-3` or `gpt-4o` <br>

## 5. Evaluate Food & Drink Recommendation Module 🍔
Run the script `evaluation/meal/evaluate.py`.
```bash
python evaluation/meal/evaluate.py \
--model <model> \
--rec_model <rec_model> \
--topk <k> \
--config evaluation/meal/config.json \
--saveDir evaluation/meal/save/ \
--GPTKey <your GPT key> \
-QwenKey <your Qwen key> \
```
`<model>`: `qwen-3` or `gpt-4o` <br>
`<rec_model>`: `zero-shot` or `feature-augmented` <br>

## 6. Analysis The Results
Run the script `evaluation/<task>/analysis/analysis.py`. <br>
```bash
python evaluation/<task>/analysis/analysis.py --saveDir evaluation/<task>/save [--rec_model <rec_model>]
```
`<task>`: `weather`, `ticket` or `meal` <br>
`<rec_model>`: `zero-shot` or `feature-augmented` (only needed when `<task>`=`meal`) <br>
