This is a repo featuring GenVR's Hinglish-LLM v1.0

To run this model, install PEFT library and then use following code:-

*********************************************************************************
```
from peft import PeftModel 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  #  same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    token=<hf_token>
)

ft_model = PeftModel.from_pretrained(model, <model directory>) 
model.config.use_cache = True 
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
    #padding_side="left",
    token=<hf_token>
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def answer(text):
  eval_prompt = f"""< s>[INST] << SYS>>
  Please be accurate and translate the given English sentence into Hinglish. Return only the translated output and nothing else.
  << /SYS>>
  {text} [/INST]\n"""
  model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

  ft_model.eval()
  with torch.no_grad():
      print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=200)[0], skip_special_tokens=True))

answer('That girl is my friend')
```
***************************************************************************************
> ye ladki meri dost hai
***************************************************************************************

To run this, you would need a token from huggingface for Llama2, as this is based on Llama2 finetune. Which you can get in below link:-

https://huggingface.co/blog/llama2

And model weights which can be downloaded from below drive link and have to be put in the same directory as rest of the repo :

https://drive.google.com/drive/folders/1Fy6Ba1OCcyeexmBAoKMV3xtTWXy5ryWu?usp=sharing

Please run: pip install -r requirements.txt
to install all dependencies.
***************************************************************************************
Here is a demo video, comparing our work with AI4Bharat-IndicTrans2

https://www.youtube.com/watch?v=wxDQSEZIqfo

Published by Tech News.

Strengths of GenVR-Hindi-LLM v1.0:-  The translated text more commonly resonates with local speakers who tend to use a hinglish kind of dialect.
Weaknesses of GenVR-Hindi-LLM v1.0:- The translated text has some halucinations.
