import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
st.title("ChatGPT-like clone with SAWYER")

STOP_TOKEN = '###STOP###'
HUMAN_TOKEN = '###HUMAN###'
BOT_TOKEN = '###BOT###'

EXTRA_TOKENS = {
    'stop_token': {
        'token': STOP_TOKEN,
        'replace_embedding_with': 'stop talking'
    },
    'human_token': {
        'token': HUMAN_TOKEN,
        'replace_embedding_with': 'The human said:'
    },
    'bot_token': {
        'token': BOT_TOKEN,
        'replace_embedding_with': 'The assistant said:'
    }
}

@st.cache_resource
def load_model_and_tokenizer():
    username, repo_name = 'profoz', 'sawyer-llama-2'

    tokenizer = AutoTokenizer.from_pretrained(f"{username}/{repo_name}")

    base_model = 'NousResearch/Llama-2-7b-hf'

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    print('Downloading..')

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0}
    )

    model.resize_token_embeddings(len(tokenizer))

    for extra_token, extra_info in EXTRA_TOKENS.items():
        token_id = tokenizer(extra_info['token'])['input_ids'][-1]
        new_embedding = model.model.embed_tokens.weight.data[tokenizer(extra_info['replace_embedding_with'])['input_ids'][1:]].mean(dim=0, keepdim=True)#.reshape(-1)
        EXTRA_TOKENS[extra_token]['new_embedding'] = new_embedding
        model.model.embed_tokens.weight.data[token_id] = EXTRA_TOKENS[extra_token]['new_embedding'].clone()
        EXTRA_TOKENS[extra_token]['token_id'] = token_id
        print(f"Replaced token \"{extra_info['token']}\" (token id {token_id}) weight with weight for \"{extra_info['replace_embedding_with']}\"")

    if 'stop_id' not in st.session_state:
        st.session_state.stop_id = EXTRA_TOKENS['stop_token']['token_id']

    rlf_model = PeftModel.from_pretrained(model, 'profoz/sawyer-llama-rlf').eval()

    return tokenizer, rlf_model

def join_convo(conversation):
    convo = ''''''
    last_speaker = None
    for speaker, message in conversation:
        last_speaker = speaker
        if speaker == 'human':
            convo += f"{EXTRA_TOKENS['human_token']['token']} {message} "
        elif speaker == 'assistant':
            convo += f"{EXTRA_TOKENS['bot_token']['token']} {message} "
    if last_speaker == 'human':
        return convo.strip() + f" {EXTRA_TOKENS['bot_token']['token']}"
    return convo.strip() + f" {EXTRA_TOKENS['stop_token']['token']}"
#
# join_convo(
#     [
#         ('human', 'hi'),
#         ('assistant', 'sup')
#     ]
# )

def query_sawyer(conversation):
    print('conversation::', conversation)
    prompt = join_convo(conversation)
    print('prompt::', prompt)
    tokenizer, rlf_model = load_model_and_tokenizer()

    output = tokenizer.decode(rlf_model.generate(
        tokenizer(prompt, return_tensors='pt')['input_ids'],
        eos_token_id=st.session_state.stop_id,
        do_sample=True,
        num_return_sequences=1,
        max_new_tokens=128
    )[0])

    return output.split(EXTRA_TOKENS['bot_token'])[-1].split(EXTRA_TOKENS['stop_token'])[0].strip()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_sawyer(
            conversation=[
                (m["role"], m["content"])
                for m in st.session_state.messages
            ]
        )
        stream = st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
