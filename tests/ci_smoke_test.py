import argparse, time
from llama_cpp import Llama

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
args = p.parse_args()

print("Loading model:", args.model)
t0 = time.time()
llm = Llama(model_path=args.model, embedding=True)
print("Loaded in", time.time()-t0, "s")

# embedding test
emb = llm.create_embedding("Hello CI test")
print("Embedding length:", len(emb['data'][0]['embedding']))

# small generate
resp = llm.create(prompt="Tell me the word 'CI test' in one sentence.", max_tokens=40, temperature=0.0)
print("Generation:", resp.get("choices", [{}])[0].get("text","").strip())
