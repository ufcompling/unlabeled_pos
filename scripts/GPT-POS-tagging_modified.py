import openai
import asyncio
import random
from pathlib import Path
from tqdm import tqdm
import tiktoken
import os
import re
import datetime
import argparse
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import statistics

def parse_args():
    parser = argparse.ArgumentParser(description="Part of Speech Tagging using the OpenAI API.")
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14", help="Model to use (default: gpt-4.1-mini-2025-04-14)")
    parser.add_argument("--treebank", default="UD_Irish-IDT", help="Treebank to process (default: UD_Irish-IDT)")
    parser.add_argument("--output", default="output.txt", help="Output file suffix (default: output.txt)")
    parser.add_argument("--concurrency", type=int, default=500, help="Maximum concurrent API calls (default: 100)") ## lower to 5 if reaching rate limit; concurrency == 1 is the lowest; even if hitting rate limit, if you let the script run, it will retry the calls that fail
    parser.add_argument("--nshots", type=int, default=-1, help="Maximum number of examples of in-context learning (default: -1)")
    parser.add_argument("--selection", type=int, default=0, help="Dataset selection ID: 0, 1 or 2 (default: 0)")
    return parser.parse_args()

# Set the OpenAI API key
os.environ['OPENAI_KEY'] = 'YOUR OPENAI_KEY'
api_key = os.environ['OPENAI_KEY']
client = openai.AsyncOpenAI(api_key=api_key)

args = parse_args()
MODEL = args.model
TREEBANK = args.treebank
MAX_CONCURRENCY = args.concurrency
MAX_RETRIES = 5
N_SHOTS = args.nshots
SELECTION_ID = args.selection

TOKEN_MODEL = 'gpt-4o'

if MODEL.startswith('gpt-4.1-mini'):
    PRICE_PER_1M_INPUT_TOKENS = 0.4
    PRICE_PER_1M_OUTPUT_TOKENS = 1.6
else:
    PRICE_PER_1M_INPUT_TOKENS = 2.0
    PRICE_PER_1M_OUTPUT_TOKENS = 8.0

# Set language
language = "pos_data/" + TREEBANK + "/"
langname = re.sub("UD_", "", TREEBANK)
langname = re.sub("-.*?$", "", langname)
langname = re.sub("_", " ", langname)

print('LANGUAGE:', langname)
print('MODEL:', MODEL)

# Tokenizer
encoding = tiktoken.encoding_for_model(TOKEN_MODEL)

# Get annotated samples for in-context learning
shots_in = []
trainin = language + f"100/%d/random/500/select0/train.100.input" % (SELECTION_ID)
with open(trainin) as f:
    for line in f:
        tokens = line.split()
        shots_in.append('\n'.join(['%d\t%s'%(i+1, t) for i, t in enumerate(tokens)]))

shots_out = []
trainout = language + f"100/%d/random/500/select0/train.100.output" % (SELECTION_ID)
with open(trainout) as f:
    for line in f:
        shots_out.append(line.rstrip())

shots_tagged = []
for words, tags in zip(shots_in, shots_out):
    tagged_sent = '\n'.join(['%s\t%s'%(w, t) for w, t in zip(words.split('\n'), tags.split())])
    shots_tagged.append(tagged_sent)

if N_SHOTS >= 0:
    shots_tagged = shots_tagged[:N_SHOTS]

print("NUMBER OF TRAINING SAMPLES:", len(shots_tagged))

# Load test data
with open(language + "test.full.input") as f:
    lines = [line.strip() for line in f]

taglist = []
goldtags = []
with open(language + "test.full.output") as f:
    for line in f:
        taglist.extend(line.split())
        goldtags.append(line.split())
tags = list(set(taglist))

system_prompt = f'''You are a helpful assistant who is an expert part of speech tagger that works with the Universal Dependencies part-of-speech tagset. 
The user will give you an sentence in {langname} to be tagged, with one token per line, where each line contains the token's index and the token.
You must provide the tagged sentence in the following format:
1\ttoken_1\ttag_1 
2\ttoken_2\ttag_2
...
n\ttoken_n\ttag_n

You must use only the tags in the following tagset: {' '.join(tags)}.

For example, if the user says:
{shots_in[0]}

You say:
{shots_tagged[0]}

IMPORTANT: the sentence that the user provides has already been tokenized, and each sequence of characters separated by whitespace is a token. DO NOT further split the tokens, and DO NOT join tokens. Also, DO NOT change anything in the sentence provided.

Please provide only the tagged sentence, and nothing else. No explanation, no alternatives, only the tagger output in the format specified above.

Here are examples of tagged sentences that show how to perform this task:
'''

for sent, tagged_sent in zip(shots_in, shots_tagged):
    system_prompt += 'SENTENCE:\n' + sent + '\n\nTAGGED SENTENCE:\n' + tagged_sent + '\n\n'

system_prompt += 'Now the user will provide a sentence, and you must provide just the tagged sentence (don\'t start with "TAGGED SENTENCE:") as shown in the examples above.'



# Token counter
def count_tokens(*texts):
    return sum(len(encoding.encode(text)) for text in texts)

# Retry with exponential backoff
async def retry_with_backoff(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except openai.RateLimitError:
            delay = 2 ** attempt + random.uniform(0, 1)
            print(f"[429] Rate limit. Retry {attempt + 1} in {delay:.1f}s")
            await asyncio.sleep(delay)
        except openai.APIError as e:
            delay = 2 ** attempt + random.uniform(0, 1)
            print(f"[APIError] {e}. Retry {attempt + 1} in {delay:.1f}s")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"[ERROR] Unhandled: {e}")
            break
    return "<ERROR>"

# Process one sample
async def annotate_sample(index, sample, gold, semaphore):
    async with semaphore:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample}
        ]
        input_tokens = count_tokens(system_prompt, sample)

        async def call():
            return await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                top_p = 1,
                n = 1,
                frequency_penalty = 0,
                presence_penalty = 0)        

        response = await retry_with_backoff(call)

        if isinstance(response, str):  # failed after retries
            return index, response, input_tokens, 0

        output = response.choices[0].message.content.strip()
        output_lines = output.split('\n')
        output_tags = [line.split()[-1] for line in output_lines]
        num_tags = len(output_tags)
        
        if num_tags != len(gold):
            print("Sentence %d: MISMATCH" % (index))
            # If there are too many mimatches, we could try again here,
            # or try token by token (expensive)

        num_corr = sum(1 for a, b in zip(gold, output_tags) if a == b)

        output_tokens = response.usage.completion_tokens
        
        return index, output, input_tokens, output_tokens, num_corr, num_tags


async def main():
    inputs = ['\n'.join(['%d\t%s'%(i+1, t) for i, t in enumerate(line.split())]) for line in lines]
#    inputs = inputs[:5] # Output format: '1\ttoken_1\n2\ttoken_2\n...\nn\ttoken_n'
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [annotate_sample(i, sample, gold, semaphore) for i, (sample, gold) in enumerate(zip(inputs, goldtags))]
    
    results = [None] * len(inputs)
    total_input_tokens = 0
    total_output_tokens = 0
    total_tags = 0
#    total_corr = 0
    precision_scores = []
    recall_scores = []
    f1_scores = []

    with tqdm(total=len(tasks)) as pbar:
        for coro in asyncio.as_completed(tasks):
            index, output, in_tok, out_tok, num_corr, num_tags = await coro
            results[index] = output
            predicted_tags = [tok.split('\t')[-1] for tok in output.split('\n')]
            gold_tags = goldtags[index]
            if len(predicted_tags) < len(gold_tags):
                num_None = len(gold_tags) - len(predicted_tags)
                predicted_tags += ['None'] * num_None
            elif len(predicted_tags) > len(gold_tags):
                predicted_tags = predicted_tags[:len(gold_tags)]
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(gold_tags, predicted_tags, average='weighted', zero_division=0)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_tags += num_tags
    #        total_corr += num_corr
            pbar.update(1)
    
    average_precision = statistics.mean(precision_scores)
    average_recall = statistics.mean(recall_scores)
    average_f1 = statistics.mean(f1_scores)

    with open(TREEBANK + "_output.txt", "w", encoding="utf-8") as f:
        for output in results:
            f.write(output + "\n")
            f.write('\n')
        f.write(f"Average Precision: {average_precision:.2f}\n")
        f.write(f"Average Recall: {average_recall:.2f}\n")
        f.write(f"Average F1: {average_f1:.2f}\n")

    cost = ((total_input_tokens / 1000000) * PRICE_PER_1M_INPUT_TOKENS + (total_output_tokens / 1000000) * PRICE_PER_1M_OUTPUT_TOKENS)

    print()
    print(f"Tagged {len(results)} sentences")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Estimated cost: ${cost:.4f}")
    print()
    print(f"Total tags: {total_tags}")
    print(f"{langname}: Average Precision: {average_precision:.2f}, Average Recall: {average_recall:.2f}, Average F1: {average_f1:.2f}")
    
#    if total_tags > 0:
#        print(f"Accuracy: %0.3f" % ((total_corr / total_tags) * 100))

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    asyncio.run(main())
    print("Elapsed time:", datetime.datetime.now() - start_time)

