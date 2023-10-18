# -*- encoding:utf-8 -*-

"""
API REQUEST PARALLEL PROCESSOR
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import tqdm
import glob
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
from queue import PriorityQueue  

import random
import numpy as np

import copy


request_header = {"Content-Type": "application/json"}
request_url = {"url": "https://api.openai.com/v1/engines/davinci/completions"}

random.seed(42)
np.random.seed(42)

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    apis_pool: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_sleep_each_loop = 0.01  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    # api_endpoint = api_endpoint_from_url(request_url)
    api_endpoint = "chat/completions"
    # request_header = {"Authorization": f"Bearer {api-key}"}
    current_api_record = apis_pool.get()
    api = current_api_record.api
    request_header.update({'api-key': api['api-key']})
    request_url.update({'url': api['request_url']})

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    status_tracker.current_api_key = api
    status_tracker.current_api_available_time = - current_api_record.current_time
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath, "r", encoding="utf8") as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    task_id = next_request.task_id
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        token_consumption = num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name)
                        if token_consumption >= 4096:
                            token_consumption = 4096
                        task_id = next(task_id_generator)
                        next_request = APIRequest(
                            task_id=task_id,
                            request_json=request_json,
                            token_consumption=token_consumption,
                            attempts_left=max_attempts,
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                            apis_pool=apis_pool,
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                logging.warning(
                    """
                    **************************************
                    All tasks finished. Breaking main loop.
                    **************************************
                    """)
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down


        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    time_of_last_switch_to_new_api_key: int = 0  # used to switch API keys periodically

class API():
    def __init__(self, current_time, api):
        self.current_time = current_time
        self.api = api

    def __lt__(self, other):
        """ API with smaller current_time is considered smaller """
        if self.current_time == other.current_time:
            if random.random() > 0.5:
                return True
            else:
                return False
        return self.current_time < other.current_time


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_api(
        self,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        apis_pool: PriorityQueue = None,
        seconds_to_switch_api_key: int = 5,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                json_data = self.request_json['api_input']
                logger_flag = True
                available_time = status_tracker.current_api_available_time
                while True:
                    apis_pool.put(API(available_time, status_tracker.current_api_key))
                    next_api_record = apis_pool.get()
                    status_tracker.current_api_key = next_api_record.api
                    status_tracker.current_api_available_time = next_api_record.current_time

                    if status_tracker.current_api_available_time > time.time():
                        await asyncio.sleep(0.1)
                        if logger_flag:
                            logging.warning(f"API key {status_tracker.current_api_key['api-key']} is paused to cool down for {round(status_tracker.current_api_available_time - time.time(), 2)}s")
                            logger_flag = False
                        available_time = status_tracker.current_api_available_time
                    else:
                        request_header.update({'api-key': status_tracker.current_api_key['api-key']})
                        request_url.update({'url': status_tracker.current_api_key['request_url']})
                        # if status_tracker.current_api_key != old_api_key:
                        logging.warning(f"Request {self.task_id} is switched to API key {status_tracker.current_api_key['api-key']} to start.")
                        break

                async with session.post(
                    url=request_url['url'], headers=request_header, json=json_data
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    error_message = response["error"]["message"]
                    retry_seconds = int(re.search(r"Please retry after (\d+) seconds", error_message)[1])
                    retry_time = time.time() + retry_seconds + 1
                    status_tracker.current_api_available_time = retry_time

                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.task_id} failed after all attempts.")
                # append_to_jsonl([self.request_json, self.result], save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            dialogue = post_process_gpt3_response(response)
            if dialogue is not None:
                output = {
                    "rounds": self.request_json['rounds'],
                    "word_counts": self.request_json['word_counts'],
                    # "settings": self.request_json['settings'],
                    "dialogue": dialogue,
                    "title": self.request_json['title'] if self.request_json.get('title') else '',
                    "reference": self.request_json['reference'],
                    "prompt": self.request_json['prompt'],
                    "meta": self.request_json['meta'],
                }
                append_to_jsonl(output, save_filepath)
                
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1
                logging.debug(f"Request {self.task_id} saved to {save_filepath}")
            else:
                if self.attempts_left:
                    retry_queue.put_nowait(self)
                else:
                    logging.error(f"Request {self.task_id} failed after all attempts.")
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")

def write_to_json(data, filename: str) -> None:
    """Write a json payload to a json file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "w", encoding="utf8") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        request_json = request_json["api_input"]
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant

            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def encode_prompt(context, rounds=None, word_counts=None, language="zh"):
    if language == "zh":
        system_input =  "要求你作为聊天机器人Assistant与人类Human进行多轮对话。对话是根据##提供信息##的内容开展的，并以#对话规划#的格式进行输出，以<start_chat>开始，以<end_chat>结束。"
    else:
        system_input = "You are asked to chat with a human as a chatbot Assistant in multiple rounds. The dialogue is based on the ##Provided Information## and is output in the format of #Conversation Plan#, starting with <start_chat> and ending with <end_chat>."

    if rounds is None and word_counts is None:
        selected_round = [2, 3, 4, 5]
        rounds = random.choices(selected_round, weights=[0.0, 0.5, 0.3, 0.2])[0]
        word_counts = [300] * rounds
    
    if rounds is None and word_counts is not None:
        rounds = len(word_counts)


    user_input = ""
    chat_format = ""
    chat_format += "<start_chat>"

    local_settings = {
        "zh": {
            'settings': [
                (["以小孩子语气提问", "以小孩听得懂方式回答"], 0.1),
                (["以年轻人语气提问", "回答[+详细解释]"], 0.1),
                (["以老年人语气提问", "回答[+详细解释]"], 0.1),
                (["以专家语气提问", "回答[+详细解释]"], 0.1),
                (["提出要求", "回答[+详细解释]"], 0.5),
                (["提出问题", "回答[+详细解释]"], 0.5),
                (["提出问题/要求","Assistant用小孩字能听得懂、通俗方式回答"], 0.3),
                (["怀有好奇心提问", "回答[+详细解释]"], 0.5),
                (["以生活实际出发来提问", "回答[+详细解释]"], 0.5),
                (["向Assistant下达具体指令", "回答[+详细解释]"], 0.5),
                (["表达自己的需求并要求Assistant帮助", "回答[+详细解释]"], 0.5),
                ],
        },             
        "en": {
            'settings': [
                (["asks in a childlike tone", "answers in a way that a child can understand"], 0.1),
                (["asks in a young person's tone", "answers [+detailed explanation]"], 0.1),
                (["asks in an old person's tone", "answers [+detailed explanation]"], 0.1),
                (["asks in an expert's tone", "answers [+detailed explanation]"], 0.1),
                (["makes a request", "answers [+detailed explanation]"], 0.5),
                (["asks a question", "answers [+detailed explanation]"], 0.5),
                (["asks a question/request", "answers in a way that a child can understand"], 0.3),
                (["asks with curiosity", "answers [+detailed explanation]"], 0.5),
                (["asks from the perspective of real life", "answers [+detailed explanation]"], 0.5),
                (["gives specific instructions to the Assistant", "answers [+detailed explanation]"], 0.5),
                (["expresses his/her needs and asks the Assistant for help", "answers [+detailed explanation]"], 0.5),
            ]
                
        }
    }

    local_settings = list(zip(*local_settings[language]['settings']))
    human_word_counts = word_counts['human']
    assistant_word_counts = word_counts['assistant']

    for i in range(rounds):
        if human_word_counts[i] < 10: human_word_counts[i] = 20
        if assistant_word_counts[i] < 100: assistant_word_counts[i] = 200
        requirements = random.choices(local_settings[0], weights=local_settings[1], k=1)[0]
        if i == 0:
            chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）{requirements[0]} <Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words){requirements[0]} <Assistant {i+1}>:"
        else:
            chat_format += f"<Human {i+1}>：（字数要求：{human_word_counts[i]}字）进一步{requirements[0]} <Assistant {i+1}>：" if language == "zh" else f"<Human {i+1}>:(word count: {human_word_counts[i]} words)further {requirements[0]} <Assistant {i+1}>:"
        chat_format  += f"（字数要求：{assistant_word_counts[i]}字）{requirements[1]} " if language == "zh" else f"(word count: {assistant_word_counts[i]} words){requirements[1]} "

    chat_format  += "<end_chat>"
    
    if language == "zh":
        prompt = \
f"""
根据上面的##提供信息##内容以及主题，用中文扩写成一段多轮对话。对话要求你作为聊天机器人Assistant与人类Human进行对话, 并帮助解决Human所提出的要求。Human会以人类的语气对Assistant基于上面的信息（但对话中不能出现”根据以上信息“类似表达）提出多个不一样的问题/要求，且后一个问题/要求是基于前面的对话历史的进一步提问。对于Human提出的每个合理的问题/要求，Assistant要尽可能详细解答，提供更多说明或者举例子。对于Human的不合理（对社会有害、不道德、违法的）请求，Asistant会拒绝回答并解释不能回答的理由，同时给出合理的建议避免这样做。对话的内容要尽可能的符合人类的语言习惯，更加贴合人类日常对话。
#对话规划#示例：“<start_chat><Human 1>:（字数要求：x字）XXX <Assistant 1>：（字数要求：x字）XXX <Human 2>：（字数要求：x字）XXX <Assistant 2>：（字数要求：x字）XXX <end_chat>”，其中“XXX”是对该角色的当前对话内容的要求，“（字数要求：x字）”是Human或者Assistant说话的最低字数要求。必须注意：对话以<start_chat>作为多轮对话的开始，<end_chat>作为多轮对话的结束。
以下对话根据该#对话规划#并遵循规划里面的字数要求进行输出：“{chat_format}”，共{rounds}轮对话。
"""
        prompt += f"以下是{rounds}轮对话："
    else:
        prompt = \
f"""
Based on the ##Provided Information## above and its relevant topic, expand it into a multi-round conversation. The conversation requires you to act as the chatbot Assistant and interact with a human, helping to solve the requests raised by the human. The human will ask multiple various questions/requests to the Assistant based on the information above (but the conversation should not include expressions like "according to the above information"), and the subsequent questions/requests will be a follow-up based on the previous conversation history. For every reasonable question/request posed by Human, Assistant should provide as detailed an answer as possible, offering further explanations or examples. For unreasonable requests from Human (those that are harmful to society, immoral, or illegal), Assistant will refuse to answer and explain the reason for not answering, while also providing reasonable advice to avoid such actions. 
#Conversation Plan# Example: "<start_chat><Human 1>:(Word count requirement: x words)XXX <Assistant 1>: (Word count requirement: x words) XXX <Human 2>:(Word count requirement: x words)XXX <Assistant 2>: (Word count requirement: x words) XXX <end_chat>", "XXX" is the requirement for the current conversation content of that role, and "(Word count requirement: x words)" specifies the minimum word count requirement for utterance of Human or Assistant. It must be noted: the conversation starts with <start_chat> as the beginning of the multi-round conversation and ends with <end_chat> as the end of the multi-round conversation.
The following conversation follows this #Conversation Plan# and word count requirements: "{chat_format}", a total of {rounds} rounds of conversation.
"""
        prompt += f"Here are the {rounds} rounds of conversation:"
        
    user_input += f"##提供信息##\n" if language == "zh" else f"##Provided Information##\n"
    user_input += context['desc']
    user_input += f"\n\n"
    user_input += prompt

    return system_input, user_input, prompt, rounds


def post_process_gpt3_response(response):
    # response = responses[1]["choices"][0]
    response = response["choices"][0]

    try:
        raw_chat = response["message"]["content"]
    except:
        print("ERROR parse!")
        return None

    # if the decoding stops due to length, the last example is likely truncated so we discard it
    if response["finish_reason"] == "length":
        print("WARNING: last example is truncated")
        return None 
    
    if not raw_chat.startswith('<start_chat>') or not raw_chat.endswith('<end_chat>'):
        return None
        
    return raw_chat



# run script
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_filepath", \
                        help="jsonl file to save results to")
    parser.add_argument("--reference_filepaths", nargs="+", \
                        help="jsonl files containing references")
    parser.add_argument("--language", default="zh", \
                        help='Language of the generated dialogue. "zh" for Chinese, "en" for English.', choices=["zh", "en"])
    parser.add_argument("--assistant_word_count", type=int, default=200, \
                        help='Number of words for the assistant to generate')
    parser.add_argument("--human_word_count", type=int, default=100, \
                        help='Number of words for the human to generate')
    parser.add_argument("--num_turn_ratios", nargs="+", type=float, default=[0, 0, 0.5, 0.5, 0], \
                        help='Ratio of the number of turns in the dialogue. The first number is the ratio of 1-turn dialogue, the second number is the ratio of 2-turn dialogue, and so on.')
    parser.add_argument("--api_config", \
                        help="jsonl file containing API keys")
    parser.add_argument("--max_requests_per_minute", type=int, default=1000, \
                        help="maximum number of requests to send per minute")
    parser.add_argument("--max_tokens_per_minute", type=int, default=400_000, \
                        help="maximum number of tokens to send per minute")
    parser.add_argument("--token_encoding_name", default="cl100k_base",
                        help="name of the token encoding to use")
    parser.add_argument("--max_attempts", type=int, default=10,
                        help="maximum number of times to retry a request")
    parser.add_argument("--num_chat_to_generate", type=int, default=1,
                        help="number of dialogues to generate")
    parser.add_argument("--max_tokens", type=int, default=3072,
                        help="maximum number of tokens to send per request")
    parser.add_argument("--logging_level", default=logging.INFO, \
                        help="logging level")

    args = parser.parse_args()

    apis = [json.loads(l) for l in open(args.api_config, "r", encoding="utf8")]
    apis_pool = PriorityQueue(len(apis))
    for i, api in enumerate(apis):
        apis_pool.put(API(time.time(), api))

    checkpoint_dict = {}
    machine_chat_data = []
    # Load the previous generated dialogues
    if os.path.exists(args.save_filepath):
        machine_chat_data = [json.loads(l) for l in open(args.save_filepath, "r", encoding="utf8")]
        print(f"Loaded {len(machine_chat_data)} machine-generated dialogues")
        checkpoint_dict.update({d['reference']: True for d in machine_chat_data})

    request_start = time.time()
    all_reference_data = []
    for s in args.reference_filepaths:
        if s.endswith(".jsonl") or s.endswith(".json"):
            references = [json.loads(l) for l in open(s, "r", encoding="utf8")]
        else:
            # if the seed task is a directory with multiple json files
            references = []
            dir_list = glob.glob(os.path.join(s, "*.json"))
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(s, x)))
            for file in dir_list:
                if file.endswith(".json") or file.endswith(".jsonl"):
                    references.extend([json.loads(l) for l in open(os.path.join(s, file), "r", encoding="utf8")])
                else:
                    with open(os.path.join(s, file), "r", encoding="utf8") as f:
                        references.append({"desc": f.read().strip()})
        
    all_reference_data.extend(references)

    print(f"Loaded {len(all_reference_data)} references!")

    with open("cache_"+args.save_filepath, "w") as f:
        pass

    valid_count = 0
    if args.num_chat_to_generate > len(all_reference_data):
        print(f"Not enough references ({len(all_reference_data)}) to generate the requested number of dialogues ({args.num_chat_to_generate}). Change the number of chat to generate to the number of references ({len(all_reference_data)}).")
        args.num_chat_to_generate = len(all_reference_data)

    progress_bar = tqdm.tqdm(total=args.num_chat_to_generate)
    if machine_chat_data:
        progress_bar.update(len(machine_chat_data))

    reference_data_generator = (i for i in all_reference_data)
    print("Start generating the api input formats.") 
    while valid_count < args.num_chat_to_generate:

        selected_round = [1, 2, 3, 4, 5]
        rounds = random.choices(selected_round, weights=args.num_turn_ratios)[0]  # number of turns in the dialogue       
        assistant_word_counts = (np.random.normal(loc=args.assistant_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        human_word_counts = (np.random.normal(loc=args.human_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        word_counts = {
            "assistant": assistant_word_counts,
            "human": human_word_counts
        }

        total_word_count = sum(assistant_word_counts)

        try:
            context = next(reference_data_generator)
        except StopIteration:
            break

        desc_text = copy.deepcopy(context["desc"])
        if len(checkpoint_dict) > 0:
            if desc_text in checkpoint_dict.keys():
                continue
        
        # truncate the reference text to the required length
        # also discard the reference text that is too short
        if args.language == "zh":
            if len(context["desc"]) < 0.8 * total_word_count:
                continue
            plain_list = context["desc"].split('\n')
            plain = ''
            for p in plain_list:
                plain = plain + p + '\n'
                if len(plain) > total_word_count:
                    break
            context["desc"] = plain
            plain_list = context["desc"].split('。')
            plain = ''
            for p in plain_list:
                temp_plain = plain + p + '。'
                if len(temp_plain) > total_word_count:
                    break
                else:
                    plain = temp_plain
            if len(plain) < 0.8 * total_word_count:
                continue  
        else:
            if len(desc_text.split(' ')) < 0.8 * total_word_count:
                continue
            plain_list = desc_text.split('\n')
            plain_idx = 0
            for p in plain_list:
                plain_idx += len(p) + 1
                if len(desc_text[:plain_idx].split(' ')) > total_word_count:
                    break
            desc_text = desc_text[:plain_idx]
            plain_list = desc_text.split('.')
            plain_idx = 0
            for p in plain_list:
                if len(desc_text[:(plain_idx + len(p) + 1)].split(' ')) > total_word_count:
                    break
                else:
                    plain_idx += len(p) + 1
            if len(desc_text[:plain_idx].split(' ')) < 0.8 * total_word_count:
                continue

        if args.language == "zh":
            context["desc"] = plain
        else:
            context["desc"] = context["desc"][:plain_idx]


        valid_count += 1
        progress_bar.update(1)

        system_input, user_input, prompt, rounds = encode_prompt(context, 
                                                                 rounds=rounds, 
                                                                 word_counts=word_counts,
                                                                 language=args.language
                                                        )
        
        # OpenAI API parameters
        decoding_args = dict(
            temperature=1.0,
            n=1,
            max_tokens=args.max_tokens,  
            top_p=1.0,
            stop=["\n20", "20.", "20."],
        )

        request_json = {}
        api_input = {}
        api_input.update(decoding_args)

        messages=[
            {
                "role": "system",
                "content": system_input
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        api_input["messages"] = messages
        
        request_json["api_input"] = api_input
        request_json["reference"] = context["desc"]
        request_json["prompt"] = prompt
        request_json["meta"] = args.reference_filepaths
        request_json["rounds"] = rounds
        request_json["word_counts"] = word_counts
        request_json["title"] = context["title"] if context.get("title") else ""
        request_json["language"] = args.language

        append_to_jsonl(request_json, filename="cache_"+args.save_filepath)

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath="cache_"+args.save_filepath,
            save_filepath=args.save_filepath,
            apis_pool=apis_pool,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )




