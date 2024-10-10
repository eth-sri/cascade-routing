import os
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from .completion_llms import *


class APIQuery:
    def __init__(self, model, 
                 timeout=30, 
                 temperature=0, 
                 max_tokens=256,
                 return_logprobs=False, 
                 api='openai', 
                 chat=True, 
                 max_retries=20,
                 requests_per_second=30, 
                 check_every_n_seconds=0.1,
                 read_cost=None,  
                 write_cost=None, 
                 **kwargs):
        """
        Initializes an instance of the API class.
        Args:
            model (str): The model to query.
            timeout (int, optional): The timeout value in seconds. Defaults to 30.
            temperature (int, optional): The temperature value. Defaults to 0.
            max_tokens (int, optional): The maximum number of tokens. Defaults to 256.
            return_logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            api (str, optional): The API to be used, one of "openai", "together", "huggingface", "gemini", "claude". Defaults to 'openai'.
            chat (bool, optional): Whether to enable chat mode. Defaults to True.
            max_retries (int, optional): The maximum number of retries. Defaults to 20.
            requests_per_second (int, optional): The number of requests per second. Defaults to 30.
            check_every_n_seconds (float, optional): The interval for checking rate limits. Defaults to 0.1.
            read_cost (float, optional): The cost of read operations. Defaults to None.
            write_cost (float, optional): The cost of write operations. Defaults to None.
            **kwargs: Additional keyword arguments for the API model.
        Returns:
            None
        """
        self.model = model
        self.return_logprobs = return_logprobs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        if read_cost is None:
            self.read_cost = 1
            self.write_cost = 1
        else:
            self.read_cost = read_cost
            self.write_cost = write_cost
        self.max_retries = max_retries
        self.api = api
        self.chat = chat
        self.timeout = timeout
        self.rate_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second, 
                                                check_every_n_seconds=check_every_n_seconds, 
                                                max_bucket_size=requests_per_second)
        self.initialize_api_model()

    def initialize_api_model(self):
        """
        Initializes the API model based on the specified API and chat settings.
        Raises:
            ValueError: If the specified API or chat settings are not supported.
        Returns:
            None
        """
        if self.api == 'openai' and self.chat:
            self.api_model = ChatOpenAI(model=self.model, max_retries=self.max_retries, 
                                        temperature=self.temperature,
                                        timeout=self.timeout, max_tokens=self.max_tokens, 
                                        rate_limiter=self.rate_limiter, 
                                        api_key=os.getenv('OPENAI_API_KEY'), seed=0, 
                                        stream_usage=True, **self.kwargs)
        elif self.api == 'openai' and not self.chat:
            self.api_model = OpenAICompletion(model=self.model, temperature=self.temperature,
                                              max_tokens=self.max_tokens, 
                                              api_key=os.getenv('OPENAI_API_KEY'), seed=0, 
                                              **self.kwargs)
        elif self.api == 'anthropic' and self.chat:
            self.api_model = ChatAnthropic(model_name=self.model, temperature=self.temperature,
                                           timeout=self.timeout, max_tokens=self.max_tokens, 
                                           api_key=os.getenv('ANTHROPIC_API_KEY'), 
                                           stream_usage=True, **self.kwargs)
        elif self.api == 'anthropic' and not self.chat:
            self.api_model = AnthropicLLMCompletion(model_name=self.model, temperature=self.temperature,
                                                    max_tokens=self.max_tokens, **self.kwargs)
        elif self.api == 'together' and self.chat:
            self.api_model = ChatTogether(model=self.model, temperature=self.temperature,
                                          timeout=self.timeout, max_tokens=self.max_tokens, 
                                          api_key=os.getenv('TOGETHER_API_KEY'), 
                                          **self.kwargs)
        elif self.api == 'together' and not self.chat:
            self.api_model = TogetherLLMCompletion(model=self.model, temperature=self.temperature,
                                                    max_tokens=self.max_tokens, **self.kwargs)
        elif self.api == 'google' and self.chat:
            self.api_model = ChatGoogleGenerativeAI(model=self.model, temperature=self.temperature,
                                                    timeout=self.timeout, max_tokens=self.max_tokens, 
                                                    api_key=os.getenv('GOOGLE_API_KEY'), 
                                                    stream_usage=True, **self.kwargs)
        else:
            raise ValueError(f'API {self.api} not supported or chat {self.chat} not supported')
        
    async def run_queries(self, queries):
        """
        Run queries against the API model.

        Args:
            queries (list): A list of queries to be executed. 
                            If chat is enabled, each query is a list of tuples, where each tuple contains ('system', 'ai', 'human') and the query message.
                            If chat is disabled, each query is a string.
        Returns:
            tuple: A tuple containing the outputs of the queries, the detailed cost, and the total cost.
            Total cost is a dictionary containing the input tokens, output tokens, and the total cost.
            Detailed cost is a list of dictionaries containing the same information for each query.

        Raises:
            ValueError: If the query type is not supported.
        """
        retry_api_model = self.api_model.with_retry(stop_after_attempt=self.max_retries)
        if self.chat:
            queries_converted = []
            for query in queries:
                current_query = []
                for query_type, query_message in query:
                    if query_type == 'system':
                        current_query.append(SystemMessage(content=query_message))
                    elif query_type == 'ai':
                        current_query.append(AIMessage(content=query_message))
                    elif query_type == 'human':
                        current_query.append(HumanMessage(content=query_message))
                    else:
                        raise ValueError(f'Query type {query_type} not supported')
                queries_converted.append(current_query)
        else:
            queries_converted = queries

        results = await self._run_with_rate_limiting(retry_api_model.abatch, 
                                                     queries_converted)
        results = self.unify_output_format(results)
        cost, detailed_cost = self.get_cost(results)
        outputs = [result['content'] for result in results]
        if self.return_logprobs:
            logprob_info = self.get_logprobs(results)
            outputs = [(result['content'], logprob) 
                       for result, logprob in zip(results, logprob_info)]
        return outputs, detailed_cost, cost
    
    def unify_output_format(self, results):
        """
        Unifies the output format of the given results across all APIs.

        Args:
            results (list): A list of results.

        Returns:
            list: A list of unified results.

        """
        unified_results = []
        for result in results:
            if not isinstance(result, dict):
                result = dict(result)
            if 'generation_info' in result:
                result = result['generation_info']
            unified_results.append(result)
        return unified_results
    
    def get_logprobs(self, results):
        """
        Retrieves the log probabilities from the given results.
        Parameters:
            results (list): A list of results.
        Returns:
            list: A nested list containing the log probabilities for each result.
        Raises:
            None
        """
        if self.api == 'huggingface':
            logprob_info = [
                [[(key, val) for key, val in result['logprobs'].items()]] for result in results
            ]
        if self.api == 'together':
            logprob_info = []
            
            for result in results:
                logprob_info.append((
                    result['response_metadata']['logprobs']['tokens'], 
                    result['response_metadata']['logprobs']['token_logprobs']
                ))
            logprob_info = [
                [[(token, logprob)] for token, logprob in zip(result[0], result[1])]
                for result in logprob_info
            ]
        elif self.api == 'openai':
            logprob_info = []
            for result in results:
                result_specific_logprob = []
                for token_logprobs in result['response_metadata']['logprobs']['content']:
                    top_logprobs_token = []
                    for token in token_logprobs['top_logprobs']:
                        top_logprobs_token.append((token['token'], token['logprob']))

                    result_specific_logprob.append(top_logprobs_token)
                logprob_info.append(result_specific_logprob)
        return logprob_info

    
    async def _run_with_rate_limiting(self, func, queries):
        """
        Runs the given function with rate limiting.

        Args:
            func (callable): The function to be executed.
            queries (list): The list of queries to be processed.

        Returns:
            list: The results of the function execution.
        """
        results = []
        for i in tqdm(range(0, len(queries), self.rate_limiter.max_bucket_size), 
                      desc='Running queries'):
            batch = queries[i:i + self.rate_limiter.max_bucket_size]
            while not self.rate_limiter.acquire():
                continue
            results.extend(await func(batch))
        return results

    def get_cost(self, results):
        """
        Calculates the cost of the given results.

        Args:
            results (list): A list of results.

        Returns:
            tuple: A tuple containing the total input tokens, total output tokens, total cost, and detailed cost for each result.

        """
        input_tokens = 0
        output_tokens = 0
        detailed_cost = []
        for result in results:
            input_tokens += result['usage_metadata']['input_tokens']
            output_tokens += result['usage_metadata']['output_tokens']
            detailed = result['usage_metadata']
            detailed['cost'] = detailed['input_tokens'] * self.read_cost / 10 ** 6 
            detailed['cost'] += detailed['output_tokens'] * self.write_cost / 10 ** 6
            detailed_cost.append(detailed)

        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': input_tokens * self.read_cost / 10 ** 6 + output_tokens * self.write_cost / 10 ** 6
        }, detailed_cost