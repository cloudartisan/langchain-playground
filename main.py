import argparse
import os
import openai

from langchain.llms import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # Get temperature argument
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for sampling text from the model",
    )
    # Get max tokens
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    return args

def main():
    # Check that OPENAI_API_KEY is set in the environment
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set in environment"
    args = parse_args()

    llm = OpenAI(temperature=args.temperature, max_tokens=args.max_tokens)
    

    



if __name__ == '__main__':
    main()