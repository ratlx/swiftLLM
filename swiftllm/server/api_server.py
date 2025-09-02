import traceback
import os

import argparse
import asyncio
import fastapi
import uvicorn

import swiftllm

from pydantic import BaseModel
from typing import Optional

import json
from transformers import AutoTokenizer

TIMEOUT_KEEP_ALIVE = 5  # in seconds

app = fastapi.FastAPI()
engine = None

class GenerateRequest(BaseModel):
    prompt: str
    output_len: int
    stream: Optional[bool] = False
    decode: Optional[bool] = True

@app.post("/generate")
async def generate(req: GenerateRequest) -> fastapi.Response:
    """
    Generate completion for the request.

    The request should be a JSON object with fields that match the `RawRequest`
    class plus the following fields:
    - `stream`: boolean, whether to stream the output or not
    - `decode`: boolean, whether to decode the output tokens (default: True)
    """
    raw_request = swiftllm.RawRequest(
        prompt=req.prompt,
        output_len=req.output_len
    )

    if req.stream:
        generator = engine.add_request_and_stream(raw_request)
        async def wrapper():
            output_token_ids = []
            prev_decoded = ""

            async for step_output in generator:
                token_id = step_output.token_id
                output_token_ids.append(token_id)

                if req.decode:
                    # Only decode new tokens
                    try:
                        # Decode current token individually
                        new_token = await engine.tokenization_engine.decode.remote([token_id], skip_special_tokens=False)
                        # Handle special case: some tokens need context from previous token
                        if not new_token and len(output_token_ids) > 1:
                            # Try decoding last two tokens
                            last_two = await engine.tokenization_engine.decode.remote(
                                output_token_ids[-2:], skip_special_tokens=False
                            )
                            # Extract new content from combined result
                            if last_two and len(last_two) > len(prev_decoded):
                                new_token = last_two[len(prev_decoded):]

                        prev_decoded += new_token
                        yield f"{prev_decoded}\n"
                    except Exception:
                        # Fallback to full decoding if incremental fails
                        decoded = await engine.tokenization_engine.decode.remote(output_token_ids, skip_special_tokens=True)
                        prev_decoded = decoded
                        yield f"{decoded}\n"
                else:
                    # Output token ID without decoding
                    yield f"{token_id}\n"

        return fastapi.responses.StreamingResponse(
            wrapper(),
            media_type="text/plain"
        )
    else:
        # TODO Abort the request when the client disconnects
        (_, output_token_ids) = await engine.add_request_and_wait(raw_request)

        response_content = {"output_token_ids": output_token_ids}

        if req.decode:
            decoded = await engine.tokenization_engine.decode.remote(output_token_ids, skip_special_tokens=True)
            response_content["output"] = decoded

        return fastapi.responses.JSONResponse(content=response_content)

@app.post("/v1/completions")
async def openai_completions(req: fastapi.Request):
    req_json = await req.json()
    prompt = req_json["prompt"]
    max_tokens = req_json.get("max_tokens", 128)
    stream = req_json.get("stream", False)

    raw_request = swiftllm.RawRequest(prompt=prompt, output_len=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained("/home/ratlx/xxq/Llama-3.2-1B-Instruct")

    if not stream:
        _, output_token_ids = await engine.add_request_and_wait(raw_request)
        output_text = tokenizer.decode(output_token_ids)
        return fastapi.responses.JSONResponse(
            content={
                "id": "cmpl-swift-test",
                "object": "text_completion",
                "created": 0,
                "model": "swift",
                "choices": [{
                    "text": output_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            }
        )
    else:
        generator = engine.add_request_and_stream(raw_request)

        async def stream_generator():
            async for step_output in generator:
                token_text = tokenizer.decode([step_output.token_id])
                data = {
                    "id": "cmpl-swift-test",
                    "object": "text_completion",
                    "created": 0,
                    "model": "swift",
                    "choices": [{
                        "text": token_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9090)
    swiftllm.EngineConfig.add_cli_args(parser)

    args = parser.parse_args()
    args = vars(args)

    host = args.pop("host")
    port = args.pop("port")
    engine = swiftllm.Engine(swiftllm.EngineConfig(**args))

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    async def main_coroutine():
        await engine.initialize()

        uvicorn_task = asyncio.create_task(uvicorn_server.serve())
        engine_task = asyncio.create_task(engine.start_all_event_loops())

        try:
            await engine_task
        except:  # pylint: disable=broad-except
            traceback.print_exc()
            uvicorn_task.cancel()
            os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
    
    asyncio.run(main_coroutine())
