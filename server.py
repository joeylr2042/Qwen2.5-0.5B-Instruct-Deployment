import asyncio
from contextlib import asynccontextmanager
import time

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, PreTrainedModel, PreTrainedTokenizer
from threading import Thread
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union
from sse_starlette.sse import EventSourceResponse

model_store = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup Phase === #
    model_name = "qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    model_store["model"] = model
    model_store["tokenizer"] = tokenizer
    print("Model and tokenizer loaded successfully.")

    yield

    # === Shutdown Phase === #
    model_store.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(
    title="Qwen2.5-0.5B-Instruct",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="qwen/Qwen2.5-0.5B-Instruct")
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    model: PreTrainedModel = model_store.get("model")
    tokenizer: PreTrainedTokenizer = model_store.get("tokenizer")

    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    text = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        "input_ids": model_inputs.input_ids,
        "max_new_tokens": request.max_tokens or 2048,
    }
    if request.temperature is not None and request.temperature < 1e-6:
        generation_kwargs['do_sample'] = False
    else:
        generation_kwargs['do_sample'] = True
        generation_kwargs['top_p'] = request.top_p
        generation_kwargs['temperature'] = request.temperature

    if request.stream:
        generator = stream_predict(generation_kwargs, model, tokenizer, request.model)
        return EventSourceResponse(generator, media_type="text/event-stream")

    generated_ids = model.generate(**generation_kwargs)

    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response_text),
        finish_reason="stop"
    )

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion"
    )

async def stream_predict(generation_kwargs: Dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_id: str):
    class QueueTextStreamer(TextStreamer):
        def __init__(self, tokenizer, skip_prompt, **kwargs):
            super().__init__(tokenizer, skip_prompt, **kwargs)
            self.token_queue = asyncio.Queue()
            self.stop_signal = "<s>"
            self.timeout = 1

        def on_finalized_text(self, text: str, stream_end: bool = False):
            self.token_queue.put_nowait(text)
            if stream_end:
                self.token_queue.put_nowait(self.stop_signal)

    streamer = QueueTextStreamer(tokenizer, skip_prompt=True)

    thread = Thread(target=model.generate, kwargs={**generation_kwargs, "streamer": streamer})
    thread.start()

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield chunk.model_dump_json(exclude_unset=True)

    while True:
        try:
            new_text = await asyncio.wait_for(streamer.token_queue.get(), timeout=streamer.timeout)
            if new_text == streamer.stop_signal:
                break

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield chunk.model_dump_json(exclude_unset=True)

        except asyncio.TimeoutError:
            if not thread.is_alive():
                break
            else:
                continue

    final_choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    final_chunk = ChatCompletionResponse(model=model_id, choices=[final_choice_data], object="chat.completion.chunk")
    yield final_chunk.model_dump_json(exclude_unset=True)

    yield '[DONE]'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)









