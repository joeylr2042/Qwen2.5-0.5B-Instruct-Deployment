import torch
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, PreTrainedModel, PreTrainedTokenizer
from threading import Thread

model_store = {}
inference_lock = asyncio.Lock()

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
    title="Qwen2.5 WebSocket Server",
    version="1.0.0",
    lifespan=lifespan
)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("Websocket accepted.")

    model:PreTrainedModel = model_store.get("model")
    tokenizer: PreTrainedTokenizer = model_store.get("tokenizer")

    if not model or not tokenizer:
        await websocket.send_text(
            "No model or tokenizer found."
        )
        await websocket.close()
        return

    # Create a separate messages list for each WebSocket session to store the history
    messages: List[Dict[str, str]] = []

    try:
        while True:
            prompt = await websocket.receive_text()
            async with inference_lock:
                print(f"Lock acquired by {websocket.client}. Processing prompt: {prompt}")

                print(f"Accepted prompt: {prompt}")

                messages.append({"role": "user", "content": prompt})

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # create a streamer queue to put text chunks
                class QueueTextStreamer(TextStreamer):
                    def __init__(self, tokenizer):
                        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        self.text_queue = asyncio.Queue()
                        self.stop_signal = object()

                    def on_finalized_text(self, text: str, stream_end: bool = False):
                        self.text_queue.put_nowait(text)
                        if stream_end:
                            self.text_queue.put_nowait(self.stop_signal)

                streamer = QueueTextStreamer(tokenizer)

                generation_kwargs = {
                    "input_ids": model_inputs["input_ids"],
                    "max_new_tokens": 2048,
                    "streamer": streamer,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                full_response = ""
                while True:
                    chunk = await streamer.text_queue.get()
                    if chunk is streamer.stop_signal:
                        break
                    full_response += chunk
                    await websocket.send_text(chunk)

                if full_response:
                    messages.append({"role": "assistant", "content": full_response})
                await websocket.send_text("<|EOM|>")
                thread.join()
                print(f"Lock released by {websocket.client}.")

    except WebSocketDisconnect:
        print(f"Websocket disconnected: {websocket.client}.")
    except Exception as e:
        print(f"Error: {e}.")
    finally:
        if websocket.client_state != "disconnected":
            await websocket.close()
        print(f'Websocket disconnected: {websocket.client}.')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)