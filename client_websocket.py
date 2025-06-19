import asyncio
import aioconsole
import websockets

END_OF_MESSAGE_SIGNAL = "<|EOM|>"

async def chat_client():
    uri = "ws://localhost:8000/ws/chat"

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected")

            assistant_done_replying = asyncio.Event()

            async def receive_messages():
                try:
                    async for message in websocket:
                        if message == END_OF_MESSAGE_SIGNAL:
                            print()
                            assistant_done_replying.set()
                        else:
                            print(message, end="", flush=True)
                except websockets.exceptions.ConnectionClosed:
                    print("\n[系统提示] 服务器连接已断开。")
                    assistant_done_replying.set()

            receive_task = asyncio.create_task(receive_messages())
            assistant_done_replying.set()

            while True:
                await assistant_done_replying.wait()

                prompt = await aioconsole.ainput("\n\nyou: ")

                assistant_done_replying.clear()

                if prompt.lower() == "exit":
                    break

                await websocket.send(prompt)

            receive_task.cancel()
    except ConnectionRefusedError:
        print("Connection refused")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    try:
        asyncio.run(chat_client())
    except KeyboardInterrupt:
        print("\nClient closed")

