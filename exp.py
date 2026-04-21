from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",
    temperature=0,
    reasoning=False,
    num_predict=300,
)

result = llm.invoke("Привет. Ответь одной короткой фразой.")
print(result)