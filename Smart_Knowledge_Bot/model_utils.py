import subprocess

def ask_llama3(query, context):
    prompt = f"""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True, text=True
    )
    return result.stdout
