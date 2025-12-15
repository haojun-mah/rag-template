import ollama
import sys
import os

def get_wsl_host_ip():
    """Reads the nameserver from /etc/resolv.conf which represents the Windows host in WSL2."""
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if "nameserver" in line:
                    return line.split()[1]
    except Exception:
        return None
    return None

def test_ollama_connection():
    print("Testing connection to Ollama...")

    # 1. Determine Host
    host = os.getenv("OLLAMA_HOST")
    
    if not host:
        print("OLLAMA_HOST env var not set. Trying localhost...")
        client = ollama.Client(host="http://localhost:11434")
    else:
        print(f"Using OLLAMA_HOST={host}")
        client = ollama.Client(host=host)

    # 2. Try connecting
    try:
        print(f"Attempting to list models from {client._client.base_url}...")
        models_response = client.list()
        models = [m['name'] for m in models_response['models']]
        print(f"Success! Found models: {models}")
    except Exception as e:
        print(f"\nLocalhost failed: {e}")
        
        # If localhost failed, try the WSL Host IP
        wsl_host_ip = get_wsl_host_ip()
        if wsl_host_ip:
            print(f"\nTrying Windows Host IP (WSL Gateway): {wsl_host_ip}...")
            try:
                client = ollama.Client(host=f"http://{wsl_host_ip}:11434")
                models_response = client.list()
                models = [m['name'] for m in models_response['models']]
                print(f"Success! Found models on Windows Host: {models}")
                print(f"\n>>> ACTION REQUIRED: Run this command in your terminal to make it permanent:")
                print(f"export OLLAMA_HOST=http://{wsl_host_ip}:11434")
                return
            except Exception as e2:
                print(f"Windows Host IP failed: {e2}")

        print("\nCRITICAL ERROR: Could not connect to Ollama.")
        print("1. On Windows, set env var: OLLAMA_HOST=0.0.0.0")
        print("2. Restart Ollama on Windows.")
        sys.exit(1)

    # 3. Test generation
    target_model = 'qwen2.5:7b'
    print(f"\nTesting generation with '{target_model}'...")
    
    try:
        response = client.chat(
            model=target_model,
            messages=[{'role': 'user', 'content': 'Say "Hello from Windows!"'}]
        )
        print(f"Response received:\n---\n{response['message']['content']}\n---")
        print("\nOllama connection test PASSED.")
    except Exception as e:
        print(f"Generation failed: {e}")
        if "not found" in str(e):
             print(f"Try running 'ollama pull {target_model}' on your Windows machine.")

if __name__ == "__main__":
    test_ollama_connection()
