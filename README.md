# Unitype 🧠🎮

**Unitype** is a tiny, from-scratch language model focused on generating **Unity gameplay code in C#**.  
It is designed to run entirely on **CPU-only systems**, making it suitable for low-resource environments, learning, and experimentation.

This project explores how far careful data curation, simple architectures, and strict domain focus can go without relying on large pretrained models or GPUs.

---

## ✨ Features

- 🎮 **Unity-first**: trained exclusively on Unity C# gameplay patterns  
- 🧠 **From scratch**: no pretrained weights, no external models  
- 💻 **CPU-only**: runs comfortably on low-end hardware  
- 📦 **INT8 quantized inference** for reduced memory usage  
- 🔁 **Interactive text generation** via CLI  
- 🧪 Educational and experimental by design  

---

## 🚫 What Unitype Is Not

To set expectations clearly:

- ❌ Not a ChatGPT replacement  
- ❌ Not capable of deep reasoning or debugging  
- ❌ Not production-ready AI tooling  
- ❌ Not trained on Unity documentation or APIs  

Unitype is a **pattern-based language model**, not an assistant with understanding.

---

## 🧩 How It Works (High Level)

Unitype is a character-level language model with a small neural architecture.  
It is trained on repetitive, domain-specific Unity C# code and learns **syntax, structure, and style**, not intent.

The model predicts the next character given a fixed context window, gradually internalizing how Unity scripts are typically written.

---

## 🚀 Getting Started

### Requirements

- Python 3.9+
- PyTorch (CPU build is sufficient)

Install PyTorch (CPU):

```bash
pip install torch
```

Run the model:

```bash
python gamedev_gpt.py
```

You will enter an interactive prompt:

```txt
Unity C# TinyGPT ready.
Type a prompt (or 'exit'):

> public class PlayerMovement
```

The model will generate Unity-style C# code based on your prompt.

---

## 🧠 Training (Optional)

If you want to retrain or fine-tune the model:

- Edit `data.txt` (Unity C# only recommended)
- Adjust hyperparameters inside `gamedev_gpt.py`
- Train on CPU (expect long runtimes)

Training is intentionally simple and transparent.

---

## 📉 Performance Notes

- Training is slow on CPU by design  
- Memory usage stays within a few hundred MB  
- INT8 inference is significantly faster than FP32  
- Best results come from clear, Unity-style prompts  

---

## 🧪 Example Prompt

```csharp
public class EnemyAI : MonoBehaviour
```

Example output (varies):

```csharp
void Update()
{
    Vector3 direction = (player.position - transform.position).normalized;
    transform.Translate(direction * speed * Time.deltaTime);
}
```

---

## 🎯 Why This Project Exists

Unitype exists to explore:

- How small can a useful domain model be?
- What can be learned without GPUs?
- How does data quality outweigh model size?
- What does learning-focused ML look like in practice?

It is intentionally minimal, honest, and hackable.

---

## 🛣️ Future Directions (Non-binding)

- Improved tokenizer (subword or BPE)
- Better sampling strategies (top-k, nucleus)
- Larger Unity-only datasets
- Unity Editor integration
- Retrieval-augmented generation (docs as context)

No timelines. No promises.

---

## 📜 License

MIT License.  
Do whatever you want, just don’t misrepresent what this is.

---

## 🙏 Acknowledgements

Inspired by:

- Tiny language models
- Educational ML projects
- Game development workflows
- Curiosity under hardware constraints

---

*Unitype is an experiment, not a product.*
