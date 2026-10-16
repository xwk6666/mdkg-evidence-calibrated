# Environment and installation boundary

Frozen replay: Python >=3.10, standard library only. Checked using Python 3.12.14.
This is the preparation/test runtime, not a historical CUDA/training lock.

The selected source imports torch, torchvision, transformers, numpy, Pillow,
huggingface-hub, conditional safetensors, scikit-learn, joblib, openai, optional
neo4j and XLSX-only openpyxl. No dependency implementations are vendored here.
requirements-inference.in lists these names without invented historical pins.
Installing them is optional, is not done by replay, and does not recover historical
weights, missing workflow producers, API versions or dataset preprocessing.

Only the frozen standard-library numerical replay has been executed/verified.
The method entrypoints may download models, invoke a paid LLM service, train/fit
parameters or modify a Neo4j database if deliberately run; do not run them as a
smoke test. No API credentials are included or required for frozen reproduction.
