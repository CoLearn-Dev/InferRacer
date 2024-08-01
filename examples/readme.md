# Examples

## Benchmark Online vLLM

Firstly, you need to launch a vLLM server on your local machine:

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 
```

After that, run the following command to benchmark the vLLM server:

```bash
python examples/online-vllm.py --time-limit=60 --batch-size=128 --max-tokens=1024
```

## Benchmark Offline vLLM

Just run the script below to benchmark the offline vLLM:

```bash
python examples/offline-vllm.py --time-limit=60 --batch-size=128 --max-tokens=1024 
```

## Benchmark Fleece Serve 

Run following script: 

```bash

python examples/fleece-serve.py --time-limit=60 --batch-size=128 --max-tokens=1024
```

The script will adjust the plan according to number of online workers. For example, if you have 2 workers, the script will benchmark it with 70B model. If you only have 1 worker, the script will benchmark it with 7B model.