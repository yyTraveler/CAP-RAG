# CAP-RAG

《基于聚类重组和预解析的检索增强生成方法》论文实验代码

## 实现说明

### 聚类算法

聚类算法的实现参见一下两个文件，这个实现参考了[RAPTOR](https://github.com/parthsarthi03/raptor):
- preprocess/raptor.py
- preprocess/raptor2.py

### 预解析算法
预解析提示词参见：
- interface/chat.py  实际和LLM交互的代码，可以看到完整的提示词
- interface/entity.py 预解析算法的格式化输出结构（Note类型）
- interface/prompts.py 基于pydantic序列化的json格式约束

这里展示下实际结构，chunk变量是分块的文字内容

```
 "Given context {chunk}, read and understand the context above and then parse it in this json format {"query":"","summary":"","reference":"","points":[{"query":"","summary":"","reference":""}]} ."
```


## 复现环境准备
使用conda环境，python=3.12.8，详见requirements.txt


## 数据集下载地址

数据集仓库地址：https://github.com/nyu-mll/quality

数据结构可以参考本仓库dataset目录的sample.json

访问https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/ 查看源文件


```shell
cd dataset/quality/dataset
unzip -oq QuALITY.v1.0.1.zip 
```

## vLLM API服务启动
```
# 默认 apikey 为"EMPTY"字符串
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --host 127.0.0.1 \
    --port 13000 \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 2048
```


## 配置文件并启动复现

根据实际环境，先后修改执行 run_pre.py 和 run_gen.py

最后可以执行如下命令评估结果

```shell
python dataset/quality/evaluator.py --path [保存的结果文件]
```


## 复现说明

由于论文中的实验结果没有设置seed，所以我们将索引阶段产生的collection直接保存了一份（scripts/cap-rag-official.zip），不想花成本跑索引阶段的话可以直接导入到milvus里。

首先，检查 `import_collection.py` 文件里的这两个配置：

- `DST_COL_NAME = "quality_preparse_official"`

- `DATASOURCE_PATH = "cap-rag-official.json"`

之后，依次运行以下命令

```python
cd scripts

unzip -oq cap-rag-official.zip

python import_collection.py

```

导入成功后，可以直接做 run_gen.py 的检索和生成实验
