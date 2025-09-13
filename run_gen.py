from config.retrieve_config import retrieval_args
from retrieve.entry import Retrieval
from dataset.config import DatasetConfig, DatasetNameEnum, ProfileEnum

retrieval_args.vs_col_name = 'quality_preparse_official'
retrieval_args.output_path = 'gen_output'
retrieval_args.top_k = 5
retrieval_args.type = 1

retrieval = Retrieval(DatasetConfig(
    name=DatasetNameEnum.quality,
    profile=ProfileEnum.train
))  

retrieval.retrieve_quality(
    type=retrieval_args.type,
    k=retrieval_args.top_k,
    metric_type=retrieval_args.metric_type
)
print("Done")


