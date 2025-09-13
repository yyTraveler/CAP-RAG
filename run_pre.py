from config.pre_config import pre_args
from dataset.config import DatasetConfig, DatasetNameEnum, ProfileEnum
from preprocess.entry import PreProcessor


pre_args.output_path = 'pre_output'
pre_args.vs_col_name = 'quality_preparse'
pre_args.metric_type='L2'
pre_args.concurrency=8

processor = PreProcessor(DatasetConfig(
    name=DatasetNameEnum.quality,
    profile=ProfileEnum.train
))

processor.run()

print("Done.")