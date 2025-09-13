from pydantic import BaseModel, Field
from enum import Enum

class DatasetNameEnum(str, Enum):
    quality = 'quality'
    
class ProfileEnum(str, Enum):
    test = 'test'
    train = 'train'
    dev = 'dev'

class DatasetConfig(BaseModel):
    name: DatasetNameEnum
    profile: ProfileEnum