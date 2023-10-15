from io import BytesIO

import oss2
import torch


def open_bucket(bucket=None):
    bucket = bucket or "graph-interns"
    auth = oss2.Auth("LTAI5tRAyRmXYFUjHJJ6Kx2y", "ZH2FcMvt4yisg95bvmQvgvlJs2yFam")
    return oss2.Bucket(auth, "http://oss-cn-zhangjiakou.aliyuncs.com", bucket)


def save_checkpoint(bucket, save_path, checkpoint):
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    while True:
        try:
            bucket.put_object(save_path, buffer.getvalue())
        except Exception as inst:
            continue
        break


def load_checkpoint(bucket, model_path):
    buffer = BytesIO(bucket.get_object(model_path).read())
    checkpoint = torch.load(buffer, map_location=lambda storage, loc: storage)
    return checkpoint
