from functools import partial

import redis
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
import aioredis
import uuid
import json
import uvicorn

from sr_service import SuperResolution

service = SuperResolution()

# 连接 Redis
redis_client = redis.StrictRedis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
app = FastAPI()


# 超分入参
class SuperResolutionRequest(BaseModel):
    image_url: str
    sr_xn: int = 2


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: str = None


class TaskResult(BaseModel):
    task_id: str
    result: str


def perform_super_resolution(task_id, sr_image):
    # 检查任务是否已取消
    task_status = redis_client.get(task_id)
    if task_status is None or json.loads(task_status)["status"] == "canceled":
        return
    # 模拟超分辨率处理，这里可以替换为实际的超分辨率处理代码
    result = service.process(sr_image)

    # 检查任务是否已取消
    task_status = redis_client.get(task_id)
    if task_status is None or json.loads(task_status)["status"] == "canceled":
        return

    redis_client.set(task_id, json.dumps({"status": "completed", "result": result}))


@app.post("/super-resolution/")
def super_resolution(request: SuperResolutionRequest, background_tasks: BackgroundTasks):
    _, sr_image = service.pre_process(request.image_url, request.sr_xn)
    if _:
        raise HTTPException(status_code=401, detail="图片尺寸过大,请调整超分倍率或更换图片")

    # 生成唯一的任务ID
    task_id = str(uuid.uuid4())

    # 添加任务到后台任务中
    background_tasks.add_task(perform_super_resolution, task_id, [sr_image])

    # 存储任务状态到 Redis
    redis_client.set(task_id, json.dumps({"status": "processing", "result": None}))

    # 返回任务ID
    return {"task_id": task_id}


@app.get("/cancel/{task_id}")
def cancel_task(task_id: str):
    # 取消任务
    task_status = redis_client.get(task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_status)
    if task_data["status"] != "processing":
        raise HTTPException(status_code=400, detail="任务已完成或已取消")

    # 更新任务状态为已取消
    redis_client.set(task_id, json.dumps({"status": "canceled", "result": None}))

    return {"message": "任务已取消"}


@app.get("/status/{task_id}", response_model=TaskStatus)
def task_status(task_id: str):
    # 获取任务状态
    task_status = redis_client.get(task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    task_status = json.loads(task_status)
    return {"task_id": task_id, "status": task_status['status']}


@app.get("/result/{task_id}", response_model=TaskResult)
def task_result(task_id: str):
    # 获取任务结果
    task_status = redis_client.get(task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_status)
    if task_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    return {"task_id": task_id, "result": task_data["result"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
