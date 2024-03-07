from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import aiohttp
import asyncio
import aioredis
import uuid
import json

app = FastAPI()


# 连接 Redis
async def connect_to_redis():
    redis = await aioredis.create_redis_pool("redis://localhost")
    return redis


redis = asyncio.run(connect_to_redis())


# 超分入参
class SuperResolutionRequest(BaseModel):
    image_url: str
    scale_factor: int = 2


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: str = None


class TaskResult(BaseModel):
    task_id: str
    result: str


async def perform_super_resolution(task_id, image_url, scale_factor):
    # 检查任务是否已取消
    task_status = await redis.hget("tasks", task_id)
    if task_status is None or json.loads(task_status)["status"] == "canceled":
        return

    # 模拟超分辨率处理，这里可以替换为实际的超分辨率处理代码
    await asyncio.sleep(20)

    # 检查任务是否已取消
    task_status = await redis.hget("tasks", task_id)
    if task_status is None or json.loads(task_status)["status"] == "canceled":
        return

    result = f"Processed image from {image_url} with scale factor {scale_factor}"
    await redis.hset("tasks", task_id, json.dumps({"status": "completed", "result": result}))


@app.post("/super-resolution/")
async def super_resolution(request: SuperResolutionRequest, background_tasks: BackgroundTasks):
    # 生成唯一的任务ID
    task_id = str(uuid.uuid4())

    # 添加任务到后台任务中
    background_tasks.add_task(perform_super_resolution, task_id, request.image_url, request.scale_factor)

    # 存储任务状态到 Redis
    await redis.hset("tasks", task_id, json.dumps({"status": "processing", "result": None}))

    # 返回任务ID
    return {"task_id": task_id}


@app.get("/cancel/{task_id}")
async def cancel_task(task_id: str):
    # 取消任务
    task_status = await redis.hget("tasks", task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_status)
    if task_data["status"] != "processing":
        raise HTTPException(status_code=400, detail="任务已完成或已取消")

    # 更新任务状态为已取消
    await redis.hset("tasks", task_id, json.dumps({"status": "canceled", "result": None}))

    return {"message": "任务已取消"}


@app.get("/status/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    # 获取任务状态
    task_status = await redis.hget("tasks", task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    return json.loads(task_status)


@app.get("/result/{task_id}", response_model=TaskResult)
async def task_result(task_id: str):
    # 获取任务结果
    task_status = await redis.hget("tasks", task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_status)
    if task_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    return {"task_id": task_id, "result": task_data["result"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
