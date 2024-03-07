import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import List
import time
import uuid

app = FastAPI()


class SuperResolutionModel:
    def __init__(self):
        self.tasks = {}

    def infer(self, task_id, image_path):
        # 模拟推理过程，这里可以替换为你的超分辨率模型推理代码
        start_time = time.time()
        while time.time() - start_time < 20:
            # 检查是否有停止任务请求
            if self.tasks[task_id]["stop_requested"]:
                return None, "Processing (stopped)"
            time.sleep(1)
        return f"Super resolution inference completed for {image_path}", "Completed"

    def stop_task(self, task_id):
        # 停止任务
        if task_id in self.tasks:
            self.tasks[task_id]["stop_requested"] = True

    def get_status(self, task_id):
        # 获取任务状态
        if task_id not in self.tasks:
            return "Not found"
        elif self.tasks[task_id]["status"] == "Completed":
            return "Completed"
        elif self.tasks[task_id]["status"] == "Failed":
            return "Failed"
        else:
            return "Processing"


super_resolution_model = SuperResolutionModel()


@app.post("/tasks/")
async def create_task(background_tasks: BackgroundTasks, image_paths: List[str]):
    task_id = str(uuid.uuid4())
    super_resolution_model.tasks[task_id] = {"status": "Processing", "result": None, "stop_requested": False}

    def infer_task():
        for image_path in image_paths:
            result, status = super_resolution_model.infer(task_id, image_path)
            super_resolution_model.tasks[task_id]["result"] = result
            super_resolution_model.tasks[task_id]["status"] = status

    background_tasks.add_task(infer_task)
    return {"task_id": task_id}


@app.put("/tasks/{task_id}/stop/")
async def stop_task(task_id: str):
    if task_id not in super_resolution_model.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    super_resolution_model.stop_task(task_id)
    return {"message": "Stop request sent for task"}


@app.get("/tasks/{task_id}/status/")
async def get_task_status(task_id: str):
    status = super_resolution_model.get_status(task_id)
    return {"status": status, "result": super_resolution_model.tasks[task_id]["result"] if task_id in super_resolution_model.tasks else None}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=4562)
    # image_url = "test/1705570348_0.png"
    # sr_xn = 2
    # sr_result_url = main(image_url, sr_xn)
    # print(sr_result_url)
    # image = read_image(sr_result_url)
    # Image.open(image).convert("RGB").show()
