import requests

if __name__ == "__main__":
    requests.post(
        "http://localhost:8000/user/signin",
        data={"username": "admin", "password": "admin"},
    )
    reply = requests.post(
        "http://localhost:8000/user/login",
        data={"username": "admin", "password": "admin"},
    )
    token = reply.json()["token"]
    header = {"Authorization": f"Bearer {token}"}
    reply = requests.post(
        "http://localhost:8000/run/lt",
        headers=header,
        json={"rule": "60s", "model": "llama"},
    )
    run_id = reply.json()["run_id"]
    print(run_id)
    for i in range(10):
        reply = requests.get(
            f"http://localhost:8000/run/lt/{run_id}/batch",
            headers=header,
            params={"offset": i * 10, "limit": i * 10 + 10, "sorted": False},
        )
        print(reply.status_code)
        reply = requests.post(
            f"http://localhost:8000/run/lt/{run_id}",
            headers=header,
            json={
                "offset": i * 10,
                "chunk_id": i,
                "data": ["hah"],
                "finished": False,
            },
        )
        print(reply.status_code)

    reply = requests.post(
        f"http://localhost:8000/run/lt/{run_id}",
        headers=header,
        json={
            "offset": 100,
            "chunk_id": 10,
            "data": ["hah"],
            "finished": True,
        },
    )
    reply = requests.get(
        f"http://localhost:8000/run/lt/{run_id}/summary", headers=header
    )
    print(reply.text)
