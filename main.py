import app
import uvicorn

if __name__ == "__main__":
    server_config = app.api_config.get('server', {})
    
    uvicorn.run(
        "app:app",
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
    )
