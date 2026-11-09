"""
Cross-platform launcher for AI-Powered Fraud Detection Engine
Can be run via: python run.py
"""
import sys
import os
import subprocess
import webbrowser

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    print("=" * 65)
    print(" Vigilance AI Fraud Detection Platform - Indian Banking Edition ")
    print(" Real-time Ensemble Inference: Isolation Forest + Random Forest")
    print(" Currency: Indian Rupees (₹) | Design: heydamir/unslop standard")
    print("=" * 65)

    # Verify key packages
    missing = []
    for pkg, import_name in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("joblib", "joblib")
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")

    print(f"\nServer ready at: http://{host}:{port}")
    if host in ["127.0.0.1", "localhost"]:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass

    import uvicorn
    from backend.server import app
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

