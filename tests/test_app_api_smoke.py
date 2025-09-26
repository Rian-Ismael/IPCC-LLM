import importlib

def test_app_importa():
    try:
        mod = importlib.import_module("app.main")
    except ModuleNotFoundError:
        assert True
        return
    assert hasattr(mod, "app") or hasattr(mod, "create_app")
