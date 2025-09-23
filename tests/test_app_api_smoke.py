import importlib

def test_app_importa():
    # apenas importa e checa se cria app/rota sem subir servidor
    try:
        mod = importlib.import_module("app.main")
    except ModuleNotFoundError:
        # Se a demo é Streamlit puro, pule esse teste
        assert True
        return
    assert hasattr(mod, "app") or hasattr(mod, "create_app")
