# scripts/graph_export.py
from pathlib import Path
import argparse

from src.graph import build_graph

def main():
    p = argparse.ArgumentParser(description="Exporta o grafo do LangGraph como PNG/SVG/Mermaid.")
    p.add_argument("--out", default="artifacts", help="Pasta de saída (default: artifacts)")
    p.add_argument("--fmt", nargs="*", choices=["png", "svg", "mermaid"], default=["png", "svg", "mermaid"],
                   help="Formatos a exportar")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    app = build_graph()
    g = app.get_graph()

    # PNG
    if "png" in args.fmt:
        if hasattr(g, "draw_png"):
            try:
                (outdir / "graph.png").write_bytes(g.draw_png())
                print(f"✔ saved: {outdir / 'graph.png'}")
            except Exception as e:
                print(f"(!) PNG falhou: {e} (instale Graphviz e pygraphviz/pydot)")
        else:
            print("(!) draw_png indisponível nesta versão do LangGraph; tente SVG/Mermaid.")

    # SVG
    if "svg" in args.fmt:
        if hasattr(g, "draw_svg"):
            try:
                (outdir / "graph.svg").write_bytes(g.draw_svg())
                print(f"✔ saved: {outdir / 'graph.svg'}")
            except Exception as e:
                print(f"(!) SVG falhou: {e} (instale Graphviz e pygraphviz/pydot)")
        else:
            print("(!) draw_svg indisponível nesta versão do LangGraph; tente PNG/Mermaid.")

    # Mermaid
    if "mermaid" in args.fmt:
        mmd = None
        for method in ("draw_mermaid", "to_mermaid"):
            if hasattr(g, method):
                try:
                    mmd = getattr(g, method)()
                    break
                except Exception as e:
                    print(f"(!) {method} falhou: {e}")
        if mmd is None:
            mmd = "flowchart LR\n  A[Supervisor] --> B[retrieve]\n  A --> C[answer]\n  A --> D[selfcheck]\n  A --> E[safety]\n  E --> F((END))\n"
        (outdir / "graph.mmd").write_text(mmd, encoding="utf-8")
        print(f"✔ saved: {outdir / 'graph.mmd'}")

if __name__ == "__main__":
    main()
