import typer
import sys
from loguru import logger

from .config import settings
from .data import load_data
from .model import LearningCurveTool
from .visualizer import LCVisualizer

app = typer.Typer(help="Learning Curve Analyzer CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def analyze(dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer")):
    logger.info(f"Generating learning curve for {dataset}...")
    X, y, fn = load_data(dataset)
    tool = LearningCurveTool()
    results = tool.compute(X, y)
    logger.info(f"Train-Val Gap: {results['gap']:.3f}")
    LCVisualizer.plot(results, save_path=settings.plots_dir / "learning_curve.png")
    logger.success("Done!")


if __name__ == "__main__":
    app()
