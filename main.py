from typer import typer

app = typer.Typer()


@app.command
def main():
    pass


methods = ["EnKF", "NN", "QPEns"]


@app.commnd
def train_nn():
    pass


@app.commnd
def assimilate():
    pass
