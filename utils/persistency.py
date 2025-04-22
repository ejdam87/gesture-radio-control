import torch


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Function to save <model> parameters in file <path>
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str) -> None:
    """
    To load the model parameters stored in the file <path> produced by <save_model>, the
    instance of the same model has to be passed in <model> argument. This function then
    fills the model with the learned parameter values.
    """
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device("cpu")))
