from models.local_models import generate_local_response, load_local_model
from models.prompt_utils import load_messages


class ModelWrapper:
    def __init__(self, args):
        self.args = args
        self.model_name = self.args.model_name
        self.model, self.tokenizer = load_local_model(self.args)

    def generate_response(self, prompt):
        messages = load_messages(self.model_name, prompt)

        return generate_local_response(self.model, self.tokenizer, messages, self.args)
        