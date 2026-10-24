from locus_core.models import ResolvedModel

BGE_SMALL = ResolvedModel(
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    max_seq_length=512,
    dim=384,
)


class FakeResolver:
    """Stands in for the Hugging Face Hub."""

    def __init__(self, *models):
        self.models = {model.repo_id: model for model in models}

    def resolve(self, repo_id):
        return self.models[repo_id]
