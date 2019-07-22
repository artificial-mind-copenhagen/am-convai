""" This module contains the class neccesary for interacting and communicating
with the gpt-2 conversational model."""
import random
from itertools import chain
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import (
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from train import SPECIAL_TOKENS, build_input_from_segments
from utils import get_dataset_personalities, download_pretrained_model
from loguru import logger


logger.info(f"Starting conv model with gpu: {torch.cuda.is_available()}")


class ConversationalModel:
    """ This will make sure the conversational model is hot and running once
    `InitModel()` has been called. """

    def __init__(self, *args, **kwargs):
        """The following parameters are available:

        * dataset_path: str = ""
        * dataset_cache:str = './dataset_cache'
        * model:str = "gpt"
        * model_checkpoint:str = ""
        * max_history:int = 2
        * device:str = "cuda" if torch.cuda.is_available() else "cpu"
        * no_sample:bool
        * max_length:int = 20
        * min_length:int = 1,
        * seed:int =  42
        * temperature:int = 0.7
        * top_k:int =  0
        * top_p:float =  0.9
        """
        self.args = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataset_path": "",
            "dataset_cache": "./dataset_cache",
            "model": "gpt",
            "model_checkpoint": "",
            "max_history": 2,
            "no_sample": False,
            "max_length": 20,
            "min_length": 1,
            "seed": 42,
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 0.9,
        }
        self.args.update(kwargs)
        # This makes sure the default variables are set and updated in case of
        # keyword arguments to the constructor. If this setup is too complicate
        # maybe update self.__dict__ instead.
        random.seed(self.args["seed"])
        torch.random.manual_seed(self.args["seed"])
        torch.cuda.manual_seed(self.args["seed"])

        # Finally. Status variables:
        self.is_ready = False

    def InitModel(self):
        """ This takes care of loading model/dataset/tokenizing. Can be called
        async or in a seperate thread so as to avoid loooong waiting time"""
        # Start with model and download pretrained if neccesary
        if self.args["model_checkpoint"] == "":
            self.args["model_checkpoint"] = download_pretrained_model()
        # do model setup and tokenize vocabulary
        tokenizer_class = (
            GPT2Tokenizer if self.args["model"] == "gpt2" else OpenAIGPTTokenizer
        )
        self.tokenizer = tokenizer_class.from_pretrained(self.args["model_checkpoint"])

        model_class = (
            GPT2LMHeadModel if self.args["model"] == "gpt2" else OpenAIGPTLMHeadModel
        )
        self.model = model_class.from_pretrained(self.args["model_checkpoint"])
        self.model.to(self.args["device"])
        self.model.eval()
        personalities = get_dataset_personalities(
            self.tokenizer, self.args["dataset_path"], self.args["dataset_cache"]
        )
        self.personality = random.choice(personalities)
        logger.info(
            f"Selected personality: "
            + f"{self.tokenizer.decode(chain(*self.personality))}"
        )
        self.is_ready = True
        logger.info("Model initialized and ready to go")

    def WriteAvailablePersonalities(self, filename="/tmp/personalities.txt"):
        """Lists and decodes all personalities and writes to filename"""
        personalities = get_dataset_personalities(
            self.tokenizer, self.args["dataset_path"], self.args["dataset_cache"]
        )
        with open(filename, "w") as out:
            maxFailures = 5
            failures = 0
            successes = 0
            for p in personalities:
                if failures > maxFailures:
                    logger.error("Too many failures. Aborting personality write")
                    break
                try:
                    out.write(self.tokenizer.decode(chain(*p)))
                    out.write("\n" + ("-" * 50) + "\n")
                    successes += 1
                except:
                    logger.warning(f"Couldn't write personality: {p}")
                    failures += 1

    def Sample(self, history, query):
        """Samples the conversational ai with a history

        History is stored as a list of strings, and will be encoded on the fly
        """
        if not self.is_ready:
            raise Exception("Please ensure initmodel has run before starting")
        encoded = [self.tokenizer.encode(line) for line in history]
        encoded.append(self.tokenizer.encode(query))
        with torch.no_grad():
            out_ids = sample_sequence(
                self.personality, encoded, self.tokenizer, self.model, self.args
            )
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text


def top_filtering(
    logits, top_k=0, top_p=0.0, threshold=-float("Inf"), filter_value=-float("Inf")
):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
        Taken from `interact.py`
    """
    assert (
        logits.dim() == 1
    )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    """A function to sample the network. Taken from `interact.py`"""
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args["max_length"]):
        instance, sequence = build_input_from_segments(
            personality, history, current_output, tokenizer, with_eos=False
        )

        input_ids = torch.tensor(
            instance["input_ids"], device=args["device"]
        ).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=args["device"]
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args["model"]:
            logits = logits[0]
        logits = logits[0, -1, :] / args["temperature"]
        logits = top_filtering(logits, top_k=args["top_k"], top_p=args["top_p"])
        probs = F.softmax(logits, dim=-1)

        prev = (
            torch.topk(probs, 1)[1]
            if args["no_sample"]
            else torch.multinomial(probs, 1)
        )
        if i < args["min_length"] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output
