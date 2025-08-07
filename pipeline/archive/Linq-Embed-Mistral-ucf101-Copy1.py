import torch
import torch.nn.functional as F
from torch import Tensor, argmax
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """This code snippet defines a function last_token_pool that extracts the embedding of the last token from a sequence."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# queries =  ['India']
# passages = ['Cricket', 'Basketball', 'volleyball']
queries = ["Soldier"]
# after tokenizing
# Soldier words tokens  {'input_ids': tensor([[    1, 20651,   749,     2]]), 'attention_mask': tensor([[1, 1, 1, 1]])
# The attention_mask is a tensor that tells the model which tokens in a sequence are actual data and which are just padding. This is crucial when processing a batch of sequences that have different lengths. the attention_mask of all ones [1, 1, 1, 1] indicates that all four tokens are real, non-padding tokens. This happens because "Soldier" was tokenized into a sequence of four tokens, and there were no other, longer sentences in the batch that would require padding.

passages = ["PlayingGuitar", "MilitaryParade", "HorseRace"]

passages = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "HammerThrow",
    "Hammering",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpRope",
    "JumpingJack",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PizzaTossing",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "Skiing",
    "Skijet",
    "SkateBoarding",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
]

n_queries = len(queries)  # 3
n_passages = len(passages)  # 3
print("total number of queries = ", n_queries)
print("total number of passages = ", n_passages)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")

max_length = 4096
input_texts = ["Soldier"]
word_dict = tokenizer(
    input_texts,
    max_length=max_length,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
print("Soldier words tokens ", word_dict)
model_output = model(**word_dict)
print((model_output.last_hidden_state))
embeddings = last_token_pool(
    model_output.last_hidden_state, word_dict["attention_mask"]
)
print("embeddings = ", embeddings)


input_texts = [*queries, *passages]
# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    max_length=max_length,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
outputs = model(**batch_dict)
# print((outputs.last_hidden_state))
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
print(
    "embeddings shape is = ", embeddings.shape
)  # torch.Size([102, 4096]) with 1 + 101 words

# Normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (
    embeddings[:1] @ embeddings[1:].T
) * 100  # here embeddings is a huge matrix whose first column is the first word and rest of them are transposed to find similarity, so similarity is only found here, which means above model output and last_token_pool return only the embeddings
# scores = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
print(scores.tolist())
best_idx = torch.argmax(scores)
print("best_idx ", best_idx)
best_match = passages[best_idx]
print("word is ", queries[0])
print("best_match ", best_match)
# [[73.72909545898438, 30.122783660888672], [29.155078887939453, 79.25374603271484]]
