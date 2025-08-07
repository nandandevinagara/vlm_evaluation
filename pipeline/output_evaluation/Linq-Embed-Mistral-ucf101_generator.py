import torch
import torch.nn.functional as F
from torch import Tensor, argmax
from transformers import AutoTokenizer, AutoModel
import json


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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")


input_texts = [*passages]
# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=4096, padding=True, truncation=True, return_tensors="pt"
)
# print('batch_dict =', batch_dict)
outputs = model(
    **batch_dict
)  # dicitonary, hence **, which is used as for loop at the models end, ** is not unpacked automatically, but list is unpacked automatically
# print((outputs.last_hidden_state))
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
print(
    "embeddings shape is = ", embeddings.shape
)  # torch.Size([102, 4096]) with 1 + 101 words

# print('before normalizing ', embeddings)
# Normalize embeddings
embeddings = F.normalize(
    embeddings, p=2, dim=1
)  # normalization is better to reduce values from 1.175 etc to 0.45 etc, still i was getting 48% etc
# print('after normalizing ', embeddings)
embeddings_list = embeddings.tolist()
# Save as JSON
embeddings_dict = dict(zip(input_texts, embeddings_list))

with open("ucf101_embeddings.json", "w") as f:
    json.dump(embeddings_dict, f, indent=2)

scores = (
    embeddings[:1] @ embeddings[1:].T
) * 100  # here embeddings is a huge matrix whose first column is the first word and rest of them are transposed to find similarity, so similarity is only found here, which means above model output and last_token_pool return only the embeddings
# scores = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
print(scores.tolist())
best_idx = torch.argmax(scores)
print("best_idx ", best_idx)
best_match = passages[best_idx]
# print('word is ', passages[0])
print("best_match ", best_match)

print("###############READING from JSON for CROSS VERIFICATION ######################")
with open("ucf101_embeddings.json", "r") as f:
    new_embeddings_dict = json.load(f)

embeddings_values = list(new_embeddings_dict.values())
# Convert the list of lists to a PyTorch tensor
embeddings_tensor = torch.tensor(embeddings_values)
print(embeddings_tensor.shape)
# getting embedding of new word
input_word = ["Salsa"]
batch_dict = tokenizer(
    input_word, max_length=4096, padding=True, truncation=True, return_tensors="pt"
)
outputs = model(**batch_dict)
embeddings_of_a_word = last_token_pool(
    outputs.last_hidden_state, batch_dict["attention_mask"]
)
embeddings_of_a_word = F.normalize(embeddings_of_a_word, p=2, dim=1)


scores = (embeddings_of_a_word @ embeddings_tensor.T) * 100
print("scores again calculated = ", scores)
print(scores.tolist())
best_idx = torch.argmax(scores)
print("best_idx ", best_idx)
best_match = passages[best_idx]
# print('word is ', queries[0])
print("best_match ", best_match)
