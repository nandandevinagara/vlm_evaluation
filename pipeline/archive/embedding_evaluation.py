from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Inference mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# UCF101 class labels (you can paste the full list here)
ucf101_labels = [
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

# Model output (you'd replace this with your actual output from the VLM)
vlm_output = "golf"  # Let's say the model outputted this


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token representation as sentence embedding
        return outputs.last_hidden_state[:, 0, :].squeeze(0)


# Get embedding for the model output
output_emb = get_bert_embedding(vlm_output)

# Compute similarities with all UCF101 class labels
similarities = []
for label in ucf101_labels:
    label_emb = get_bert_embedding(label)
    sim = F.cosine_similarity(output_emb, label_emb, dim=0).item()
    similarities.append((label, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Show top matches
print("Top 5 similar classes to VLM output:")
for label, sim in similarities[:5]:
    print(f"{label}: {sim:.4f}")
