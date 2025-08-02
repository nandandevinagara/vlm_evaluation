from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "frame_00060.jpg"},
            # {"type": "image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            # {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
            {
                "type": "text",
                "text": "You are given an image from a video. Choose exactly one class name that best describes the main action or activity in the image. Return only the class name (no explanation, no additional words). The class name must exactly match one of the following class labels: ApplyEyeMakeup BlowingCandles Drumming  HighJump  MilitaryParade  PlayingTabla   Shotput  TennisSwing ApplyLipstick   BodyWeightSquats  Fencing    HorseRace  Mixing  PlayingViolin  SkateBoarding  ThrowDiscus Archery  Bowling  FieldHockeyPenalty  HorseRiding  MoppingFloor  PoleVault    Skiing   TrampolineJumping BabyCrawling  BoxingPunchingBag  FloorGymnastics HulaHoop Nunchucks  PommelHorse  Skijet Typing BalanceBeam BoxingSpeedBag FrisbeeCatch  IceDancing ParallelBars PullUps  SkyDiving UnevenBars BandMarching BreastStroke FrontCrawl JavelinThrow  PizzaTossing Punch SoccerJuggling  VolleyballSpiking BaseballPitch  BrushingTeeth GolfSwing  JugglingBalls  PlayingCello PushUps SoccerPenalty WalkingWithDog Basketball CleanAndJerk  Haircut JumpingJack PlayingDaf Rafting  StillRings  WallPushups BasketballDunk  CliffDiving  Hammering JumpRope PlayingDhol RockClimbingIndoor  SumoWrestling    WritingOnBoard BenchPress CricketBowling  HammerThrow Kayaking  PlayingFlute RopeClimbing Surfing YoYo Biking CricketShot HandstandPushups  Knitting PlayingGuitar Rowing   Swing Billiards CuttingInKitchen HandstandWalking LongJump  PlayingPiano    SalsaSpin TableTennisShot BlowDryHair Diving  HeadMassage Lunges PlayingSitar ShavingBeard TaiChi",
            },
        ],
    },
]

out = pipe(text=messages, max_new_tokens=5)
print(out[0]["generated_text"][1]["content"])


# WORKS
