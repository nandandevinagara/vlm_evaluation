import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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


# Each query must come with a one-sentence instruction that describes the task
task = "Given a question, retrieve Wikipedia passages that answer the question"
queries = [
    get_detailed_instruct(task, "최초의 원자력 발전소는 무엇인가?"),
    get_detailed_instruct(task, "Who invented Hangul?"),
    get_detailed_instruct(task, "where is india?"),
]
# No need to add instruction for retrieval documents
passages = [
    "현재 사용되는 핵분열 방식을 이용한 전력생산은 1948년 9월 미국 테네시주 오크리지에 설치된 X-10 흑연원자로에서 전구의 불을 밝히는 데 사용되면서 시작되었다. 그리고 1954년 6월에 구소련의 오브닌스크에 건설된 흑연감속 비등경수 압력관형 원자로를 사용한 오브닌스크 원자력 발전소가 시험적으로 전력생산을 시작하였고, 최초의 상업용 원자력 엉더이로를 사용한 영국 셀라필드 원자력 단지에 위치한 콜더 홀(Calder Hall) 원자력 발전소로, 1956년 10월 17일 상업 운전을 시작하였다.",
    "India is in Asia",
    "Hangul was personally created and promulgated by the fourth king of the Joseon dynasty, Sejong the Great.[1][2] Sejong's scholarly institute, the Hall of Worthies, is often credited with the work, and at least one of its scholars was heavily involved in its creation, but it appears to have also been a personal project of Sejong.",
]
n_queries = len(queries)  # 3
n_passages = len(passages)  # 3
print("total number of queries = ", n_queries)
print("total number of passages = ", n_passages)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")

max_length = 4096
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
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

# Normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
print(scores.tolist())
# [[73.72909545898438, 30.122783660888672], [29.155078887939453, 79.25374603271484]]
