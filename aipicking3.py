import torch
import torch.nn as nn
import json
import requests
import random
data=[
    {"text": "Find me a romance manhwa with a strong female lead", "label": 0},
    {"text": "Recommendations for action manga similar to Solo Leveling", "label": 0},
    {"text": "Is there a new chapter for Tower of God?", "label": 0},
    {"text": "Search for horror manga with psychological themes", "label": 0},
    {"text": "Best manhua about cultivation and martial arts", "label": 0},
    {"text": "Looking for a comedy manga set in high school", "label": 0},
    {"text": "Suggest some completed sports manga", "label": 0},
    {"text": "Are there any slice of life manhwa worth reading?", "label": 0},
    {"text": "I need a fantasy manga with dragons", "label": 0},
    {"text": "Top rated historical manhua list", "label": 0},
    {"text": "Manga where the MC is a villain", "label": 0},
    {"text": "Recommend me something like Berserk", "label": 0},
    {"text": "Latest updates on One Piece manga", "label": 0},
    {"text": "Sad manga that will make me cry", "label": 0},
    {"text": "Find a mystery manhwa with good art", "label": 0},
    {"text": "What are some good isekai manga?", "label": 0},
    {"text": "Manhua with cultivation and alchemy", "label": 0},
    {"text": "Supernatural manga recommendations", "label": 0},
    {"text": "Show me urban fantasy manhwa", "label": 0},
    {"text": "Manga with 100+ chapters", "label": 0},
    {"text": "Hidden gem manga that are underrated", "label": 0},
    {"text": "Sci-fi manga list", "label": 0},
    {"text": "Manhwa with a leveling system", "label": 0},
    {"text": "Dark fantasy manga recommendations", "label": 0},
    {"text": "Romantic comedy manhua", "label": 0},
    {"text": "Hi there, how are you today?", "label": 1},
    {"text": "Good morning bot!", "label": 1},
    {"text": "What is your favorite color?", "label": 1},
    {"text": "Tell me a joke", "label": 1},
    {"text": "How was your weekend?", "label": 1},
    {"text": "Hey, what's up?", "label": 1},
    {"text": "Do you like humans?", "label": 1},
    {"text": "Nice to meet you", "label": 1},
    {"text": "Are you a robot or a human?", "label": 1},
    {"text": "Can we be friends?", "label": 1},
    {"text": "What do you think about AI?", "label": 1},
    {"text": "You are very helpful, thank you", "label": 1},
    {"text": "How old are you?", "label": 1},
    {"text": "What time is it there?", "label": 1},
    {"text": "I am feeling a bit tired today", "label": 1},
    {"text": "Hello world!", "label": 1},
    {"text": "Talk to me about life", "label": 1},
    {"text": "Do you have any hobbies?", "label": 1},
    {"text": "You are funny", "label": 1},
    {"text": "I hope you have a great day", "label": 1},
    {"text": "What's the weather like in the cloud?", "label": 1},
    {"text": "Just saying hello", "label": 1},
    {"text": "Can you sing a song?", "label": 1},
    {"text": "Tell me something interesting", "label": 1},
    {"text": "Goodnight bot", "label": 1},
    {"text": "Clear the chat history", "label": 2},
    {"text": "Check system status", "label": 2},
    {"text": "Update the database now", "label": 2},
    {"text": "Reboot the bot", "label": 2},
    {"text": "Change my username in the settings", "label": 2},
    {"text": "Show me my current profile", "label": 2},
    {"text": "Reset my preferences", "label": 2},
    {"text": "Export my data to a CSV file", "label": 2},
    {"text": "Turn off notifications", "label": 2},
    {"text": "Enable dark mode", "label": 2},
    {"text": "Logout of the session", "label": 2},
    {"text": "Sync my account", "label": 2},
    {"text": "Show system logs", "label": 2},
    {"text": "Delete my account permanently", "label": 2},
    {"text": "Check for updates", "label": 2},
    {"text": "Set a reminder for tomorrow", "label": 2},
    {"text": "Stop current process", "label": 2},
    {"text": "Show version information", "label": 2},
    {"text": "Refresh the cache", "label": 2},
    {"text": "Open the help menu", "label": 2},
    {"text": "List all active users", "label": 2},
    {"text": "Disconnect from the server", "label": 2},
    {"text": "Change API key", "label": 2},
    {"text": "Back up the library", "label": 2},
    {"text": "Shutdown the bot", "label": 2}
]
random.shuffle(data)
JINA_TOKEN="your_jina_token_here"
jina_url="https://api.jina.ai/v1/embeddings"
headers={
    "Authorization" :f"Bearer {JINA_TOKEN}",
    "Content-Type" : "application/json"
}
first_Y=[]
first_X=[]
for j in range(len(data)):
    first_X.append(data[j]["text"])
    first_Y.append(data[j]["label"])
secand_X=[]
payload={
    "model":"jina-embeddings-v2-base-en",
    "input":first_X
}
jina_send=requests.post(url=jina_url,headers=headers,json=payload)
jina_get=jina_send.json()
for j in range(len(data)):
    secand_X.append(jina_get["data"][j]["embedding"])
third_X=torch.tensor(secand_X)
third_Y=torch.tensor(first_Y)
train_X=third_X[:60]
train_Y=third_Y[:60]
test_X=third_X[60:]
test_Y=third_Y[60:]
class Brain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent=nn.Linear(768,64)
        self.dropout=nn.Dropout(0.3)
        self.boss=nn.Linear(64,3)
    def forward(self,x):
        x=torch.relu(self.agent(x))
        x=self.dropout(x)
        x=self.boss(x)
        return x
brain_on=Brain()
edit=torch.optim.Adam(brain_on.parameters(),lr=0.01)
creation=nn.CrossEntropyLoss()
for epoch in range (200):
    edit.zero_grad()
    outpute=brain_on(train_X)
    loss=creation(outpute,train_Y)
    loss.backward()
    edit.step()
    if epoch % 25==0:
        print(f" epoch : {epoch}|| lose :{loss}")
brain_on.eval()
with torch.no_grad():
    outpute=brain_on(test_X)
    predictions=torch.argmax(outpute,dim=1)
    correct=(predictions==test_Y).sum().item()
    accuracy=(correct/len(test_Y))*100
    print(f"Test Accuracy: {accuracy}%")
    print(f"AI Predictions: {predictions}")
    print(f"Real Labels:   {test_Y}")
torch.save(brain_on.state_dict() ,"brain3.pth")
labels_map = {0: "Manga/Manhwa", 1: "Social/Chat", 2: "System/Control"}

print("\n--- BRAIN STRESS TEST ---")
print("Type 'quit' to stop.")

while True:
    user_input = input("Enter a message to route: ")
    if user_input.lower() == 'quit':
        break

    payload = {"model": "jina-embeddings-v2-base-en", "input": [user_input]}
    res = requests.post(url=jina_url, headers=headers, json=payload).json()
    embedding = torch.tensor(res["data"][0]["embedding"]).unsqueeze(0) # unsqueeze adds the batch dimension

    brain_on.eval()
    with torch.no_grad():
        raw_output = brain_on(embedding)
        prediction = torch.argmax(raw_output, dim=1).item()
    print(f">> Prediction: {labels_map[prediction]} (Label: {prediction})")
    print("-" * 30)