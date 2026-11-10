from models.train_and_predict import train_online_model, save_model, load_model, predict_campaign

model = train_online_model(scope="us")
path = save_model(model)

model = load_model(path)
p = predict_campaign(model, {
    "name": "Indie Album", "category": "Music", "subcategory": "Rock",
    "location": "Austin, TX", "goal": 5000, "duration": 30,
    "reward levels": "$10,$25,$100,$500", "launch": "2026-03-01",
})
print(p[0])
