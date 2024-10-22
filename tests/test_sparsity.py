import whisper
from datasets import load_dataset

data = load_dataset("openslr/librispeech_asr", "clean", split="test")

model = whisper.load_model("medium", mlp_on=True)
model.set_mlp_predictor()
model.set_quant_predictor(2)

for i, d in enumerate(data):
    audio_path = d["audio"]["path"].split("/")[:-1]
    audio_file = d["audio"]["path"].split("/")[-1]
    audio_dir = (audio_file.split("-")[0], audio_file.split("-")[1])
    audio_path = "/".join(audio_path)
    audio_path = f"{audio_path}/LibriSpeech/test-clean/{audio_dir[0]}/{audio_dir[1]}/{audio_file}"

    result = model.transcribe(audio_path)
    print(result["text"])

    if i == 50:
        break
