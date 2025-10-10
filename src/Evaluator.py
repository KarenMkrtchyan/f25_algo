from utils import model_config
import utils.Setup


if __name__ == "__main__":
    try:
        model = model_config.load_model("gpt2_small")
        print("model_config.py works! Loaded model:", model.cfg.model_name)
    except Exception as e:
        print("model_config.py error:", e)